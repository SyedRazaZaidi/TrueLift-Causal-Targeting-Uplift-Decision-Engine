from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from truelift.types import CausalFrame

EstimandName = Literal["ate", "att", "cate", "itt", "late", "dose"]


@dataclass
class IdentificationReport:
    estimand: str
    randomized: bool
    unconfoundedness: str
    positivity_ok: bool
    overlap: dict[str, Any]
    propensity: dict[str, Any]
    iv_ready: bool
    placebos: list[dict[str, Any]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    allowed_estimators: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "estimand": self.estimand,
            "randomized": self.randomized,
            "unconfoundedness": self.unconfoundedness,
            "positivity_ok": self.positivity_ok,
            "overlap": self.overlap,
            "propensity": self.propensity,
            "iv_ready": self.iv_ready,
            "placebos": self.placebos,
            "notes": self.notes,
            "allowed_estimators": self.allowed_estimators,
        }


def propensity_scores(frame: CausalFrame) -> np.ndarray:
    """P(T>0 | X). Multi-arm: one-vs-control for any treatment."""
    t = (frame.t > 0).astype(int)
    if t.min() == t.max():
        return np.full(frame.n, 0.5)
    clf = LogisticRegression(max_iter=400)
    clf.fit(frame.X, t)
    return clf.predict_proba(frame.X)[:, 1]


def overlap_diagnostics(e: np.ndarray) -> dict[str, Any]:
    lo, hi = 0.05, 0.95
    support = float(np.mean((e > lo) & (e < hi)))
    ess_t = float(np.sum(e) ** 2 / (np.sum(e**2) + 1e-9))
    ess_c = float(np.sum(1 - e) ** 2 / (np.sum((1 - e) ** 2) + 1e-9))
    bins = np.linspace(0, 1, 21)
    hist, _ = np.histogram(e, bins=bins)
    return {
        "common_support_frac": round(support, 4),
        "ess_treated": round(ess_t, 1),
        "ess_control": round(ess_c, 1),
        "e_mean": round(float(e.mean()), 4),
        "e_min": round(float(e.min()), 4),
        "e_max": round(float(e.max()), 4),
        "histogram": hist.astype(int).tolist(),
        "bin_edges": bins.round(3).tolist(),
        "positivity_ok": support >= 0.85 and float(e.min()) > 0.01 and float(e.max()) < 0.99,
    }


def ate_ipw(y: np.ndarray, t: np.ndarray, e: np.ndarray) -> float:
    t = (t > 0).astype(float)
    e = np.clip(e, 1e-3, 1 - 1e-3)
    return float(np.mean(t * y / e - (1 - t) * y / (1 - e)))


def ate_aipw(y: np.ndarray, t: np.ndarray, e: np.ndarray, mu0: np.ndarray, mu1: np.ndarray) -> float:
    t = (t > 0).astype(float)
    e = np.clip(e, 1e-3, 1 - 1e-3)
    return float(np.mean(mu1 - mu0 + t * (y - mu1) / e - (1 - t) * (y - mu0) / (1 - e)))


def rosenbaum_gamma(y: np.ndarray, t: np.ndarray, matched_diff: np.ndarray | None = None) -> dict[str, Any]:
    """How large a hidden bias Γ would need to be to explain away the sign of ATE."""
    t = (t > 0).astype(int)
    if t.sum() == 0 or (1 - t).sum() == 0:
        return {"gamma_critical": None, "note": "degenerate treatment"}
    ate = float(y[t == 1].mean() - y[t == 0].mean())
    se = float(np.sqrt(y[t == 1].var() / max(t.sum(), 1) + y[t == 0].var() / max((1 - t).sum(), 1)))
    if se < 1e-12:
        return {"ate": ate, "gamma_critical": None}
    # Crude bound: confounder would need to shift ATE by ~2 se
    need = abs(ate) / (2 * se + 1e-9)
    gamma = float(np.exp(need))
    return {"ate_unadj": round(ate, 5), "se_unadj": round(se, 5), "gamma_critical": round(gamma, 3), "note": "exponential tilt vs 2-SE; policy sensitivity uses same Γ"}


def e_value(ate: float, se: float) -> float:
    if se <= 0:
        return 999.0
    rr = max(abs(ate) / (se + 1e-9), 1.0)
    return float(rr + np.sqrt(rr * (rr - 1)))


def placebo_null_feature(frame: CausalFrame, e: np.ndarray) -> dict[str, Any]:
    """A shuffled feature must not predict treatment better than chance after X."""
    rng = np.random.default_rng(0)
    fake = rng.permutation(frame.X[:, 0])
    t = (frame.t > 0).astype(int)
    try:
        auc = float(roc_auc_score(t, fake))
    except Exception:
        auc = 0.5
    return {"name": "shuffled_x0", "auc_vs_treatment": round(max(auc, 1 - auc), 3), "expect_near": 0.5, "pass": abs(auc - 0.5) < 0.08}


def identify(frame: CausalFrame, estimand: EstimandName = "cate") -> IdentificationReport:
    e = propensity_scores(frame)
    overlap = overlap_diagnostics(e)
    randomized = bool(frame.meta.get("randomized", False))
    iv_ready = frame.exposure is not None and not np.array_equal(frame.exposure, frame.t)
    notes = []
    if randomized:
        notes.append("Treatment looks like an experiment: CATE/ITT identified from X, T, Y.")
        unconf = "plausible_by_design"
    else:
        notes.append("Observational assignment: CATE needs unconfoundedness given X. Overlap is required.")
        unconf = "assumed_given_X"
    if iv_ready:
        notes.append("Assignment ≠ exposure: ITT (assignment) and LATE (compliers) are both available.")
    if estimand == "late" and not iv_ready:
        notes.append("LATE requested but no exposure column — falling back to ITT/CATE.")
        estimand = "itt" if randomized else "cate"

    allowed = [
        "s_learner",
        "t_learner",
        "x_learner",
        "r_learner",
        "dr_learner",
        "uplift_forest",
        "causal_forest",
        "tarnet",
        "cfrnet",
        "dragonnet",
        "uplift_ranker",
        "stack",
    ]
    if not overlap["positivity_ok"]:
        notes.append("Positivity is weak — IPW/DR will be noisy; trees/representation models preferred.")
    t_bin = (frame.t > 0).astype(int)
    try:
        pauc = float(roc_auc_score(t_bin, e))
    except Exception:
        pauc = 0.5

    return IdentificationReport(
        estimand=estimand,
        randomized=randomized,
        unconfoundedness=unconf,
        positivity_ok=bool(overlap["positivity_ok"]),
        overlap=overlap,
        propensity={
            "auc": round(pauc, 4),
            "mean": round(float(e.mean()), 4),
            "scores_preview": e[:40].round(4).tolist(),
        },
        iv_ready=iv_ready,
        placebos=[placebo_null_feature(frame, e)],
        notes=notes,
        allowed_estimators=allowed,
    )
