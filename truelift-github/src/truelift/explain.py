from __future__ import annotations

from typing import Any

import numpy as np

from truelift.types import CausalFrame


def shap_like_tau(model, X: np.ndarray, feature_names: list[str], i: int, n_ref: int = 40) -> list[dict[str, Any]]:
    """Interventional feature contribution to tau via finite differences vs mean baseline."""
    rng = np.random.default_rng(0)
    n = min(n_ref, len(X))
    ref_idx = rng.choice(len(X), n, replace=False)
    base = X[ref_idx].mean(axis=0)
    x = X[i].copy()
    _, tau0 = model.predict(base.reshape(1, -1))
    t0 = float(tau0.reshape(-1)[0])
    contribs = []
    for j, name in enumerate(feature_names):
        xj = base.copy()
        xj[j] = x[j]
        _, tauj = model.predict(xj.reshape(1, -1))
        contribs.append({"feature": name, "delta_tau": float(tauj.reshape(-1)[0] - t0)})
    contribs.sort(key=lambda r: abs(r["delta_tau"]), reverse=True)
    return contribs[:12]


def person_dossier(
    frame: CausalFrame,
    i: int,
    mu: np.ndarray,
    tau: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
    assign: np.ndarray,
    response_score: np.ndarray,
    contribs: list[dict[str, Any]],
) -> dict[str, Any]:
    arm = int(np.argmax(tau[i])) if tau.ndim > 1 else 0
    tcol = tau[i, arm] if tau.ndim > 1 else float(tau[i])
    loc = lo[i, arm] if lo.ndim > 1 else float(lo[i])
    hic = hi[i, arm] if hi.ndim > 1 else float(hi[i])
    mu0 = float(mu[i, 0])
    mu1 = float(mu[i, arm + 1] if mu.shape[1] > arm + 1 else mu[i, min(1, mu.shape[1] - 1)])
    kind = "persuadable"
    if tcol < -0.01:
        kind = "sleeping_dog"
    elif abs(tcol) <= 0.01 and mu0 >= 0.15:
        kind = "sure_thing"
    elif abs(tcol) <= 0.01 and mu0 < 0.08:
        kind = "lost_cause"
    return {
        "id": int(frame.ids[i]),
        "index": i,
        "kind": kind,
        "features": {n: float(frame.X[i, j]) for j, n in enumerate(frame.feature_names)},
        "mu0": mu0,
        "mu1": mu1,
        "tau": float(tcol),
        "tau_lo": float(loc),
        "tau_hi": float(hic),
        "assign": int(assign[i]),
        "response_score": float(response_score[i]),
        "response_would_treat": bool(response_score[i] >= np.quantile(response_score, 0.7)),
        "we_treat": bool(assign[i] > 0),
        "contributions": contribs,
        "group": int(frame.group[i]) if frame.group is not None else None,
        "true_tau": float(frame.true_tau[i, arm]) if frame.true_tau is not None else None,
    }


def recourse(model, x: np.ndarray, feature_names: list[str], threshold: float = 0.02) -> list[dict[str, Any]]:
    """What single-feature nudge would push tau over the treat threshold."""
    _, tau = model.predict(x.reshape(1, -1))
    t0 = float(tau.reshape(-1)[0])
    ideas = []
    for j, name in enumerate(feature_names):
        for delta in (0.5, 1.0, -0.5, -1.0):
            x2 = x.copy()
            x2[j] = x2[j] + delta
            _, t2 = model.predict(x2.reshape(1, -1))
            nt = float(t2.reshape(-1)[0])
            if t0 < threshold <= nt:
                ideas.append({"feature": name, "delta": delta, "tau_before": t0, "tau_after": nt})
                break
    return ideas[:8]


def segment_profiles(frame: CausalFrame, tau: np.ndarray, top_frac: float = 0.1) -> dict[str, Any]:
    score = tau[:, 0] if tau.ndim > 1 else tau
    k = max(1, int(len(score) * top_frac))
    top = np.argsort(-score)[:k]
    bot = np.argsort(score)[:k]
    def summary(idx):
        return {
            "n": int(len(idx)),
            "tau_mean": float(score[idx].mean()),
            "feature_means": {n: float(frame.X[idx, j].mean()) for j, n in enumerate(frame.feature_names[:12])},
        }
    return {"persuadable_top": summary(top), "sleeping_dog_bottom": summary(bot)}
