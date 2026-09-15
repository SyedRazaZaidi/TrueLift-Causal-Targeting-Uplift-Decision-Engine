from __future__ import annotations

from typing import Any

import numpy as np

from truelift.types import CausalFrame


def qini_curve(y: np.ndarray, t: np.ndarray, score: np.ndarray, n_grid: int = 21) -> dict[str, Any]:
    """
    Incremental outcomes vs fraction targeted, ranking by score (uplift).
    Uses the two-sample estimator: among the top-u, treated mean - control mean,
    times number targeted (Radcliffe).
    """
    t = (t > 0).astype(int)
    n = len(y)
    order = np.argsort(-score)
    y_s, t_s = y[order], t[order]
    fracs = np.linspace(0, 1, n_grid)
    qini = []
    random = []
    ate = float(y[t == 1].mean() - y[t == 0].mean()) if t.min() != t.max() else 0.0
    for f in fracs:
        k = max(1, int(round(f * n))) if f > 0 else 0
        if k == 0:
            qini.append(0.0)
            random.append(0.0)
            continue
        yt, tt = y_s[:k], t_s[:k]
        if tt.min() == tt.max():
            inc = 0.0
        else:
            inc = float(yt[tt == 1].mean() - yt[tt == 0].mean()) * k
        qini.append(inc)
        random.append(ate * k)
    treat_all = ate * n
    auuc = float(np.trapezoid(qini, fracs))
    auuc_rand = float(np.trapezoid(random, fracs))
    denom = abs(auuc_rand) + 1e-9
    return {
        "fracs": fracs.tolist(),
        "qini": [float(v) for v in qini],
        "random": [float(v) for v in random],
        "auuc": auuc,
        "auuc_random": auuc_rand,
        "normalized_auuc": float((auuc - auuc_rand) / denom),
        "treat_all": float(treat_all),
        "ate": ate,
    }


def qini_bands(y: np.ndarray, t: np.ndarray, score: np.ndarray, n_boot: int = 80, seed: int = 0, n_grid: int = 21) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    n = len(y)
    curves = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        curves.append(qini_curve(y[idx], t[idx], score[idx], n_grid=n_grid)["qini"])
    M = np.array(curves)
    return {
        "qini_lo": np.quantile(M, 0.025, axis=0).tolist(),
        "qini_hi": np.quantile(M, 0.975, axis=0).tolist(),
    }


def uplift_at_k(y: np.ndarray, t: np.ndarray, score: np.ndarray, ks: tuple[float, ...] = (0.1, 0.3, 0.5)) -> dict[str, float]:
    t = (t > 0).astype(int)
    n = len(y)
    order = np.argsort(-score)
    out = {}
    for f in ks:
        k = max(1, int(round(f * n)))
        yt, tt = y[order][:k], t[order][:k]
        if tt.min() == tt.max():
            u = 0.0
        else:
            u = float(yt[tt == 1].mean() - yt[tt == 0].mean())
        out[f"uplift@{int(f*100)}"] = u
    return out


def policy_value_dr(y: np.ndarray, t: np.ndarray, e: np.ndarray, mu0: np.ndarray, mu1: np.ndarray, pi: np.ndarray) -> float:
    """Doubly robust value of a binary policy pi in {0,1}."""
    t = (t > 0).astype(float)
    e = np.clip(e, 1e-3, 1 - 1e-3)
    pi = pi.astype(float)
    v = pi * (mu1 + t * (y - mu1) / e) + (1 - pi) * (mu0 + (1 - t) * (y - mu0) / (1 - e))
    return float(np.mean(v))


def pehe(tau_hat: np.ndarray, tau_true: np.ndarray) -> float:
    a = tau_hat.reshape(-1, tau_hat.shape[-1] if tau_hat.ndim > 1 else 1)[:, 0]
    b = tau_true.reshape(-1, tau_true.shape[-1] if tau_true.ndim > 1 else 1)[:, 0]
    return float(np.sqrt(np.mean((a - b) ** 2)))


def ate_error(tau_hat: np.ndarray, tau_true: np.ndarray) -> float:
    a = tau_hat.reshape(-1, tau_hat.shape[-1] if tau_hat.ndim > 1 else 1)[:, 0]
    b = tau_true.reshape(-1, tau_true.shape[-1] if tau_true.ndim > 1 else 1)[:, 0]
    return float(abs(a.mean() - b.mean()))


def bootstrap_auuc(y: np.ndarray, t: np.ndarray, score: np.ndarray, n_boot: int = 200, seed: int = 0) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    n = len(y)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        vals.append(qini_curve(y[idx], t[idx], score[idx])["auuc"])
    vals = np.array(vals)
    return {
        "auuc_mean": float(vals.mean()),
        "auuc_lo": float(np.quantile(vals, 0.025)),
        "auuc_hi": float(np.quantile(vals, 0.975)),
    }


def dominance_test(y: np.ndarray, t: np.ndarray, score_a: np.ndarray, score_b: np.ndarray, n_boot: int = 200, seed: int = 1) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    n = len(y)
    diff = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        ua = qini_curve(y[idx], t[idx], score_a[idx])["auuc"]
        ub = qini_curve(y[idx], t[idx], score_b[idx])["auuc"]
        diff.append(ua - ub)
    d = np.array(diff)
    return {
        "delta_auuc_mean": float(d.mean()),
        "delta_lo": float(np.quantile(d, 0.025)),
        "delta_hi": float(np.quantile(d, 0.975)),
        "beats": bool(np.quantile(d, 0.025) > 0),
    }


def response_score(y: np.ndarray, t: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Naive P(Y=1|X) using treated-only as a competing policy score."""
    from sklearn.linear_model import LogisticRegression, LinearRegression

    if len(np.unique(y)) <= 2:
        m = LogisticRegression(max_iter=300)
        m.fit(X, (y > 0).astype(int))
        return m.predict_proba(X)[:, 1]
    m = LinearRegression()
    m.fit(X, y)
    return m.predict(X)


def ship_gates(metrics: dict[str, Any]) -> list[dict[str, Any]]:
    gates = []
    auuc = metrics.get("champion_auuc", 0)
    rand = metrics.get("random_auuc", 0)
    gates.append({"id": "beats_random", "ok": auuc >= rand, "detail": f"AUUC {auuc:.4f} vs random {rand:.4f}"})
    beats_resp = metrics.get("beats_response", False)
    gates.append({"id": "beats_response_model", "ok": bool(beats_resp), "detail": "Champion AUUC vs conversion-model ranking"})
    overlap_ok = metrics.get("positivity_ok", True)
    gates.append({"id": "positivity", "ok": bool(overlap_ok), "detail": "Overlap / common support"})
    gates.append({"id": "not_inverted", "ok": auuc >= rand * 0.5, "detail": "Qini not inverted vs random"})
    return gates
