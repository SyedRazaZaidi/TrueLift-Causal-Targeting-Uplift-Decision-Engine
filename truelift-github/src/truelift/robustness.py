from __future__ import annotations

from typing import Any, Callable

import numpy as np

from truelift.evaluate import qini_curve
from truelift.types import CausalFrame


def inject_confounding(frame: CausalFrame, strength: float = 0.4, seed: int = 1) -> CausalFrame:
    rng = np.random.default_rng(seed)
    u = rng.normal(size=frame.n)
    # hide U from X; bias treatment and outcome
    logits = 0.0 * frame.t + strength * u
    p = 1 / (1 + np.exp(-logits))
    t = (rng.random(frame.n) < np.clip(p + 0.15 * (frame.t > 0), 0.05, 0.95)).astype(int)
    y = np.clip(frame.y + 0.15 * strength * (u > 0), 0, 1) if frame.y.max() <= 1.01 else frame.y + strength * u
    out = frame.subset(np.ones(frame.n, dtype=bool))
    out.t = t
    out.y = y
    out.meta = {**frame.meta, "bias": "confounding", "strength": strength, "randomized": False}
    return out


def inject_noncompliance(frame: CausalFrame, rate: float = 0.25, seed: int = 2) -> CausalFrame:
    rng = np.random.default_rng(seed)
    out = frame.subset(np.ones(frame.n, dtype=bool))
    exp = out.t.copy()
    sneak = rng.random(out.n) < rate
    exp[sneak & (out.t > 0)] = 0
    out.exposure = exp
    out.meta = {**frame.meta, "bias": "noncompliance", "rate": rate}
    return out


def inject_spillover(frame: CausalFrame, rate: float = 0.1, seed: int = 3) -> CausalFrame:
    rng = np.random.default_rng(seed)
    out = frame.subset(np.ones(frame.n, dtype=bool))
    # some controls get a diluted treated outcome
    spill = rng.random(out.n) < rate
    if out.y.max() <= 1.01:
        out.y = np.clip(out.y + spill * (out.t == 0) * 0.08, 0, 1)
    else:
        out.y = out.y + spill * (out.t == 0) * 0.25
    out.meta = {**frame.meta, "bias": "spillover", "rate": rate}
    return out


def robustness_report(frame: CausalFrame, score_fn: Callable[[CausalFrame], np.ndarray]) -> dict[str, Any]:
    """Re-rank models' Qini under planted biases (uses provided score on original fit — caller passes tau)."""
    base_score = score_fn(frame)
    scenarios = {
        "clean": frame,
        "confounding": inject_confounding(frame),
        "noncompliance": inject_noncompliance(frame),
        "spillover": inject_spillover(frame),
    }
    rows = []
    for name, fr in scenarios.items():
        q = qini_curve(fr.y, fr.t, base_score)
        rows.append({"scenario": name, "auuc": q["auuc"], "normalized_auuc": q["normalized_auuc"]})
    return {"scenarios": rows}
