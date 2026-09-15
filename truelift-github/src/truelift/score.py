from __future__ import annotations

from typing import Any

import joblib
import numpy as np
import pandas as pd

from truelift.data import encode_like
from truelift.paths import ARTIFACTS
from truelift.policy import budget_policy, knapsack_policy


def load_scorer() -> dict[str, Any]:
    path = ARTIFACTS / "scorer.joblib"
    if not path.exists():
        raise FileNotFoundError("No scorer.joblib — train first.")
    return joblib.load(path)


def score_dataframe(df: pd.DataFrame, budget: float = 0.3, mode: str = "knapsack", conformal: bool = False) -> pd.DataFrame:
    pack = load_scorer()
    model = pack["model"]
    names = pack["feature_names"]
    arms = pack["arm_names"]
    revenue = float(pack.get("revenue", 12.0))
    contact = float(pack.get("contact_cost", 1.0))
    drop = {"treatment", "visit", "conversion", "spend", "id", "split", "exposure", "group", "cost", "dose", "true_tau", "assign", "tau", "decision"}
    X = encode_like(df, names, drop)
    mu, tau = model.predict(X)
    score = tau[:, 0] if tau.ndim > 1 else tau
    cost = df["cost"].to_numpy(dtype=float) if "cost" in df.columns else np.full(len(df), contact)
    lo = None
    if conformal and "conformal_width" in pack:
        lo = tau - float(pack["conformal_width"])
    if mode == "topk":
        assign = budget_policy(tau, budget, 0, lo)
        spent = float((assign > 0).sum() * contact)
    else:
        kn = knapsack_policy(tau, cost, budget * float(cost.sum()), revenue, lo=lo)
        assign = kn["assign"]
        spent = kn["spent"]
    arm_out = ["skip" if a == 0 else (arms[min(a, len(arms) - 1)] if a < len(arms) else f"arm_{a}") for a in assign]
    out = df.copy()
    out["tau"] = score
    out["mu0"] = mu[:, 0]
    out["mu1"] = mu[:, min(1, mu.shape[1] - 1)]
    out["assign"] = assign
    out["decision"] = arm_out
    out["budget"] = budget
    out.attrs["spent"] = spent
    return out
