from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.tree import DecisionTreeClassifier

from truelift.types import CausalFrame


def rank_by_tau(tau: np.ndarray, arm: int = 0) -> np.ndarray:
    col = tau[:, arm] if tau.ndim > 1 else tau
    return np.argsort(-col)


def budget_policy(tau: np.ndarray, budget_frac: float, arm: int = 0, lo: np.ndarray | None = None) -> np.ndarray:
    """Treat top fraction on chosen arm. If lo provided, require lo>0 (conformal)."""
    n = tau.shape[0]
    k = max(1, int(round(n * budget_frac)))
    score = tau[:, arm] if tau.ndim > 1 else tau
    if lo is not None:
        loc = lo[:, arm] if lo.ndim > 1 else lo
        score = np.where(loc > 0, score, -np.inf)
    order = np.argsort(-score)
    treat = np.zeros(n, dtype=int)
    # skip -inf
    picked = 0
    for i in order:
        if score[i] == -np.inf:
            break
        treat[i] = arm + 1
        picked += 1
        if picked >= k:
            break
    return treat  # 0 = nothing, else arm index+? wait we use arm id: 1..K


def knapsack_policy(
    tau: np.ndarray,
    cost: np.ndarray,
    budget_total: float,
    revenue: float,
    lo: np.ndarray | None = None,
    group: np.ndarray | None = None,
    fairness_slack: float = 1.0,
) -> dict[str, Any]:
    """
    Multi-choice knapsack (greedy by value/cost).
    Each person: pick best arm (including none) then greedy fill.
    Fairness: treatment rate difference across binary group <= slack.
    """
    n, n_treat = tau.shape if tau.ndim > 1 else (len(tau), 1)
    if tau.ndim == 1:
        tau = tau[:, None]
    value = tau * revenue
    if lo is not None:
        loc = lo if lo.ndim > 1 else lo[:, None]
        value = np.where(loc > 0, value, 0.0)

    # best arm per person
    best_arm = np.argmax(value, axis=1)
    best_val = value[np.arange(n), best_arm]
    best_val = np.where(best_val > 0, best_val, 0.0)
    c = np.asarray(cost, dtype=float)
    if c.shape[0] != n:
        c = np.full(n, float(c.reshape(-1)[0]))
    assign = np.zeros(n, dtype=int)
    spent = 0.0
    # Equal costs → exact: take top-k by value
    if float(np.nanmax(c) - np.nanmin(c)) <= 1e-9 * max(float(np.nanmean(c)), 1.0):
        unit = float(c.mean()) if c.mean() > 0 else 1.0
        k = int(budget_total / unit)
        order = np.argsort(-best_val)
        picked = 0
        for i in order:
            if best_val[i] <= 0 or picked >= k:
                break
            assign[i] = int(best_arm[i] + 1)
            spent += unit
            picked += 1
    else:
        roi = best_val / np.clip(c, 1e-6, None)
        order = np.argsort(-roi)
        for i in order:
            if best_val[i] <= 0:
                continue
            if spent + c[i] > budget_total:
                continue
            assign[i] = int(best_arm[i] + 1)
            spent += c[i]

    # Fairness repair: if treated rates differ too much, drop from over-represented group
    if group is not None and fairness_slack < 0.99:
        assign = _balance_rates(assign, group, fairness_slack)

    return {
        "assign": assign,
        "spent": float(spent),
        "n_treated": int((assign > 0).sum()),
        "budget_total": float(budget_total),
        "expected_value": float(best_val[assign > 0].sum()) if (assign > 0).any() else 0.0,
    }


def _balance_rates(assign: np.ndarray, group: np.ndarray, slack: float) -> np.ndarray:
    treated = assign > 0
    if treated.sum() < 10:
        return assign
    g0 = group == 0
    g1 = group == 1
    r0 = treated[g0].mean() if g0.any() else 0
    r1 = treated[g1].mean() if g1.any() else 0
    out = assign.copy()
    if abs(r0 - r1) <= slack:
        return out
    # drop from higher-rate group until slack met
    high = 0 if r0 > r1 else 1
    idx = np.where((out > 0) & (group == high))[0]
    rng = np.random.default_rng(0)
    rng.shuffle(idx)
    for i in idx:
        out[i] = 0
        treated = out > 0
        r0 = treated[g0].mean() if g0.any() else 0
        r1 = treated[g1].mean() if g1.any() else 0
        if abs(r0 - r1) <= slack:
            break
    return out


def dose_policy(tau_dose: np.ndarray, doses: np.ndarray, budget_total: float, unit_cost: float) -> dict[str, Any]:
    """
    tau_dose: (n, n_doses) incremental outcome vs dose 0.
    Pick a dose per person (greedy value/cost).
    """
    n, k = tau_dose.shape
    cost = doses.reshape(1, -1) * unit_cost
    value = tau_dose
    roi = value / np.clip(cost, 1e-6, None)
    best = np.argmax(roi, axis=1)
    best_val = value[np.arange(n), best]
    best_cost = cost[0, best]
    order = np.argsort(-(best_val / np.clip(best_cost, 1e-6, None)))
    chosen = np.zeros(n, dtype=int)
    spent = 0.0
    for i in order:
        if best_val[i] <= 0:
            continue
        if spent + best_cost[i] > budget_total:
            continue
        chosen[i] = int(best[i])
        spent += best_cost[i]
    return {"dose_index": chosen, "doses": doses.tolist(), "spent": float(spent)}


def distill_policy_tree(X: np.ndarray, assign: np.ndarray, feature_names: list[str], max_depth: int = 4) -> dict[str, Any]:
    y = (assign > 0).astype(int)
    if y.min() == y.max():
        return {"rules": ["always " + ("treat" if y[0] else "skip")], "depth": 0}
    tree = DecisionTreeClassifier(max_depth=max_depth, min_samples_leaf=40)
    tree.fit(X, y)
    return {"n_leaves": int(tree.get_n_leaves()), "depth": int(tree.get_depth()), "importances": dict(zip(feature_names, tree.feature_importances_.round(4).tolist()))}


def pareto_fairness(tau: np.ndarray, group: np.ndarray, budgets: list[float], revenue: float, cost: np.ndarray) -> list[dict[str, Any]]:
    pts = []
    for b in budgets:
        total = float(b * (cost.mean() * len(tau)))
        kn = knapsack_policy(tau, cost, total, revenue, group=group, fairness_slack=0.04)
        treated = kn["assign"] > 0
        if group is None:
            gap = 0.0
        else:
            r0 = treated[group == 0].mean() if (group == 0).any() else 0
            r1 = treated[group == 1].mean() if (group == 1).any() else 0
            gap = abs(float(r0 - r1))
        pts.append({"budget_frac": b, "value": kn["expected_value"], "fairness_gap": gap, "n_treated": kn["n_treated"]})
    return pts
