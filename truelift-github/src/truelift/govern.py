from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import numpy as np

from truelift.evaluate import ship_gates
from truelift.types import CausalFrame


def fairness_report(assign: np.ndarray, group: np.ndarray | None, tau: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    if group is None:
        return {"enabled": False}
    treated = assign > 0
    score = tau[:, 0] if tau.ndim > 1 else tau
    rows = []
    for g in sorted(set(group.tolist())):
        m = group == g
        rows.append(
            {
                "group": int(g),
                "n": int(m.sum()),
                "treat_rate": float(treated[m].mean()) if m.any() else 0.0,
                "mean_tau": float(score[m].mean()) if m.any() else 0.0,
                "mean_tau_treated": float(score[m & treated].mean()) if (m & treated).any() else 0.0,
            }
        )
    rates = [r["treat_rate"] for r in rows]
    gap = max(rates) - min(rates) if rates else 0.0
    # equal incremental opportunity: mean tau among treated vs population
    return {"enabled": True, "slices": rows, "treat_rate_gap": gap, "ok": gap <= 0.08}


def model_card(
    frame: CausalFrame,
    ident: dict[str, Any],
    champion: str,
    metrics: dict[str, Any],
    gates: list[dict[str, Any]],
    fairness: dict[str, Any],
) -> dict[str, Any]:
    return {
        "name": "TrueLift targeting policy",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": frame.meta,
        "estimand": ident.get("estimand"),
        "assumptions": ident.get("notes"),
        "champion": champion,
        "metrics": {
            "auuc": metrics.get("champion_auuc"),
            "normalized_auuc": metrics.get("normalized_auuc"),
            "uplift@30": metrics.get("uplift@30"),
            "pehe": metrics.get("pehe"),
        },
        "gates": gates,
        "fairness": fairness,
        "limits": [
            "CATE is an average for similar people, not individual destiny.",
            "Observational mode requires unconfoundedness given X.",
            "Do not treat this as medical or credit advice without a live experiment.",
        ],
        "ship": all(g["ok"] for g in gates),
    }


def drift_report(X_ref: np.ndarray, X_now: np.ndarray, feature_names: list[str]) -> dict[str, Any]:
    rows = []
    for j, name in enumerate(feature_names[:24]):
        mu0, mu1 = float(X_ref[:, j].mean()), float(X_now[:, j].mean())
        s0 = float(X_ref[:, j].std() + 1e-9)
        psi = abs(mu1 - mu0) / s0
        rows.append({"feature": name, "psi": psi, "flag": psi > 0.25})
    return {"features": rows, "n_flagged": int(sum(r["flag"] for r in rows))}
