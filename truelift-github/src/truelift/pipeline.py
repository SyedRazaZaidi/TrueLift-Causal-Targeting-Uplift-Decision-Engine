from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from typing import Any

import joblib
import numpy as np

from truelift.config import Settings
from truelift.data import leakage_report, load_dataset, save_prepared
from truelift.evaluate import (
    bootstrap_auuc,
    dominance_test,
    pehe,
    ate_error,
    qini_curve,
    qini_bands,
    response_score,
    ship_gates,
    uplift_at_k,
    policy_value_dr,
)
from truelift.explain import person_dossier, segment_profiles, shap_like_tau
from truelift.govern import fairness_report, model_card
from truelift.identify import identify, propensity_scores, ate_ipw, rosenbaum_gamma
from truelift.models import FACTORY, calibration_quantile, make_model
from truelift.paths import ARTIFACTS, REPORTS, ensure_dirs
from truelift.policy import budget_policy, distill_policy_tree, knapsack_policy, pareto_fairness
from truelift.robustness import robustness_report
from truelift.types import CausalFrame

DEFAULT_MODELS = [
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
]


def _score_col(tau: np.ndarray, arm: int = 0) -> np.ndarray:
    return tau[:, arm] if tau.ndim > 1 else tau


def run_engine(settings: Settings | None = None, models: list[str] | None = None, frame: CausalFrame | None = None) -> dict[str, Any]:
    settings = settings or Settings()
    models = models or DEFAULT_MODELS
    ensure_dirs()
    if frame is None:
        frame = load_dataset(settings)
    save_prepared(frame, settings.dataset.replace(".", "_"))
    leak = leakage_report(frame)
    ident = identify(frame, "cate" if not frame.meta.get("randomized") else "itt")
    if frame.exposure is not None and not np.array_equal(frame.exposure, frame.t):
        ident.notes.append("Exposure differs from assignment — LATE path is available in the desk.")

    train = frame.subset(frame.mask("train"))
    val = frame.subset(frame.mask("val")) if frame.mask("val").any() else train
    test = frame.subset(frame.mask("test"))

    fitted = {}
    board = []
    qinis = {}
    taus_test: dict[str, np.ndarray] = {}
    mus_test: dict[str, np.ndarray] = {}

    from sklearn.linear_model import LogisticRegression, LinearRegression

    if len(np.unique(train.y)) <= 2:
        rm = LogisticRegression(max_iter=400).fit(train.X, (train.y > 0).astype(int))
        resp_test = rm.predict_proba(test.X)[:, 1]
    else:
        rm = LinearRegression().fit(train.X, train.y)
        resp_test = rm.predict(test.X)

    q_resp = qini_curve(test.y, test.t, resp_test)
    board.append(
        {
            "model": "response_baseline",
            "auuc": q_resp["auuc"],
            "normalized_auuc": q_resp["normalized_auuc"],
            **uplift_at_k(test.y, test.t, resp_test),
            "baseline": True,
        }
    )
    qinis["response_baseline"] = q_resp

    if test.true_tau is not None:
        q_or = qini_curve(test.y, test.t, test.true_tau[:, 0])
        board.append(
            {
                "model": "oracle_tau",
                "auuc": q_or["auuc"],
                "normalized_auuc": q_or["normalized_auuc"],
                **uplift_at_k(test.y, test.t, test.true_tau[:, 0]),
                "baseline": True,
            }
        )
        qinis["oracle_tau"] = q_or

    for name in models:
        m = make_model(name, settings)
        m.fit(train)
        fitted[name] = m
        mu, tau = m.predict(test.X)
        mus_test[name] = mu
        taus_test[name] = tau
        score = _score_col(tau)
        q = qini_curve(test.y, test.t, score)
        upl = uplift_at_k(test.y, test.t, score)
        row: dict[str, Any] = {"model": name, "auuc": q["auuc"], "normalized_auuc": q["normalized_auuc"], **upl}
        if test.true_tau is not None:
            row["pehe"] = pehe(tau, test.true_tau)
            row["ate_error"] = ate_error(tau, test.true_tau)
        board.append(row)
        qinis[name] = q

    board.sort(key=lambda r: r["auuc"], reverse=True)
    champ_rows = [r for r in board if not r.get("baseline")]
    champion = champ_rows[0]["model"]
    tau_c = taus_test[champion]
    mu_c = mus_test[champion]
    score_c = _score_col(tau_c)

    boot = bootstrap_auuc(test.y, test.t, score_c, n_boot=min(settings.bootstrap, 120), seed=settings.seed)
    bands = qini_bands(test.y, test.t, score_c, n_boot=min(80, settings.bootstrap), seed=settings.seed)
    qinis[champion] = {**qinis[champion], **bands}
    resp_bands = qini_bands(test.y, test.t, resp_test, n_boot=min(80, settings.bootstrap), seed=settings.seed + 3)
    q_resp = {**q_resp, **resp_bands}
    qinis["response_baseline"] = q_resp
    dom = dominance_test(test.y, test.t, score_c, resp_test, n_boot=min(settings.bootstrap, 120), seed=settings.seed + 1)

    mu_val, tau_val = fitted[champion].predict(val.X)
    q_width = calibration_quantile(val, mu_val, settings.conformal_alpha, tau_val)
    lo = tau_c - q_width
    hi = tau_c + q_width

    e_test = propensity_scores(test)
    ate_hat = ate_ipw(test.y, test.t, e_test)
    gamma = rosenbaum_gamma(test.y, test.t)

    cost = test.cost if test.cost is not None else np.full(test.n, settings.contact_cost)
    policies = {}
    for frac in (0.1, 0.2, 0.3, 0.5):
        assign = budget_policy(tau_c, frac, arm=0, lo=None)
        assign_cf = budget_policy(tau_c, frac, arm=0, lo=lo)
        total = float(frac * cost.sum())
        kn = knapsack_policy(tau_c, cost, total, settings.revenue_per_outcome, lo=None, group=test.group, fairness_slack=1.0)
        kn_f = knapsack_policy(tau_c, cost, total, settings.revenue_per_outcome, lo=lo, group=test.group, fairness_slack=settings.fairness_slack)
        pi = (assign > 0).astype(float)
        mu0, mu1 = mu_c[:, 0], mu_c[:, min(1, mu_c.shape[1] - 1)]
        pv = policy_value_dr(test.y, test.t, e_test, mu0, mu1, pi)
        policies[str(frac)] = {
            "budget_frac": frac,
            "topk_n": int((assign > 0).sum()),
            "conformal_n": int((assign_cf > 0).sum()),
            "knapsack": {"n": kn["n_treated"], "spent": kn["spent"], "value": kn["expected_value"]},
            "knapsack_fair_conformal": {"n": kn_f["n_treated"], "spent": kn_f["spent"], "value": kn_f["expected_value"]},
            "policy_value_dr": pv,
            "assign_topk": assign.tolist(),
            "assign_conformal": assign_cf.tolist(),
            "assign_knapsack": kn["assign"].tolist(),
        }

    default_frac = "0.3"
    assign = np.array(policies[default_frac]["assign_knapsack"], dtype=int)
    tree = distill_policy_tree(test.X, assign, frame.feature_names)
    pareto = pareto_fairness(tau_c, test.group if test.group is not None else np.zeros(test.n, dtype=int), [0.1, 0.2, 0.3, 0.5], settings.revenue_per_outcome, cost)

    def _score(fr: CausalFrame) -> np.ndarray:
        _, ttau = fitted[champion].predict(fr.X)
        return _score_col(ttau)

    robust = robustness_report(test, _score)
    fair = fairness_report(assign, test.group, tau_c, test.y)

    metrics_gate = {
        "champion_auuc": champ_rows[0]["auuc"],
        "random_auuc": qinis[champion]["auuc_random"],
        "beats_response": champ_rows[0]["auuc"] > q_resp["auuc"] or bool(dom["beats"]),
        "positivity_ok": ident.positivity_ok,
        "normalized_auuc": champ_rows[0]["normalized_auuc"],
        "uplift@30": champ_rows[0].get("uplift@30"),
        "pehe": champ_rows[0].get("pehe"),
    }
    gates = ship_gates(metrics_gate)
    card = model_card(frame, ident.as_dict(), champion, metrics_gate, gates, fair)

    # people: mix of kinds
    score = score_c
    idxs = [
        int(np.argmax(score)),
        int(np.argmin(score)),
        int(np.argmax(resp_test - score)),
        int(np.argsort(-score)[len(score) // 10]),
        int(np.argsort(-mu_c[:, 0])[0]),
    ]
    people = []
    for i in dict.fromkeys(idxs):
        contribs = shap_like_tau(fitted[champion], test.X, frame.feature_names, i)
        people.append(person_dossier(test, i, mu_c, tau_c, lo, hi, assign, resp_test, contribs))

    segs = segment_profiles(test, tau_c)

    # LATE if exposure
    late = None
    if test.exposure is not None:
        from sklearn.linear_model import LinearRegression

        # Wald: ATE / first stage
        itt = float(test.y[test.t > 0].mean() - test.y[test.t == 0].mean()) if (test.t == 0).any() else 0.0
        fs = float((test.exposure[test.t > 0] > 0).mean() - (test.exposure[test.t == 0] > 0).mean()) if (test.t == 0).any() else 0.0
        late = {"itt": itt, "first_stage": fs, "wald_late": itt / (fs + 1e-9)}

    artifact = {
        "dataset": frame.meta,
        "n_train": int(train.n),
        "n_val": int(val.n),
        "n_test": int(test.n),
        "feature_names": frame.feature_names,
        "arm_names": frame.arm_names,
        "leakage": leak,
        "identification": ident.as_dict(),
        "leaderboard": board,
        "champion": champion,
        "qini": qinis,
        "qini_response": q_resp,
        "bootstrap": boot,
        "dominance_vs_response": dom,
        "conformal_width": q_width,
        "ate_ipw": ate_hat,
        "rosenbaum": gamma,
        "late": late,
        "policies": {k: {kk: vv for kk, vv in v.items() if not kk.startswith("assign")} for k, v in policies.items()},
        "pareto": pareto,
        "policy_tree": tree,
        "robustness": robust,
        "fairness": fair,
        "gates": gates,
        "card": card,
        "people": people,
        "segments": segs,
        "stack_weights": getattr(fitted.get("stack"), "w", None),
    }

    # persist scores for the live desk
    np.savez_compressed(
        ARTIFACTS / "test_scores.npz",
        ids=test.ids,
        X=test.X,
        t=test.t,
        y=test.y,
        tau=tau_c,
        mu=mu_c,
        lo=lo,
        hi=hi,
        resp=resp_test,
        assign=assign,
        group=test.group if test.group is not None else np.array([]),
        true_tau=test.true_tau if test.true_tau is not None else np.array([]),
        cost=cost,
        exposure=test.exposure if test.exposure is not None else np.array([]),
        split=test.split.astype("U8"),
    )
    # store all model scores
    np.savez_compressed(ARTIFACTS / "all_tau.npz", **{k: v for k, v in taus_test.items()})

    joblib.dump({k: v for k, v in fitted.items() if k not in {"tarnet", "cfrnet", "dragonnet"}}, ARTIFACTS / "models.joblib")
    serve_model = fitted.get("t_learner") or fitted.get("x_learner") or fitted.get(champion)
    joblib.dump(
        {
            "model": serve_model,
            "feature_names": frame.feature_names,
            "arm_names": frame.arm_names,
            "revenue": settings.revenue_per_outcome,
            "contact_cost": settings.contact_cost,
            "conformal_width": q_width,
            "champion": champion,
        },
        ARTIFACTS / "scorer.joblib",
    )
    assign_df = pd.DataFrame(
        {
            "id": test.ids,
            "tau": score_c,
            "mu0": mu_c[:, 0],
            "assign": assign,
            "decision": np.where(assign > 0, "treat", "skip"),
            "response_score": resp_test,
            "y": test.y,
            "t": test.t,
        }
    )
    assign_df.to_csv(ARTIFACTS / "assignment.csv", index=False)
    for nn in ("tarnet", "cfrnet", "dragonnet"):
        if nn in fitted and hasattr(fitted[nn], "net"):
            import torch

            torch.save(fitted[nn].net.state_dict(), ARTIFACTS / f"{nn}.pt")

    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "run.json").write_text(json.dumps(artifact, indent=2, default=str), encoding="utf-8")
    (ARTIFACTS / "run.json").write_text(json.dumps(artifact, indent=2, default=str), encoding="utf-8")
    return artifact
