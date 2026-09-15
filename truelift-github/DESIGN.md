# TrueLift design

Full causal decisioning engine. Not a demo slice.

## Estimands

ATE, ATT, CATE, ITT, LATE (assignment vs exposure), multi-arm vs control. Observational mode states unconfoundedness.

## Estimators

S, T, X, R, DR meta-learners (LightGBM), uplift forest, causal forest (transformed-outcome GRF-style + T blend), TARNet, CFR-MMD, DragonNet (propensity head), uplift ranker (Qini proxy), optional stack.

## Policy

Top-k, multi-choice knapsack (value/cost), conformal treat-if-credible, fairness slack on treat rates, Pareto of value vs gap, distill to a policy tree.

## Evaluation

Qini, AUUC, normalized AUUC, uplift@k, DR policy value, bootstrap CIs, dominance vs response model, PEHE/ATE error on synthetic/IHDP, robustness gym (confounding, noncompliance, spillover), ship gates.

## Product

FastAPI + control-plane dashboard (six rooms). CSV/custom campaigns via `load_csv`. Criteo path for scale.
