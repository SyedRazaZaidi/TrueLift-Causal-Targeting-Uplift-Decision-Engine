# TrueLift

**Causal decisioning engine** for targeting: who to treat, which offer, whether the policy is worth shipping.

This is a product, not a notebook. You load a campaign (or a public benchmark), the engine names the estimand, fits the estimator zoo, allocates under budget / cost / fairness / uncertainty, and audits the policy on a frozen test split. A control-plane dashboard is how you operate it.

## What you can do with it

- Rank people by **uplift** (effect of the action), not by “who will convert anyway.”
- Allocate a **budget** with top-k, knapsack, conformal “treat only if τ_lo > 0,” and fairness slack.
- Compare **S / T / X / R / DR learners**, uplift forest, causal forest, TARNet, DragonNet, Qini-oriented ranker — same people, same metrics.
- Read **Qini / AUUC**, bootstrap intervals, dominance vs a conversion model, PEHE when the true effect is known.
- Inspect **overlap, leakage, Rosenbaum Γ, ITT vs LATE** (when assignment ≠ exposure).
- Open a **person dossier**: Y(0), Y(1), τ interval, type (persuadable / sure thing / lost cause / sleeping dog), feature contributions.
- Export a **model card** and ship/hold gates.

## Quick start

Python 3.10+. From `D:\MyProjects\truelift`:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
truelift all --dataset synthetic
truelift serve
```

`truelift serve` starts the **API on :8000** and **Next.js on :3000** (first run runs `npm install` in `web/`). Open **[http://127.0.0.1:3000](http://127.0.0.1:3000)** for the primary dashboard. API-only fallback: `truelift serve --api-only` → [http://127.0.0.1:8000](http://127.0.0.1:8000).

Datasets:

| Flag | What |
|---|---|
| `synthetic` | Known CATE, two offers, groups, 12% noncompliance — always works offline |
| `hillstrom` | MineThatData email RCT (download, else replica) |
| `ihdp` | Semi-synthetic CATE, PEHE |
| `jobs` | Job-training style RCT |
| `criteo` | Criteo Uplift v2 (large; samples `TRUELIFT_CRITEO_SAMPLE`, default 200k) |

```powershell
truelift all --dataset hillstrom --outcome visit
```

## Dashboard (control plane)

First screen is the **Campaign console**: three policies at your budget (conversion ranker vs treat-all vs TrueLift), wasted spend, rescued persuadables, quadrant map (baseline propensity vs τ), Qini, and one-click **Export list**.

| Room | Purpose |
|---|---|
| **Campaign** | Budget slider, policy comparison, business impact, ship gates |
| **Models** | Estimator zoo + Qini curves |
| **Identify** | Estimand, overlap, Rosenbaum, leakage |
| **Allocate** | Knapsack / conformal / fairness + Pareto |
| **People** | Person dossiers by archetype |
| **Data** | Train on CSV or benchmarks; score cold audience |
| **Governance** | Model card, robustness gym, bootstrap |

Sample CSVs: `data/samples/campaign_labeled.csv`, `new_customers.csv`, `tiny_walkthrough.csv`.

**Campaign map:** click any dot for a person dossier (slide-over). Charts animate on load.

Hard-refresh after static asset upgrades: **Ctrl+F5** on `:8000` (fallback UI) or restart `truelift serve` for Next.

## Architecture

```
DATA → IDENTIFY → ESTIMATE → ALLOCATE → EVALUATE → EXPLAIN → GOVERN → SERVE
```

CLI: `prepare` · `train` · `serve` · `all`

Artifacts: `artifacts/run.json`, `artifacts/test_scores.npz`.

## Tests

```powershell
pytest -q
```

## License

MIT. Cite GroupLens/Hillstrom and Criteo AI Lab when you publish numbers from their data.
