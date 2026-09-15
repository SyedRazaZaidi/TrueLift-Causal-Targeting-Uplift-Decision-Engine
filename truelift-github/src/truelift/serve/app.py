from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles

from truelift.config import Settings
from truelift.data import campaign_template, frame_from_dataframe
from truelift.paths import ARTIFACTS, RAW, REPORTS, ensure_dirs
from truelift.pipeline import DEFAULT_MODELS, run_engine
from truelift.policy import budget_policy, knapsack_policy
from truelift.score import score_dataframe

STATIC = Path(__file__).resolve().parent / "static"


def _run() -> dict:
    path = ARTIFACTS / "run.json"
    if not path.exists():
        path = REPORTS / "run.json"
    if not path.exists():
        raise HTTPException(404, "No run yet. Execute: truelift all --dataset synthetic")
    return json.loads(path.read_text(encoding="utf-8"))


def _scores():
    path = ARTIFACTS / "test_scores.npz"
    if not path.exists():
        raise HTTPException(404, "Missing test_scores.npz — train first.")
    return np.load(path, allow_pickle=True)


def create_app(*, redirect_root: str | None = None) -> FastAPI:
    app = FastAPI(title="TrueLift", version="1.0.0")
    _redirect_root = redirect_root
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/api/health")
    def health():
        ok = (ARTIFACTS / "run.json").exists() or (REPORTS / "run.json").exists()
        return {"ok": ok, "engine": "truelift", "product": True}

    @app.get("/api/run")
    def run():
        return _run()

    @app.get("/api/leaderboard")
    def leaderboard():
        r = _run()
        return {"champion": r["champion"], "rows": r["leaderboard"], "dominance": r["dominance_vs_response"]}

    @app.get("/api/identification")
    def identification():
        r = _run()
        return r["identification"] | {"leakage": r["leakage"], "rosenbaum": r["rosenbaum"], "late": r["late"], "ate_ipw": r["ate_ipw"]}

    @app.get("/api/qini")
    def qini(model: str | None = None):
        r = _run()
        name = model or r["champion"]
        q = r["qini"].get(name)
        if q is None:
            raise HTTPException(404, name)
        return {"model": name, "curve": q, "response": r["qini_response"], "bootstrap": r["bootstrap"]}

    @app.get("/api/policy")
    def policy(budget: float = 0.3, mode: str = "knapsack", conformal: bool = False, fairness: bool = False):
        r = _run()
        z = _scores()
        tau = z["tau"]
        cost = z["cost"]
        lo = z["lo"] if conformal else None
        group = z["group"] if z["group"].size and fairness else None
        n = tau.shape[0]
        if mode == "topk":
            assign = budget_policy(tau, budget, 0, lo)
            spent = float((assign > 0).sum() * (float(cost.mean()) if cost.size else 1))
            payload = {"assign": assign.tolist(), "n_treated": int((assign > 0).sum()), "spent": spent}
        else:
            total = float(budget * float(np.sum(cost)))
            kn = knapsack_policy(tau, cost, total, 12.0, lo=lo, group=group if group is not None and group.size else None, fairness_slack=0.05 if fairness else 1.0)
            payload = kn
            payload["assign"] = kn["assign"].tolist()
            assign = kn["assign"]
        t = z["t"]
        y = z["y"]
        score = tau[:, 0] if tau.ndim > 1 else tau
        # incremental among selected using two-sample
        sel = assign > 0 if isinstance(assign, np.ndarray) else np.array(payload["assign"]) > 0
        if sel.sum() and t[sel].min() != t[sel].max():
            inc = float(y[sel][t[sel] > 0].mean() - y[sel][t[sel] == 0].mean())
        else:
            inc = 0.0
        kinds = _kinds(z, np.array(payload["assign"]))
        return {
            "budget": budget,
            "mode": mode,
            "conformal": conformal,
            "fairness": fairness,
            "n_treated": int(sel.sum()),
            "incremental_rate": inc,
            "kinds": kinds,
            **{k: v for k, v in payload.items() if k != "assign"},
            "assign_preview": np.array(payload["assign"])[:80].tolist(),
        }

    @app.get("/api/people")
    def people(kind: str | None = None, limit: int = 40):
        r = _run()
        if kind is None:
            return {"featured": r["people"]}
        z = _scores()
        assign = z["assign"]
        items = []
        tau = z["tau"]
        mu = z["mu"]
        resp = z["resp"]
        score = tau[:, 0] if tau.ndim > 1 else tau
        for i in range(min(len(assign), 2000)):
            k = _kind_one(mu[i], score[i])
            if kind != "all" and k != kind:
                continue
            items.append(
                {
                    "id": int(z["ids"][i]),
                    "index": i,
                    "kind": k,
                    "tau": float(score[i]),
                    "mu0": float(mu[i, 0]),
                    "mu1": float(mu[i, min(1, mu.shape[1] - 1)]),
                    "treat": bool(assign[i] > 0),
                    "response": float(resp[i]),
                }
            )
            if len(items) >= limit:
                break
        return {"items": items}

    @app.get("/api/person/{index}")
    def person(index: int):
        r = _run()
        for p in r["people"]:
            if p["index"] == index or p["id"] == index:
                return p
        z = _scores()
        if index < 0 or index >= len(z["ids"]):
            raise HTTPException(404, "person")
        i = index
        tau = z["tau"]
        mu = z["mu"]
        score = float(tau[i, 0] if tau.ndim > 1 else tau[i])
        return {
            "id": int(z["ids"][i]),
            "index": i,
            "kind": _kind_one(mu[i], score),
            "tau": score,
            "tau_lo": float(z["lo"][i, 0] if z["lo"].ndim > 1 else z["lo"][i]),
            "tau_hi": float(z["hi"][i, 0] if z["hi"].ndim > 1 else z["hi"][i]),
            "mu0": float(mu[i, 0]),
            "mu1": float(mu[i, min(1, mu.shape[1] - 1)]),
            "assign": int(z["assign"][i]),
            "response_score": float(z["resp"][i]),
            "features": {n: float(z["X"][i, j]) for j, n in enumerate(r["feature_names"])},
            "true_tau": float(z["true_tau"][i, 0]) if z["true_tau"].size else None,
        }

    @app.get("/api/risk")
    def risk():
        r = _run()
        return {
            "card": r["card"],
            "gates": r["gates"],
            "fairness": r["fairness"],
            "robustness": r["robustness"],
            "policy_tree": r["policy_tree"],
            "pareto": r["pareto"],
        }

    @app.get("/api/overview")
    def overview():
        r = _run()
        return {
            "dataset": r["dataset"],
            "champion": r["champion"],
            "n_test": r["n_test"],
            "leaderboard_top": r["leaderboard"][:5],
            "gates": r["gates"],
            "ship": r["card"]["ship"],
            "bootstrap": r["bootstrap"],
            "dominance": r["dominance_vs_response"],
            "arms": r["arm_names"],
        }

    @app.get("/api/stage")
    def stage(budget: float = 0.3):
        """Hero payload: three policies, wasted spend, offer mix, quadrant map."""
        return build_stage(budget)

    @app.get("/api/template.csv")
    def template_csv():
        buf = io.StringIO()
        campaign_template().to_csv(buf, index=False)
        return Response(buf.getvalue(), media_type="text/csv", headers={"Content-Disposition": "attachment; filename=truelift_campaign_template.csv"})

    @app.post("/api/train")
    async def train(
        dataset: str = Form("synthetic"),
        treatment: str = Form("treatment"),
        outcome: str = Form("visit"),
        file: UploadFile | None = File(None),
    ):
        settings = Settings()
        settings.treatment_col = treatment
        settings.outcome = outcome
        frame = None
        if file is not None and file.filename:
            ensure_dirs()
            raw = await file.read()
            dest = RAW / "upload.csv"
            dest.write_bytes(raw)
            df = pd.read_csv(io.BytesIO(raw))
            frame = frame_from_dataframe(df, settings, treatment, outcome)
            frame.meta["dataset"] = file.filename
            settings.dataset = str(dest)
        else:
            settings.dataset = dataset
        models = DEFAULT_MODELS
        if frame is not None and frame.n < 2500:
            models = ["s_learner", "t_learner", "x_learner", "uplift_ranker", "causal_forest"]
        art = run_engine(settings, models, frame=frame)
        champ = next(r for r in art["leaderboard"] if r["model"] == art["champion"])
        return {"champion": art["champion"], "auuc": champ["auuc"], "ship": art["card"]["ship"], "n_test": art["n_test"], "gates": art["gates"]}

    @app.post("/api/score")
    async def score(
        budget: float = Form(0.3),
        mode: str = Form("knapsack"),
        file: UploadFile = File(...),
    ):
        raw = await file.read()
        df = pd.read_csv(io.BytesIO(raw))
        try:
            out = score_dataframe(df, budget=budget, mode=mode)
        except FileNotFoundError as e:
            raise HTTPException(400, str(e)) from e
        buf = io.StringIO()
        out.to_csv(buf, index=False)
        return Response(buf.getvalue(), media_type="text/csv", headers={"Content-Disposition": "attachment; filename=truelift_scored.csv"})

    def _current_assign(budget: float, mode: str, conformal: bool, fairness: bool):
        z = _scores()
        tau = z["tau"]
        cost = z["cost"]
        lo = z["lo"] if conformal else None
        group = z["group"] if z["group"].size and fairness else None
        if mode == "topk":
            assign = budget_policy(tau, budget, 0, lo)
        else:
            total = float(budget * float(np.sum(cost)))
            kn = knapsack_policy(tau, cost, total, 12.0, lo=lo, group=group if group is not None and getattr(group, "size", 0) else None, fairness_slack=0.05 if fairness else 1.0)
            assign = kn["assign"]
        return z, np.asarray(assign)

    @app.get("/api/export.csv")
    def export_csv(budget: float = 0.3, mode: str = "knapsack", conformal: bool = False, fairness: bool = False):
        z, assign = _current_assign(budget, mode, conformal, fairness)
        tau = z["tau"]
        score = tau[:, 0] if tau.ndim > 1 else tau
        mu = z["mu"]
        df = pd.DataFrame(
            {
                "id": z["ids"],
                "tau": score,
                "tau_lo": z["lo"][:, 0] if z["lo"].ndim > 1 else z["lo"],
                "mu0": mu[:, 0],
                "mu1": mu[:, min(1, mu.shape[1] - 1)],
                "assign": assign,
                "decision": np.where(assign > 0, "treat", "skip"),
                "response_score": z["resp"],
                "y": z["y"],
                "t": z["t"],
            }
        )
        buf = io.StringIO()
        df.to_csv(buf, index=False)
        return Response(buf.getvalue(), media_type="text/csv", headers={"Content-Disposition": "attachment; filename=truelift_assignment.csv"})

    if STATIC.exists():
        app.mount("/assets", StaticFiles(directory=STATIC), name="assets")

        @app.get("/")
        def index():
            if _redirect_root:
                return RedirectResponse(_redirect_root, status_code=302)
            return FileResponse(STATIC / "index.html")

        @app.get("/desk")
        def desk_fallback():
            """Legacy static desk; primary UI is Next.js."""
            if _redirect_root:
                return RedirectResponse(_redirect_root, status_code=302)
            return FileResponse(STATIC / "index.html")

    return app


def _kind_one(mu_row: np.ndarray, tau: float) -> str:
    mu0 = float(mu_row[0])
    if tau < -0.01:
        return "sleeping_dog"
    if abs(tau) <= 0.01 and mu0 >= 0.15:
        return "sure_thing"
    if abs(tau) <= 0.01 and mu0 < 0.08:
        return "lost_cause"
    return "persuadable"


def _topk_mask(score: np.ndarray, budget: float) -> np.ndarray:
    n = len(score)
    k = max(1, int(round(n * budget)))
    sel = np.zeros(n, dtype=bool)
    sel[np.argsort(-score)[:k]] = True
    return sel


def _uplift(y: np.ndarray, t: np.ndarray, sel: np.ndarray) -> float:
    if sel.sum() < 8:
        return 0.0
    yt, tt = y[sel], (t[sel] > 0)
    if tt.min() == tt.max():
        return 0.0
    return float(yt[tt].mean() - yt[~tt].mean())


def build_stage(budget: float = 0.3) -> dict:
    r = _run()
    z = _scores()
    tau = z["tau"]
    score = tau[:, 0] if tau.ndim > 1 else tau
    mu = z["mu"]
    mu0 = mu[:, 0]
    resp = z["resp"]
    y, t = z["y"], z["t"]
    n = len(score)
    cost = float(np.mean(z["cost"])) if z["cost"].size else 1.0
    revenue = 12.0

    ours = _topk_mask(score, budget)
    theirs = _topk_mask(resp, budget)
    everyone = np.ones(n, dtype=bool)

    def pack(name, sel, role):
        inc = _uplift(y, t, sel)
        n_t = int(sel.sum())
        extra = inc * n_t
        kinds = {"persuadable": 0, "sure_thing": 0, "lost_cause": 0, "sleeping_dog": 0}
        for i in np.where(sel)[0]:
            kinds[_kind_one(mu[i], float(score[i]))] += 1
        return {
            "name": name,
            "role": role,
            "n_treated": n_t,
            "incremental_rate": inc,
            "extra_visits": extra,
            "spend": n_t * cost,
            "kinds": kinds,
        }

    p_ours = pack("TrueLift", ours, "us")
    p_resp = pack("Conversion model", theirs, "naive")
    p_all = pack("Treat everyone", everyone, "waste")
    wasted_mask = theirs & (score <= 0.02)
    rescued_mask = ours & ~theirs & (score > 0.02)
    dogs_mailed = theirs & (score < -0.01)

    arms = r.get("arm_names") or ["control", "treat"]
    offer = {"skip": int((~ours).sum())}
    if tau.ndim > 1 and tau.shape[1] >= 2:
        best = np.argmax(tau, axis=1) + 1
        for a, name in enumerate(arms[1:], start=1):
            offer[name] = int(((best == a) & ours).sum())
    else:
        offer[arms[1] if len(arms) > 1 else "treat"] = int(ours.sum())

    rng = np.random.default_rng(0)
    take = rng.choice(n, size=min(900, n), replace=False)
    points = []
    for i in take:
        knd = _kind_one(mu[i], float(score[i]))
        points.append(
            {
                "i": int(i),
                "x": float(mu0[i]),
                "y": float(score[i]),
                "kind": knd,
                "ours": bool(ours[i]),
                "theirs": bool(theirs[i]),
                "resp": float(resp[i]),
            }
        )

    champ = r["champion"]
    q = r["qini"].get(champ, {})
    ds = r.get("dataset", {})
    ds_name = ds.get("dataset", "campaign")
    n_total = ds.get("n", n)
    title = {
        "synthetic": "Forge Retail · synthetic incrementality holdout",
        "hillstrom": "Hillstrom · MineThatData email RCT",
        "ihdp": "IHDP · semi-synthetic CATE benchmark",
        "jobs": "Jobs · training program RCT",
        "criteo_v2": "Criteo · uplift v2 incrementality sample",
        "csv": "Uploaded campaign · randomized experiment",
    }.get(ds_name, f"{ds_name} · causal targeting holdout")

    return {
        "campaign": title,
        "thesis": "Don’t pay for people who were going to convert anyway.",
        "ship": r["card"]["ship"],
        "n_test": n,
        "budget": budget,
        "champion": champ,
        "arms": arms,
        "policies": [p_resp, p_all, p_ours],
        "wasted": {
            "n": int(wasted_mask.sum()),
            "spend": float(wasted_mask.sum() * cost),
            "label": "contacts the conversion model would buy that have ~zero lift",
        },
        "rescued": {
            "n": int(rescued_mask.sum()),
            "label": "persuadables we mail that the conversion model would skip",
        },
        "dogs_mailed": int(dogs_mailed.sum()),
        "offers": offer,
        "map": points,
        "qini": q,
        "qini_response": r.get("qini_response"),
        "gates": r["gates"],
        "auuc": next((row["auuc"] for row in r["leaderboard"] if row["model"] == champ), None),
        "response_auuc": r.get("qini_response", {}).get("auuc"),
        "cost": cost,
        "revenue": revenue,
        "dataset": r.get("dataset", {}),
        "leaderboard": [x for x in r.get("leaderboard", []) if not x.get("baseline")][:8],
    }


def _kinds(z, assign: np.ndarray) -> dict[str, int]:
    tau = z["tau"]
    mu = z["mu"]
    score = tau[:, 0] if tau.ndim > 1 else tau
    c = {"persuadable": 0, "sure_thing": 0, "lost_cause": 0, "sleeping_dog": 0}
    for i in range(len(assign)):
        c[_kind_one(mu[i], float(score[i]))] += 1
    return c
