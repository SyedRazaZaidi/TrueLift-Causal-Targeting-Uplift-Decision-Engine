from __future__ import annotations

import io
import zipfile
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import pandas as pd

from truelift.config import Settings
from truelift.paths import PREPARED, RAW, ensure_dirs
from truelift.types import CausalFrame, encode_categoricals

HILLSTROM_URLS = [
    "https://raw.githubusercontent.com/benmiroglio/pymc-marketing/main/data/Hillstrom.csv",
    "https://www.minethatdata.com/Kevin_Hillstrom_MineThatData_E-MailAnalytics_Challenge.csv",
]

CRITEO_URL = "http://go.criteo.net/criteo-research-uplift-v2.1.csv.gz"


def _split(n: int, seed: int, test_size: float, val_size: float) -> np.ndarray:
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_test = int(n * test_size)
    n_val = int(n * val_size)
    split = np.full(n, "train", dtype=object)
    split[idx[:n_test]] = "test"
    split[idx[n_test : n_test + n_val]] = "val"
    return split


def _fetch(url: str, dest: Path, timeout: int = 60) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urlopen(url, timeout=timeout) as r, open(dest, "wb") as f:
        f.write(r.read())


def load_synthetic(settings: Settings, n: int = 8000) -> CausalFrame:
    """Orthogonal DGP: y0 from spend history only; tau from offer-match only."""
    rng = np.random.default_rng(settings.seed)
    recency = rng.integers(1, 13, n).astype(float)
    history = rng.lognormal(4.0, 0.9, n)
    log_h = np.log1p(history)
    z_h = (log_h - log_h.mean()) / (log_h.std() + 1e-9)
    mens = rng.binomial(1, 0.5, n).astype(float)
    womens = rng.binomial(1, 0.5, n).astype(float)
    newbie = rng.binomial(1, 0.40, n).astype(float)
    urban = rng.binomial(1, 0.35, n).astype(float)
    X = np.column_stack([recency, log_h, mens, womens, newbie, urban])
    names = ["recency", "log_history", "mens", "womens", "newbie", "urban"]

    y0 = 1 / (1 + np.exp(-(-2.2 + 1.35 * z_h)))
    y0 = np.clip(y0, 0.03, 0.45)
    high = z_h > 0.85
    n_high = int(high.sum())
    if n_high:
        y0[high] = np.clip(0.48 + 0.12 * rng.random(n_high), 0.48, 0.70)

    tau_w = 0.26 * womens * (0.35 + 0.65 * newbie) * (~high).astype(float)
    tau_m = 0.24 * mens * (0.35 + 0.65 * (1 - newbie)) * (~high).astype(float)
    dog = (mens > 0.5) & (recency >= 9) & (~high)
    tau_m[dog] = -0.15
    tau_w[dog] = np.minimum(tau_w[dog], -0.04)
    tau_w = np.clip(tau_w, -0.18, 0.34)
    tau_m = np.clip(tau_m, -0.18, 0.34)
    y1w = np.clip(y0 + tau_w, 0.0, 0.95)
    y1m = np.clip(y0 + tau_m, 0.0, 0.95)

    t = rng.integers(0, 3, n)
    y_pot = np.where(t == 0, y0, np.where(t == 1, y1w, y1m))
    y = rng.binomial(1, y_pot).astype(float)

    exposure = t.copy()
    sneak = rng.random(n) < 0.10
    exposure[sneak & (t > 0)] = 0

    return CausalFrame(
        ids=np.arange(n),
        X=X,
        feature_names=names,
        t=t.astype(int),
        y=y,
        arm_names=["control", "womens_offer", "mens_offer"],
        split=_split(n, settings.seed, settings.test_size, settings.val_size),
        exposure=exposure,
        true_tau=np.column_stack([tau_w, tau_m]),
        group=(womens > 0.5).astype(int),
        group_names=["other", "womens_buyer"],
        cost=np.full(n, settings.contact_cost),
        spend=y * rng.uniform(8, 40, n),
        meta={
            "dataset": "synthetic",
            "randomized": True,
            "known_cate": True,
            "n": n,
            "notes": "y0 from history; tau from offer-match. Sure things are high-history. Response ranking should lose.",
        },
    )


def load_ihdp(settings: Settings) -> CausalFrame:
    """IHDP-style semi-synthetic: known CATE, observational treatment."""
    rng = np.random.default_rng(settings.seed)
    n, d = 747, 25
    X = rng.normal(size=(n, d))
    X[:, :6] = rng.binomial(1, 0.4, size=(n, 6)).astype(float)
    # Confounded assignment
    logits = 0.4 * X[:, 0] + 0.3 * X[:, 1] - 0.2 * X[:, 2]
    p = 1 / (1 + np.exp(-logits))
    t = rng.binomial(1, p)
    beta = rng.normal(0, 0.15, d)
    mu0 = X @ beta
    tau = 4.0 + 0.5 * X[:, 0] - 0.3 * X[:, 3]  # classic IHDP-like response surface A
    mu1 = mu0 + tau
    y = np.where(t == 1, mu1, mu0) + rng.normal(0, 1.0, n)
    names = [f"x{i}" for i in range(d)]
    return CausalFrame(
        ids=np.arange(n),
        X=X,
        feature_names=names,
        t=t.astype(int),
        y=y,
        arm_names=["control", "treat"],
        split=_split(n, settings.seed, settings.test_size, settings.val_size),
        true_tau=tau[:, None],
        group=(X[:, 0] > 0).astype(int),
        group_names=["g0", "g1"],
        cost=np.full(n, settings.contact_cost),
        meta={"dataset": "ihdp_sim", "randomized": False, "known_cate": True, "continuous_y": True},
    )


def load_jobs(settings: Settings) -> CausalFrame:
    """Lalonde/Jobs-style: job training, binary employment-like outcome."""
    rng = np.random.default_rng(settings.seed + 7)
    n = 3212
    age = rng.integers(16, 55, n).astype(float)
    educ = rng.integers(6, 18, n).astype(float)
    black = rng.binomial(1, 0.4, n).astype(float)
    hisp = rng.binomial(1, 0.1, n).astype(float)
    married = rng.binomial(1, 0.35, n).astype(float)
    re74 = rng.lognormal(8.2, 1.1, n)
    re75 = re74 * rng.uniform(0.6, 1.2, n)
    X = np.column_stack([age, educ, black, hisp, married, np.log1p(re74), np.log1p(re75)])
    names = ["age", "educ", "black", "hisp", "married", "log_re74", "log_re75"]
    tau = 0.08 + 0.04 * (educ > 12) - 0.03 * (age > 40)
    y0 = np.clip(0.15 + 0.02 * (educ - 10) / 8 + 0.05 * married, 0.02, 0.8)
    y1 = np.clip(y0 + tau, 0.02, 0.95)
    t = rng.binomial(1, 0.35, n)
    y = rng.binomial(1, np.where(t == 1, y1, y0)).astype(float)
    return CausalFrame(
        ids=np.arange(n),
        X=X,
        feature_names=names,
        t=t.astype(int),
        y=y,
        arm_names=["control", "job_training"],
        split=_split(n, settings.seed, settings.test_size, settings.val_size),
        true_tau=tau[:, None],
        group=black.astype(int),
        group_names=["other", "black"],
        cost=np.full(n, 4.0),
        meta={"dataset": "jobs_sim", "randomized": True, "known_cate": True},
    )


def _hillstrom_offline(settings: Settings, n: int = 64000) -> pd.DataFrame:
    rng = np.random.default_rng(settings.seed + 3)
    recency = rng.integers(1, 13, n)
    history = np.round(rng.lognormal(4.5, 0.85, n), 2)
    mens = rng.binomial(1, 0.55, n)
    womens = rng.binomial(1, 0.55, n)
    zip_code = rng.choice(["Urban", "Suburban", "Rural"], n, p=[0.28, 0.45, 0.27])
    newbie = rng.binomial(1, 0.35, n)
    channel = rng.choice(["Phone", "Web", "Multichannel"], n, p=[0.28, 0.56, 0.16])
    segment = rng.choice(["No E-Mail", "Womens E-Mail", "Mens E-Mail"], n)
    t_w = segment == "Womens E-Mail"
    t_m = segment == "Mens E-Mail"
    p0 = 0.106 + 0.04 * (history > 150) - 0.02 * newbie
    p_w = p0 + 0.05 * womens + 0.03 * newbie - 0.01 * (zip_code == "Rural")
    p_m = p0 + 0.045 * mens - 0.02 * newbie
    p = np.clip(np.where(t_w, p_w, np.where(t_m, p_m, p0)), 0.02, 0.55)
    visit = rng.binomial(1, p)
    conversion = visit * rng.binomial(1, 0.05, n)
    spend = conversion * rng.uniform(20, 180, n)
    return pd.DataFrame(
        {
            "recency": recency,
            "history": history,
            "mens": mens,
            "womens": womens,
            "zip_code": zip_code,
            "newbie": newbie,
            "channel": channel,
            "segment": segment,
            "visit": visit,
            "conversion": conversion,
            "spend": spend,
        }
    )


def load_hillstrom(settings: Settings) -> CausalFrame:
    ensure_dirs()
    dest = RAW / "hillstrom.csv"
    df = None
    src = "offline_replica"
    if dest.exists():
        df = pd.read_csv(dest)
        src = "cache"
    else:
        for url in HILLSTROM_URLS:
            try:
                _fetch(url, dest, timeout=45)
                df = pd.read_csv(dest)
                src = "download"
                break
            except Exception:
                continue
        if df is None:
            df = _hillstrom_offline(settings)
            df.to_csv(dest, index=False)
            src = "offline_replica"
    # Normalize column names
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    if "segment" not in df.columns:
        raise ValueError("Hillstrom CSV missing segment")
    ycol = settings.outcome if settings.outcome in df.columns else "visit"
    arm_map = {"no e-mail": 0, "womens e-mail": 1, "mens e-mail": 2}
    t = df["segment"].astype(str).str.lower().map(arm_map).fillna(0).astype(int).to_numpy()
    feat_cols = [c for c in ["recency", "history", "mens", "womens", "zip_code", "newbie", "channel"] if c in df.columns]
    X, names = encode_categoricals(df, feat_cols)
    y = df[ycol].astype(float).to_numpy()
    spend = df["spend"].astype(float).to_numpy() if "spend" in df.columns else None
    group = df["womens"].astype(int).to_numpy() if "womens" in df.columns else None
    n = len(df)
    return CausalFrame(
        ids=np.arange(n),
        X=X.astype(float),
        feature_names=names,
        t=t,
        y=y,
        arm_names=["control", "womens_email", "mens_email"],
        split=_split(n, settings.seed, settings.test_size, settings.val_size),
        group=group,
        group_names=["other", "womens_buyer"] if group is not None else [],
        cost=np.full(n, settings.contact_cost),
        spend=spend,
        meta={"dataset": "hillstrom", "randomized": True, "known_cate": False, "source": src if dest.exists() else "offline_replica", "outcome": ycol, "n": n},
    )


def load_criteo(settings: Settings, sample: int | None = None) -> CausalFrame:
    ensure_dirs()
    dest = RAW / "criteo_uplift_v2.csv.gz"
    if not dest.exists():
        try:
            _fetch(CRITEO_URL, dest, timeout=180)
        except Exception as e:
            raise RuntimeError(
                "Criteo download failed. Place criteo-research-uplift-v2.1.csv.gz in data/raw/. "
                f"({e})"
            ) from e
    usecols = [f"f{i}" for i in range(12)] + ["treatment", "conversion", "visit", "exposure"]
    sample = sample or settings.criteo_sample
    df = pd.read_csv(dest, usecols=lambda c: c in usecols, nrows=sample)
    ycol = settings.outcome if settings.outcome in df.columns else "visit"
    X = df[[c for c in df.columns if c.startswith("f")]].to_numpy(dtype=float)
    names = [c for c in df.columns if c.startswith("f")]
    t = df["treatment"].astype(int).to_numpy()
    y = df[ycol].astype(float).to_numpy()
    exposure = df["exposure"].astype(int).to_numpy() if "exposure" in df.columns else None
    n = len(df)
    return CausalFrame(
        ids=np.arange(n),
        X=X,
        feature_names=names,
        t=t,
        y=y,
        arm_names=["control", "ad"],
        split=_split(n, settings.seed, settings.test_size, settings.val_size),
        exposure=exposure,
        cost=np.full(n, settings.contact_cost),
        meta={"dataset": "criteo_v2", "randomized": True, "known_cate": False, "n": n, "sampled": sample, "outcome": ycol},
    )


def frame_from_dataframe(df: pd.DataFrame, settings: Settings, treatment_col: str | None = None, outcome_col: str | None = None) -> CausalFrame:
    treatment_col = treatment_col or settings.treatment_col
    outcome_col = outcome_col or settings.outcome
    if len(df) > settings.csv_max_rows:
        df = df.sample(settings.csv_max_rows, random_state=settings.seed).reset_index(drop=True)
    if treatment_col not in df.columns:
        raise ValueError(f"Missing treatment column '{treatment_col}'. Columns: {list(df.columns)}")
    if outcome_col not in df.columns:
        raise ValueError(f"Missing outcome column '{outcome_col}'. Columns: {list(df.columns)}")
    return load_csv_df(df, settings, treatment_col, outcome_col)


def load_csv_df(df: pd.DataFrame, settings: Settings, treatment_col: str, outcome_col: str, arm_order: list[str] | None = None) -> CausalFrame:
    t_raw = df[treatment_col]
    if t_raw.dtype == object or str(t_raw.dtype).startswith("string"):
        arms = arm_order or sorted(t_raw.astype(str).unique(), key=lambda x: (x.lower() not in {"control", "0", "no e-mail"}, x))
        mapping = {a: i for i, a in enumerate(arms)}
        t = t_raw.astype(str).map(mapping).fillna(0).astype(int).to_numpy()
        arm_names = list(arms)
    else:
        t = t_raw.astype(int).to_numpy()
        k = int(t.max()) + 1
        arm_names = ["control"] + [f"arm_{i}" for i in range(1, k)]
    drop = {
        treatment_col,
        outcome_col,
        "id",
        "split",
        "exposure",
        "group",
        "cost",
        "dose",
        "true_tau",
        "gold_uplift",
        "gold_uplift_mens",
        "spend",
        "segment",
    }
    if outcome_col != "visit":
        drop.add("visit")
    if outcome_col != "conversion":
        drop.add("conversion")
    feat_cols = [c for c in df.columns if c not in drop]
    X, names = encode_categoricals(df, feat_cols)
    n = len(df)
    randomized = bool(df.attrs.get("randomized", False)) if hasattr(df, "attrs") else False
    return CausalFrame(
        ids=df["id"].to_numpy() if "id" in df.columns else np.arange(n),
        X=X.astype(float),
        feature_names=names,
        t=t,
        y=df[outcome_col].astype(float).to_numpy(),
        arm_names=arm_names,
        split=df["split"].astype(str).to_numpy() if "split" in df.columns else _split(n, settings.seed, settings.test_size, settings.val_size),
        exposure=df["exposure"].astype(int).to_numpy() if "exposure" in df.columns else None,
        group=df["group"].astype(int).to_numpy() if "group" in df.columns else None,
        cost=df["cost"].astype(float).to_numpy() if "cost" in df.columns else np.full(n, settings.contact_cost),
        meta={"dataset": "csv", "randomized": randomized, "known_cate": False, "n": n, "outcome": outcome_col},
    )


def encode_like(df: pd.DataFrame, feature_names: list[str], drop: set[str]) -> np.ndarray:
    feat_cols = [c for c in df.columns if c not in drop]
    X, names = encode_categoricals(df, feat_cols)
    out = np.zeros((len(df), len(feature_names)), dtype=float)
    idx = {n: i for i, n in enumerate(names)}
    for j, name in enumerate(feature_names):
        if name in idx:
            out[:, j] = X[:, idx[name]]
    return out


def campaign_template() -> pd.DataFrame:
    """Labeled campaign CSV a user can extend or replace."""
    rng = np.random.default_rng(0)
    n = 40
    return pd.DataFrame(
        {
            "id": np.arange(n),
            "recency": rng.integers(1, 13, n),
            "history": np.round(rng.lognormal(4.0, 0.7, n), 2),
            "mens": rng.integers(0, 2, n),
            "womens": rng.integers(0, 2, n),
            "newbie": rng.integers(0, 2, n),
            "urban": rng.integers(0, 2, n),
            "treatment": rng.choice(["control", "womens_offer", "mens_offer"], n),
            "visit": rng.integers(0, 2, n),
        }
    )


def load_csv(path: str | Path, settings: Settings, treatment_col: str, outcome_col: str, arm_order: list[str] | None = None) -> CausalFrame:
    return load_csv_df(pd.read_csv(path), settings, treatment_col, outcome_col, arm_order)


def load_dataset(settings: Settings) -> CausalFrame:
    name = settings.dataset
    key = name.lower()
    path = Path(name)
    if key in {"synthetic", "synth"}:
        return load_synthetic(settings)
    if key in {"hillstrom", "email"}:
        return load_hillstrom(settings)
    if key in {"ihdp"}:
        return load_ihdp(settings)
    if key in {"jobs"}:
        return load_jobs(settings)
    if key.startswith("criteo"):
        return load_criteo(settings)
    if path.exists() and path.suffix.lower() == ".csv":
        return load_csv(path, settings, settings.treatment_col, settings.outcome)
    if key.endswith(".csv"):
        return load_csv(name, settings, settings.treatment_col, settings.outcome)
    raise ValueError(f"Unknown dataset {settings.dataset}")


def leakage_report(frame: CausalFrame) -> dict:
    """Flag features that are suspiciously aligned with treatment (post-treatment risk)."""
    from sklearn.metrics import roc_auc_score

    flags = []
    t_bin = (frame.t > 0).astype(int)
    if t_bin.min() == t_bin.max():
        return {"flags": [], "ok": True}
    for i, name in enumerate(frame.feature_names):
        x = frame.X[:, i]
        if np.nanstd(x) < 1e-12:
            continue
        try:
            auc = float(roc_auc_score(t_bin, x))
        except Exception:
            continue
        auc = max(auc, 1 - auc)
        if auc >= 0.92:
            flags.append({"feature": name, "auc_vs_treatment": round(auc, 3), "severity": "high"})
        elif auc >= 0.80:
            flags.append({"feature": name, "auc_vs_treatment": round(auc, 3), "severity": "warn"})
    return {"flags": flags, "ok": len([f for f in flags if f["severity"] == "high"]) == 0}


def save_prepared(frame: CausalFrame, name: str) -> Path:
    ensure_dirs()
    path = PREPARED / f"{name}.npz"
    np.savez_compressed(
        path,
        ids=frame.ids,
        X=frame.X,
        t=frame.t,
        y=frame.y,
        split=frame.split.astype("U16"),
        feature_names=np.array(frame.feature_names),
        arm_names=np.array(frame.arm_names),
        exposure=frame.exposure if frame.exposure is not None else np.array([]),
        true_tau=frame.true_tau if frame.true_tau is not None else np.array([]),
        group=frame.group if frame.group is not None else np.array([]),
        cost=frame.cost if frame.cost is not None else np.array([]),
        spend=frame.spend if frame.spend is not None else np.array([]),
        group_names=np.array(frame.group_names),
        meta_json=np.array([str(frame.meta)]),
    )
    return path
