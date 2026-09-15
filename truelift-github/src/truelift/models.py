from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import lightgbm as lgb
import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression

from truelift.config import Settings
from truelift.types import CausalFrame

Array = np.ndarray


def _is_binary(y: Array) -> bool:
    u = np.unique(y[~np.isnan(y)])
    return len(u) <= 2 and set(np.round(u, 8)).issubset({0.0, 1.0})


class OutcomeModel:
    def __init__(self, settings: Settings, binary: bool):
        self.binary = binary
        self.m = (
            lgb.LGBMClassifier(
                n_estimators=settings.n_estimators,
                learning_rate=0.05,
                num_leaves=31,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=settings.seed,
                verbose=-1,
            )
            if binary
            else lgb.LGBMRegressor(
                n_estimators=settings.n_estimators,
                learning_rate=0.05,
                num_leaves=31,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=settings.seed,
                verbose=-1,
            )
        )

    def fit(self, X, y):
        self.m.fit(X, y)
        return self

    def predict(self, X) -> Array:
        if self.binary:
            return self.m.predict_proba(X)[:, 1]
        return self.m.predict(X)


@dataclass
class FitResult:
    name: str
    mu: Array  # (n, n_arms) on the frame passed to predict
    tau: Array  # (n, n_treat) vs control
    extra: dict[str, Any]


class CATEModel:
    name = "base"

    def fit(self, frame: CausalFrame) -> "CATEModel":
        raise NotImplementedError

    def predict(self, X: Array) -> tuple[Array, Array]:
        raise NotImplementedError


class SLearner(CATEModel):
    name = "s_learner"

    def __init__(self, settings: Settings):
        self.settings = settings
        self.models: dict[int, OutcomeModel] = {}
        self.n_arms = 2
        self.binary = True

    def fit(self, frame: CausalFrame) -> "SLearner":
        self.n_arms = frame.n_arms
        self.binary = _is_binary(frame.y)
        Xt = np.column_stack([frame.X, frame.t.astype(float)])
        self.core = OutcomeModel(self.settings, self.binary).fit(Xt, frame.y)
        return self

    def predict(self, X: Array) -> tuple[Array, Array]:
        mus = []
        for a in range(self.n_arms):
            Xt = np.column_stack([X, np.full(len(X), float(a))])
            mus.append(self.core.predict(Xt))
        mu = np.column_stack(mus)
        tau = mu[:, 1:] - mu[:, 0:1]
        return mu, tau


class TLearner(CATEModel):
    name = "t_learner"

    def __init__(self, settings: Settings):
        self.settings = settings
        self.arm_models: list[OutcomeModel] = []
        self.n_arms = 2

    def fit(self, frame: CausalFrame) -> "TLearner":
        self.n_arms = frame.n_arms
        binary = _is_binary(frame.y)
        self.arm_models = []
        for a in range(self.n_arms):
            m = OutcomeModel(self.settings, binary)
            mask = frame.t == a
            if mask.sum() < 20:
                m.fit(frame.X, frame.y)
            else:
                m.fit(frame.X[mask], frame.y[mask])
            self.arm_models.append(m)
        return self

    def predict(self, X: Array) -> tuple[Array, Array]:
        mu = np.column_stack([m.predict(X) for m in self.arm_models])
        tau = mu[:, 1:] - mu[:, 0:1]
        return mu, tau


class XLearner(CATEModel):
    name = "x_learner"

    def __init__(self, settings: Settings):
        self.settings = settings
        self.t_learner = TLearner(settings)
        self.tau_models: list[OutcomeModel] = []
        self.prop: LogisticRegression | None = None
        self.n_arms = 2

    def fit(self, frame: CausalFrame) -> "XLearner":
        self.n_arms = frame.n_arms
        self.t_learner.fit(frame)
        mu, _ = self.t_learner.predict(frame.X)
        binary = False
        self.tau_models = []
        for a in range(1, self.n_arms):
            # imputed effects
            d1 = frame.y[frame.t == a] - mu[frame.t == a, 0]
            d0 = mu[frame.t == 0, a] - frame.y[frame.t == 0]
            m1 = OutcomeModel(self.settings, binary).fit(frame.X[frame.t == a], d1)
            m0 = OutcomeModel(self.settings, binary).fit(frame.X[frame.t == 0], d0)
            self.tau_models.append((m0, m1))
        self.prop = LogisticRegression(max_iter=400)
        self.prop.fit(frame.X, (frame.t > 0).astype(int))
        return self

    def predict(self, X: Array) -> tuple[Array, Array]:
        e = self.prop.predict_proba(X)[:, 1]
        mu_t, _ = self.t_learner.predict(X)
        taus = []
        for i, (m0, m1) in enumerate(self.tau_models):
            t0 = m0.predict(X)
            t1 = m1.predict(X)
            taus.append(e * t0 + (1 - e) * t1)
        tau = np.column_stack(taus)
        mu = mu_t.copy()
        mu[:, 1:] = mu[:, 0:1] + tau
        return mu, tau


class RLearner(CATEModel):
    name = "r_learner"

    def __init__(self, settings: Settings):
        self.settings = settings
        self.m_y: OutcomeModel | None = None
        self.m_e: LogisticRegression | None = None
        self.tau_m: OutcomeModel | None = None
        self.n_arms = 2

    def fit(self, frame: CausalFrame) -> "RLearner":
        self.n_arms = 2 if frame.n_arms >= 2 else frame.n_arms
        binary = _is_binary(frame.y)
        self.m_y = OutcomeModel(self.settings, binary).fit(frame.X, frame.y)
        self.m_e = LogisticRegression(max_iter=400)
        tbin = (frame.t > 0).astype(int)
        self.m_e.fit(frame.X, tbin)
        yres = frame.y - self.m_y.predict(frame.X)
        tres = tbin - self.m_e.predict_proba(frame.X)[:, 1]
        tres = np.clip(tres, -0.99, 0.99)
        # residual-on-residual: yres ≈ tau * tres
        target = yres / (tres + 1e-6)
        w = tres**2
        self.tau_m = OutcomeModel(self.settings, False)
        self.tau_m.m.set_params(n_estimators=min(self.settings.n_estimators, 150))
        self.tau_m.fit(frame.X, target)
        self._w = w
        self.binary = binary
        self._mu0 = OutcomeModel(self.settings, binary).fit(frame.X[tbin == 0], frame.y[tbin == 0])
        return self

    def predict(self, X: Array) -> tuple[Array, Array]:
        tau = self.tau_m.predict(X)[:, None]
        mu0 = self._mu0.predict(X)[:, None]
        mu1 = mu0 + tau
        mu = np.hstack([mu0, mu1])
        if self.n_arms > 2:
            extra = np.repeat(mu1, self.n_arms - 2, axis=1)
            mu = np.hstack([mu, extra])
            tau = np.hstack([tau, extra - mu0])
        return mu, tau


class DRLearner(CATEModel):
    name = "dr_learner"

    def __init__(self, settings: Settings):
        self.settings = settings
        self.t = TLearner(settings)
        self.prop: LogisticRegression | None = None
        self.tau_m: OutcomeModel | None = None
        self.n_arms = 2

    def fit(self, frame: CausalFrame) -> "DRLearner":
        self.n_arms = frame.n_arms
        self.t.fit(frame)
        mu, _ = self.t.predict(frame.X)
        self.prop = LogisticRegression(max_iter=400)
        tbin = (frame.t > 0).astype(int)
        self.prop.fit(frame.X, tbin)
        e = np.clip(self.prop.predict_proba(frame.X)[:, 1], 1e-3, 1 - 1e-3)
        # pseudo-outcome for arm 1 vs 0
        a = 1 if self.n_arms > 1 else 0
        t = (frame.t == a).astype(float)
        # generalized DR for first treatment
        t1 = (frame.t == 1).astype(float) if self.n_arms > 1 else tbin.astype(float)
        t0 = (frame.t == 0).astype(float)
        mu0, mu1 = mu[:, 0], mu[:, 1] if mu.shape[1] > 1 else mu[:, 0]
        e1 = e
        psi = (mu1 - mu0) + t1 * (frame.y - mu1) / e1 - t0 * (frame.y - mu0) / (1 - e1)
        self.tau_m = OutcomeModel(self.settings, False).fit(frame.X, psi)
        self._t = self.t
        return self

    def predict(self, X: Array) -> tuple[Array, Array]:
        mu, _ = self._t.predict(X)
        tau1 = self.tau_m.predict(X)
        tau = mu[:, 1:] - mu[:, 0:1]
        if tau.shape[1]:
            tau[:, 0] = tau1
            mu[:, 1] = mu[:, 0] + tau1
        return mu, tau


class UpliftForest(CATEModel):
    """RF on treated and control; tau = difference. Uplift-style via two forests."""

    name = "uplift_forest"

    def __init__(self, settings: Settings):
        self.settings = settings
        self.n_arms = 2
        self.forests: list[RandomForestRegressor] = []

    def fit(self, frame: CausalFrame) -> "UpliftForest":
        self.n_arms = frame.n_arms
        self.forests = []
        for a in range(self.n_arms):
            rf = RandomForestRegressor(
                n_estimators=min(200, self.settings.n_estimators),
                min_samples_leaf=20,
                random_state=self.settings.seed,
                n_jobs=-1,
            )
            mask = frame.t == a
            rf.fit(frame.X[mask], frame.y[mask])
            self.forests.append(rf)
        return self

    def predict(self, X: Array) -> tuple[Array, Array]:
        mu = np.column_stack([f.predict(X) for f in self.forests])
        return mu, mu[:, 1:] - mu[:, 0:1]


class CausalForest(CATEModel):
    """Honest-ish causal forest: forest on transformed outcome 2T(2T-1) style + T-learner blend."""

    name = "causal_forest"

    def __init__(self, settings: Settings):
        self.settings = settings
        self.n_arms = 2

    def fit(self, frame: CausalFrame) -> "CausalForest":
        self.n_arms = frame.n_arms
        tbin = (frame.t > 0).astype(float)
        # transformed outcome (Athey/Imbens style under known e~0.5; we residualize)
        e = np.clip(tbin.mean(), 0.05, 0.95)
        z = tbin * frame.y / e - (1 - tbin) * frame.y / (1 - e)
        self.rf = RandomForestRegressor(
            n_estimators=min(250, self.settings.n_estimators + 50),
            min_samples_leaf=25,
            max_depth=8,
            random_state=self.settings.seed,
            n_jobs=-1,
        )
        self.rf.fit(frame.X, z)
        self.t_learn = TLearner(self.settings).fit(frame)
        return self

    def predict(self, X: Array) -> tuple[Array, Array]:
        mu, tau_t = self.t_learn.predict(X)
        tau_cf = self.rf.predict(X)[:, None]
        tau = 0.5 * tau_t + 0.5 * np.repeat(tau_cf, tau_t.shape[1], axis=1) if tau_t.size else tau_cf
        if tau.shape[1] == 0:
            tau = tau_cf
        mu = mu.copy()
        if mu.shape[1] > 1:
            mu[:, 1:] = mu[:, 0:1] + tau
        return mu, tau


class _Net(nn.Module):
    def __init__(self, d: int, hidden: int, n_arms: int, propensity: bool, mmd: bool):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(d, hidden), nn.ELU(), nn.Linear(hidden, hidden), nn.ELU())
        self.heads = nn.ModuleList([nn.Sequential(nn.Linear(hidden, hidden // 2), nn.ELU(), nn.Linear(hidden // 2, 1)) for _ in range(n_arms)])
        self.prop = nn.Linear(hidden, 1) if propensity else None
        self.mmd = mmd
        self.n_arms = n_arms

    def forward(self, x):
        z = self.enc(x)
        mus = torch.cat([h(z) for h in self.heads], dim=1)
        p = torch.sigmoid(self.prop(z)).squeeze(-1) if self.prop is not None else None
        return mus, p, z


def _mmd(z, t):
    z1 = z[t > 0]
    z0 = z[t == 0]
    if len(z1) < 2 or len(z0) < 2:
        return z.sum() * 0
    d11 = torch.cdist(z1, z1).mean()
    d00 = torch.cdist(z0, z0).mean()
    d10 = torch.cdist(z1, z0).mean()
    return d11 + d00 - 2 * d10


class NeuralCATE(CATEModel):
    def __init__(self, settings: Settings, name: str, propensity: bool, mmd: bool):
        self.settings = settings
        self.name = name
        self.propensity = propensity
        self.mmd = mmd
        self.n_arms = 2
        self.device = torch.device("cpu")

    def fit(self, frame: CausalFrame) -> "NeuralCATE":
        self.n_arms = frame.n_arms
        d = frame.X.shape[1]
        net = _Net(d, self.settings.neural_hidden, self.n_arms, self.propensity, self.mmd)
        opt = torch.optim.Adam(net.parameters(), lr=1e-3)
        X = torch.tensor(frame.X, dtype=torch.float32)
        y = torch.tensor(frame.y, dtype=torch.float32)
        t = torch.tensor(frame.t, dtype=torch.long)
        binary = _is_binary(frame.y)
        n = len(frame.y)
        batch = min(256, n)
        net.train()
        for epoch in range(self.settings.neural_epochs):
            perm = torch.randperm(n)
            for i in range(0, n, batch):
                idx = perm[i : i + batch]
                mus, p, z = net(X[idx])
                ya = mus[torch.arange(len(idx)), t[idx]]
                if binary:
                    loss = nn.functional.binary_cross_entropy_with_logits(ya, y[idx])
                    # store as probabilities at predict via sigmoid
                else:
                    loss = nn.functional.mse_loss(ya, y[idx])
                if p is not None:
                    loss = loss + 0.5 * nn.functional.binary_cross_entropy(p.clamp(1e-4, 1 - 1e-4), (t[idx] > 0).float())
                if self.mmd:
                    loss = loss + 0.1 * _mmd(z, t[idx])
                opt.zero_grad()
                loss.backward()
                opt.step()
        self.net = net.eval()
        self.binary = binary
        self._mean = frame.X.mean(0)
        self._std = frame.X.std(0) + 1e-6
        return self

    def predict(self, X: Array) -> tuple[Array, Array]:
        xt = torch.tensor(X, dtype=torch.float32)
        with torch.no_grad():
            mus, _, _ = self.net(xt)
            if self.binary:
                mus = torch.sigmoid(mus)
            mu = mus.numpy()
        tau = mu[:, 1:] - mu[:, 0:1]
        return mu, tau


def tarnet(settings: Settings) -> NeuralCATE:
    return NeuralCATE(settings, "tarnet", False, False)


def cfrnet(settings: Settings) -> NeuralCATE:
    return NeuralCATE(settings, "cfrnet", False, True)


def dragonnet(settings: Settings) -> NeuralCATE:
    return NeuralCATE(settings, "dragonnet", True, False)


class UpliftRanker(CATEModel):
    """LightGBM trained to rank by a transformed-outcome proxy for Qini."""

    name = "uplift_ranker"

    def __init__(self, settings: Settings):
        self.settings = settings

    def fit(self, frame: CausalFrame) -> "UpliftRanker":
        tbin = (frame.t > 0).astype(float)
        e = np.clip(tbin.mean(), 0.05, 0.95)
        z = tbin * frame.y / e - (1 - tbin) * frame.y / (1 - e)
        self.m = lgb.LGBMRegressor(
            n_estimators=self.settings.n_estimators,
            learning_rate=0.05,
            num_leaves=31,
            random_state=self.settings.seed,
            verbose=-1,
        )
        self.m.fit(frame.X, z)
        self.t_learn = TLearner(self.settings).fit(frame)
        return self

    def predict(self, X: Array) -> tuple[Array, Array]:
        mu, tau = self.t_learn.predict(X)
        z = self.m.predict(X)[:, None]
        if tau.shape[1]:
            tau = 0.4 * tau + 0.6 * np.repeat(z, tau.shape[1], axis=1)
            mu = mu.copy()
            mu[:, 1:] = mu[:, 0:1] + tau
        return mu, tau


class Stack(CATEModel):
    name = "stack"

    def __init__(self, settings: Settings, members: list[CATEModel]):
        self.settings = settings
        self.members = members
        self.w = None

    def fit(self, frame: CausalFrame) -> "Stack":
        val = frame.subset(frame.mask("val")) if frame.mask("val").any() else frame
        train = frame.subset(frame.mask("train")) if frame.mask("train").any() else frame
        preds = []
        fitted = []
        for m in self.members:
            m.fit(train)
            _, tau = m.predict(val.X)
            preds.append(tau[:, 0])
            fitted.append(m)
        P = np.column_stack(preds)
        # weights: correlation with transformed outcome on val
        tbin = (val.t > 0).astype(float)
        e = np.clip(tbin.mean(), 0.05, 0.95)
        z = tbin * val.y / e - (1 - tbin) * val.y / (1 - e)
        corr = np.array([np.corrcoef(P[:, j], z)[0, 1] if np.std(P[:, j]) > 1e-9 else 0.0 for j in range(P.shape[1])])
        corr = np.nan_to_num(corr, nan=0.0)
        corr = np.clip(corr, 0, None)
        if corr.sum() == 0:
            corr = np.ones_like(corr)
        self.w = corr / corr.sum()
        self.members = fitted
        self.n_arms = frame.n_arms
        return self

    def predict(self, X: Array) -> tuple[Array, Array]:
        mus, taus = [], []
        for m in self.members:
            mu, tau = m.predict(X)
            mus.append(mu)
            taus.append(tau)
        tau = sum(w * t for w, t in zip(self.w, taus))
        mu0 = mus[0][:, 0:1]
        mu = np.hstack([mu0, mu0 + tau])
        return mu, tau


def conformal_bounds(tau: Array, cal_resid: float, alpha: float) -> tuple[Array, Array]:
    q = cal_resid  # already a quantile
    lo = tau - q
    hi = tau + q
    return lo, hi


def calibration_quantile(frame: CausalFrame, mu: Array, alpha: float, tau: Array | None = None) -> float:
    """Conformal half-width. Prefer transformed-outcome residuals when tau is given."""
    if tau is not None:
        tbin = (frame.t > 0).astype(float)
        e = float(np.clip(tbin.mean(), 0.05, 0.95))
        z = tbin * frame.y / e - (1 - tbin) * frame.y / (1 - e)
        hat = tau[:, 0] if tau.ndim > 1 else tau
        resid = np.abs(z - hat)
        return float(np.quantile(resid, min(1.0, 1 - alpha)))
    yhat = mu[np.arange(len(frame.t)), np.clip(frame.t, 0, mu.shape[1] - 1)]
    resid = np.abs(frame.y - yhat)
    return float(np.quantile(resid, min(1.0, 1 - alpha)))


FACTORY: dict[str, Callable[[Settings], CATEModel]] = {
    "s_learner": SLearner,
    "t_learner": TLearner,
    "x_learner": XLearner,
    "r_learner": RLearner,
    "dr_learner": DRLearner,
    "uplift_forest": UpliftForest,
    "causal_forest": CausalForest,
    "tarnet": tarnet,
    "cfrnet": cfrnet,
    "dragonnet": dragonnet,
    "uplift_ranker": UpliftRanker,
}


def make_model(name: str, settings: Settings) -> CATEModel:
    if name == "stack":
        members = [TLearner(settings), XLearner(settings), UpliftRanker(settings), dragonnet(settings)]
        return Stack(settings, members)
    fn = FACTORY[name]
    return fn(settings)
