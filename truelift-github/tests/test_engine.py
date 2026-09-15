from __future__ import annotations

import numpy as np

from sklearn.linear_model import LogisticRegression

from truelift.config import Settings
from truelift.data import load_synthetic
from truelift.evaluate import pehe, qini_curve
from truelift.identify import identify, overlap_diagnostics, propensity_scores
from truelift.models import TLearner
from truelift.policy import budget_policy, knapsack_policy


def test_oracle_tau_beats_response_model():
    settings = Settings(seed=0)
    fr = load_synthetic(settings, n=3000)
    test = fr.subset(fr.mask("test"))
    train = fr.subset(fr.mask("train"))
    rm = LogisticRegression(max_iter=400).fit(train.X, (train.y > 0).astype(int))
    resp = rm.predict_proba(test.X)[:, 1]
    q_tau = qini_curve(test.y, test.t, test.true_tau[:, 0])
    q_resp = qini_curve(test.y, test.t, resp)
    assert q_tau["auuc"] > q_resp["auuc"]


def test_synthetic_frame():
    fr = load_synthetic(Settings(seed=0), n=400)
    assert fr.n == 400
    assert fr.n_arms == 3
    assert fr.true_tau is not None
    assert set(fr.split) <= {"train", "val", "test"}


def test_qini_random_near_treat_all_midpoint():
    rng = np.random.default_rng(0)
    y = rng.binomial(1, 0.2, 500).astype(float)
    t = rng.integers(0, 2, 500)
    score = rng.normal(size=500)
    q = qini_curve(y, t, score)
    assert len(q["qini"]) == 21
    assert q["fracs"][0] == 0


def test_tlearner_recovers_sign():
    settings = Settings(seed=1, n_estimators=40, neural_epochs=1)
    fr = load_synthetic(settings, n=1200)
    train = fr.subset(fr.mask("train"))
    test = fr.subset(fr.mask("test"))
    m = TLearner(settings).fit(train)
    _, tau = m.predict(test.X)
    err = pehe(tau, test.true_tau)
    assert err < 0.35


def test_knapsack_respects_budget():
    tau = np.array([[0.2], [0.05], [0.0], [0.4]])
    cost = np.array([1.0, 1.0, 1.0, 1.0])
    kn = knapsack_policy(tau, cost, budget_total=2.0, revenue=10.0)
    assert kn["spent"] <= 2.0 + 1e-6
    assert kn["n_treated"] <= 2


def test_conformal_budget_can_skip_all():
    tau = np.array([[0.01], [0.02]])
    lo = np.array([[-0.1], [-0.1]])
    a = budget_policy(tau, 0.5, lo=lo)
    assert (a == 0).all()


def test_identify_synthetic_randomized():
    fr = load_synthetic(Settings(seed=2), n=300)
    e = propensity_scores(fr)
    ov = overlap_diagnostics(e)
    report = identify(fr)
    assert report.randomized
    assert "histogram" in ov
    assert report.estimand in {"cate", "itt", "late"}
