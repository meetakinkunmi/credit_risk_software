"""Fast unit tests on synthetic data (no real dataset needed).  Run: pytest -q"""
import numpy as np
import pandas as pd

from src.cost import CostOptimizer
from src.data import GermanCreditData
from src.fairness import ReweighingMitigator
from src.monitor import DriftMonitor
from src.scorecard import Scorecard, WoEBinner
from src.schema import TARGET
from src.synthetic import make_synthetic_raw


def _df():
    return GermanCreditData.prepare(make_synthetic_raw(800, seed=1))


def test_prepare_creates_target_and_audit_columns():
    df = _df()
    assert TARGET in df and {"sex", "age_group", "monthly_payment"} <= set(df.columns)


def test_invalid_code_is_rejected():
    raw = make_synthetic_raw(5)
    raw.loc[0, "housing"] = "A999"
    try:
        GermanCreditData.prepare(raw)
        assert False
    except ValueError:
        pass


def test_woe_iv_positive_for_informative_feature():
    df = _df()
    b = WoEBinner().fit(df, df[TARGET], ["checking_status", "residence_since"])
    assert b.iv_["checking_status"] > b.iv_["residence_since"] >= 0


def test_scorecard_points_sum_to_score():
    df = _df()
    sc = Scorecard(columns=["checking_status", "credit_history", "duration_months", "age"]).fit(df, df[TARGET])
    pts = sc.points_matrix(df).sum(axis=1).values
    assert np.allclose(pts, sc.score(df), atol=1e-6)
    assert np.allclose(sc.pd_to_score(sc.predict_pd(df)), sc.score(df), atol=1e-4)


def test_theoretical_threshold_is_one_sixth():
    assert abs(CostOptimizer(5, 1).theoretical_threshold - 1 / 6) < 1e-12


def test_cost_optimizer_beats_or_ties_any_threshold():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 500)
    p = np.clip(0.3 * y + rng.random(500) * 0.7, 0, 1)
    co = CostOptimizer()
    t = co.optimal_threshold(y, p)
    assert 0 < t < 1
    assert co.curve(y, p)["total_cost"].min() <= co.total_cost(y, p < 0.5)


def test_reweighing_makes_label_independent_of_group():
    df = _df()
    w = ReweighingMitigator("sex").weights(df, df[TARGET])
    d = pd.DataFrame({"g": df["sex"], "y": df[TARGET], "w": w})
    rates = d.groupby("g").apply(lambda s: np.average(s.y, weights=s.w))
    assert rates.max() - rates.min() < 1e-9


def test_psi_zero_for_identical_distribution():
    assert DriftMonitor.psi([0.2, 0.3, 0.5], [0.2, 0.3, 0.5]) == 0.0
