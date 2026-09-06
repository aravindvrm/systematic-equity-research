"""Regression tests for the 2026-09-05 methodology audit.

Each test pins a bug that was found, or a property of the corrected framework
that must not silently regress.
"""
import numpy as np
import pandas as pd
import pytest

from algo import backtest, backtest2, evaluation, metrics, strategies
from algo.costs import IBKR_US_EQUITY, ZERO_COST


@pytest.fixture
def panel():
    rng = np.random.default_rng(3)
    idx = pd.bdate_range("2015-01-01", periods=1500)
    px = 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.012, (len(idx), 8)), axis=0))
    return pd.DataFrame(px, index=idx, columns=list("ABCDEFGH"))


def test_constant_target_still_pays_turnover(panel):
    """THE ENGINE BUG.

    v1 computed turnover from TARGET changes, so a constant equal-weight target
    paid exactly zero cost while implicitly being rebalanced for free. Holding
    fixed weights requires trading as prices drift, and that costs money.
    """
    w = strategies.equal_weight(panel)
    v1 = backtest.run_legacy(panel, w, cost_model=IBKR_US_EQUITY)
    v2 = backtest2.run(panel, w, cost_model=IBKR_US_EQUITY)

    assert v1.turnover.sum() == pytest.approx(0.0, abs=1e-12), (
        "v1 is expected to report zero turnover here -- that is the bug it had")
    assert v2.turnover.sum() > 0.5, (
        "v2 must charge for rebalancing a constant target back against drift")
    assert v2.costs.sum() > 0


def test_v2_gross_agrees_with_v1_gross(panel):
    """The engines must agree on GROSS return; they differ only on cost."""
    w = strategies.inverse_volatility(panel, 60)
    v1 = backtest.run_legacy(panel, w, cost_model=ZERO_COST)
    v2 = backtest2.run(panel, w, cost_model=ZERO_COST)
    assert metrics.sharpe(v1.gross_returns) == pytest.approx(
        metrics.sharpe(v2.gross_returns), abs=0.05)


def test_sortino_matches_reference():
    """THE METRIC BUG.

    Sortino used the std of the NEGATIVE SUBSET, which ran ~16% high. The
    standard definition is the RMS of the full series with upside clipped to 0.
    """
    ep = pytest.importorskip("empyrical")
    rng = np.random.default_rng(0)
    r = pd.Series(rng.normal(0.0004, 0.011, 2000),
                  index=pd.bdate_range("2015-01-01", periods=2000))
    assert metrics.sortino(r) == pytest.approx(ep.sortino_ratio(r), rel=1e-6)


def test_adding_the_core_to_itself_adds_nothing():
    """Validates the marginal framework.

    A sleeve that IS the core cannot improve the core. If this ever returns a
    positive gain, the evaluation is manufacturing alpha out of nothing.
    """
    rng = np.random.default_rng(11)
    idx = pd.bdate_range("2015-01-01", periods=2000)
    core = pd.Series(rng.normal(0.0004, 0.010, len(idx)), index=idx)
    ev = evaluation.evaluate(core, core, label="self")
    assert ev["corr_to_core"] == pytest.approx(1.0, abs=1e-9)
    assert ev["alpha_ann"] == pytest.approx(0.0, abs=1e-6)
    assert abs(ev["delta_at_10pct"]) < 1e-9


def test_pure_de_risking_shows_no_genuine_gain():
    """THE FRAMING BUG.

    A 'strategy' that is just 60% of the core plus cash raises portfolio Sharpe
    by lowering volatility. That is free -- anyone can hold less equity. The
    beta-matched control must score it at ~zero, not reward it.
    """
    rng = np.random.default_rng(5)
    idx = pd.bdate_range("2010-01-01", periods=3000)
    core = pd.Series(rng.normal(0.0004, 0.011, len(idx)), index=idx)
    cash = pd.Series(0.02 / 252, index=idx)
    sleeve = 0.6 * core + 0.4 * cash          # pure de-risking, no skill

    beta = evaluation.alpha_beta(sleeve, core)["beta"]
    assert beta == pytest.approx(0.6, abs=0.02)

    matched = beta * core + (1 - beta) * cash
    gain = (metrics.sharpe(0.9 * core + 0.1 * sleeve)
            - metrics.sharpe(0.9 * core + 0.1 * matched))
    assert abs(gain) < 0.005, (
        f"de-risking scored {gain:+.4f} of 'genuine gain' -- the control is "
        "not removing the beta effect")

    # And it DOES raise naive standalone Sharpe, which is exactly why the naive
    # comparison was misleading.
    assert metrics.sharpe(sleeve) > metrics.sharpe(core)


def test_optimal_weight_is_bounded():
    """No shorting the core, no levering the sleeve -- cash account."""
    rng = np.random.default_rng(7)
    idx = pd.bdate_range("2015-01-01", periods=1200)
    core = pd.Series(rng.normal(0.0002, 0.010, len(idx)), index=idx)
    great = pd.Series(rng.normal(0.0030, 0.008, len(idx)), index=idx)
    awful = pd.Series(rng.normal(-0.0020, 0.010, len(idx)), index=idx)
    assert 0.0 <= evaluation.optimal_weight(great, core) <= 1.0
    assert 0.0 <= evaluation.optimal_weight(awful, core) <= 1.0


def test_random_signals_produce_fake_alpha_through_a_vol_targeted_stack():
    """THE CONTROL FAILURE that the null floor exists to catch.

    A beta-matched core/cash mix does NOT control a vol-targeted sleeve. Vol
    targeting cuts exposure when volatility spikes -- exactly when the core
    crashes -- so the sleeve's beta is time-varying and conditionally low when
    it matters. A static full-sample beta cannot represent that, and the
    residual appears as alpha.

    This test builds a core with volatility CLUSTERING (as real markets have),
    runs a zero-information sleeve that de-risks in high-vol periods, and
    asserts that the beta-matched control still reports positive "alpha" --
    proving the control is insufficient on its own.
    """
    rng = np.random.default_rng(17)
    n = 4000
    idx = pd.bdate_range("2005-01-01", periods=n)

    # Two-state volatility: calm and crisis. Crises are where a static beta lies.
    vol = np.where(rng.random(n) < 0.12, 0.030, 0.008)
    core = pd.Series(rng.normal(0.0004, 1.0, n) * vol, index=idx)

    # Zero-information sleeve: same returns as the core, but scaled DOWN when
    # trailing vol is high. It knows nothing about direction.
    trail = core.rolling(60, min_periods=20).std().bfill()
    scale = (0.010 / trail).clip(upper=1.0)
    sleeve = core * scale.shift(1).fillna(1.0)

    beta = evaluation.alpha_beta(sleeve, core)["beta"]
    cash = pd.Series(0.02 / 252, index=idx)
    matched = beta * core + (1 - beta) * cash
    gain = (metrics.sharpe(0.9 * core + 0.1 * sleeve)
            - metrics.sharpe(0.9 * core + 0.1 * matched))

    assert gain > 0.0, (
        "the vol-targeted sleeve should beat its own beta-matched control "
        "despite having zero information -- that is the artifact")

    # And percentile_vs_null must place a value inside its null at <95.
    pct = evaluation.percentile_vs_null(0.029, [0.028, 0.029, 0.030, 0.031, 0.033])
    assert pct < 95


def test_percentile_vs_null_basic():
    vals = [0.01, 0.02, 0.03, 0.04, 0.05]
    assert evaluation.percentile_vs_null(0.06, vals) == 100.0
    assert evaluation.percentile_vs_null(0.00, vals) == 0.0
    assert evaluation.percentile_vs_null(0.035, vals) == pytest.approx(60.0)
