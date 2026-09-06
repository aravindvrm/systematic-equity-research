"""Tests for the backtest engine.

The lookahead tests are the ones that matter. They are constructed so that a
naive engine (one that forgets to lag signals) produces an impossibly good
result, and a correct engine produces roughly nothing.
"""
import numpy as np
import pandas as pd
import pytest

from algo import backtest, metrics
from algo.costs import ZERO_COST, IBKR_US_EQUITY, IBKR_CRYPTO


@pytest.fixture
def random_walk():
    rng = np.random.default_rng(42)
    idx = pd.bdate_range("2015-01-01", periods=1500)
    px = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, len(idx))))
    return pd.DataFrame({"A": px}, index=idx)


def test_lookahead_is_prevented(random_walk):
    """Same-bar information must not be tradeable.

    The weight at bar t is set to the sign of the return INTO bar t -- a value
    you only know once bar t has closed. An engine that forgets to lag would
    apply that weight to bar t's own return and capture every single up-move,
    giving an astronomical Sharpe. A correct engine defers it to bar t+1, where
    on a random walk it predicts nothing.

    If someone deletes the .shift(1) in backtest.run(), this test fails loudly.
    """
    prices = random_walk
    same_bar_return = prices.pct_change()
    cheating_weights = (same_bar_return > 0).astype(float)

    res = backtest.run(prices, cheating_weights, cost_model=ZERO_COST)
    sharpe = metrics.sharpe(res.returns)

    assert abs(sharpe) < 1.0, (
        f"Sharpe {sharpe:.2f} on a random walk implies same-bar information is "
        "leaking through. The .shift(1) in backtest.run() is the guard."
    )


def test_engine_can_still_profit_from_a_genuine_signal(random_walk):
    """Control for the test above.

    Here the weight at bar t is the sign of the return from t to t+1 -- a real
    forecast. After the engine's lag it lands on exactly the bar it predicts, so
    it should win big. This proves the lookahead test fails for the right
    reason: the engine rejects future information, it does not reject signal.
    """
    prices = random_walk
    forecast = (prices.pct_change().shift(-1) > 0).astype(float)
    res = backtest.run(prices, forecast, cost_model=ZERO_COST)
    assert metrics.sharpe(res.returns) > 5.0


def test_costs_reduce_returns(random_walk):
    """Net return must be strictly worse than gross whenever there is turnover."""
    prices = random_walk
    rng = np.random.default_rng(0)
    w = pd.DataFrame(rng.integers(0, 2, (len(prices), 1)).astype(float),
                     index=prices.index, columns=prices.columns)

    free = backtest.run(prices, w, cost_model=ZERO_COST)
    paid = backtest.run(prices, w, cost_model=IBKR_US_EQUITY)

    assert paid.equity.iloc[-1] < free.equity.iloc[-1]
    assert paid.costs.sum() > 0


def test_crypto_costs_dominate_equity_costs(random_walk):
    """Same signal, same code, different venue -- crypto must cost far more."""
    prices = random_walk
    rng = np.random.default_rng(1)
    w = pd.DataFrame(rng.integers(0, 2, (len(prices), 1)).astype(float),
                     index=prices.index, columns=prices.columns)

    eq = backtest.run(prices, w, cost_model=IBKR_US_EQUITY)
    cr = backtest.run(prices, w, cost_model=IBKR_CRYPTO)

    assert cr.costs.sum() > 5 * eq.costs.sum()


def test_buy_and_hold_matches_price_return(random_walk):
    """Buy and hold should track the asset almost exactly (one bar of lag)."""
    prices = random_walk
    res = backtest.buy_and_hold(prices, "A", cost_model=ZERO_COST)
    # Weight is 1 from bar 0, so it is held over every return from bar 1 onward,
    # which compounds to exactly the full price move from bar 0 to the end.
    asset_total = prices["A"].iloc[-1] / prices["A"].iloc[0] - 1
    bt_total = res.equity.iloc[-1] / res.equity.iloc[0] - 1
    assert bt_total == pytest.approx(asset_total, rel=0.02)


def test_zero_weights_produce_flat_equity(random_walk):
    prices = random_walk
    w = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    res = backtest.run(prices, w, cost_model=IBKR_US_EQUITY)
    assert res.equity.std() == pytest.approx(0.0, abs=1e-9)


def test_deflated_sharpe_haircuts_multiple_trials():
    """Trying more variants must lower the Sharpe you are entitled to claim."""
    raw = 1.5
    d1 = metrics.deflated_sharpe(raw, n_trials=1, n_obs=1000)
    d100 = metrics.deflated_sharpe(raw, n_trials=100, n_obs=1000)
    d1000 = metrics.deflated_sharpe(raw, n_trials=1000, n_obs=1000)
    assert d1 == raw
    assert d100 < d1
    assert d1000 < d100


def test_max_drawdown_sign_and_magnitude():
    eq = pd.Series([100, 120, 60, 90])
    assert metrics.max_drawdown(eq) == pytest.approx(-0.5)


def test_sharpe_of_zero_vol_is_zero():
    assert metrics.sharpe(pd.Series([0.001] * 100)) == 0.0


def test_engine_output_is_float_not_object(random_walk):
    """Guard against dtype poisoning.

    pd.NA anywhere upstream promotes frames to object dtype. Arithmetic still
    "works", so nothing fails until something like scipy.stats.skew is handed an
    object array and raises an error that points nowhere near the cause.
    """
    prices = random_walk
    w = pd.DataFrame({"A": [0.0, 1.0] * (len(prices) // 2)}, index=prices.index)
    w = w.astype(object)  # simulate the poisoning
    res = backtest.run(prices, w, cost_model=ZERO_COST)
    for name, s in [("returns", res.returns), ("equity", res.equity),
                    ("turnover", res.turnover), ("gross", res.gross_returns)]:
        assert s.dtype.kind == "f", f"{name} is {s.dtype}, expected float"


def test_strategies_return_float_weights():
    """Every strategy must emit float weights, not object."""
    from algo import strategies
    rng = np.random.default_rng(7)
    idx = pd.bdate_range("2015-01-01", periods=800)
    px = pd.DataFrame(
        100 * np.exp(np.cumsum(rng.normal(0, 0.01, (len(idx), 4)), axis=0)),
        index=idx, columns=list("ABCD"))
    for fn in [strategies.equal_weight, strategies.time_series_momentum,
               strategies.cross_sectional_momentum, strategies.sma_crossover,
               strategies.inverse_volatility]:
        w = fn(px)
        assert all(d.kind == "f" for d in w.dtypes), f"{fn.__name__} emitted {set(w.dtypes)}"


def test_latest_prices_ignores_unpopulated_final_row(monkeypatch):
    """A price panel's last row is often today's unfinished bar: all NaN.

    Regression test -- this silently produced NaN spot prices in the options
    collector, which made every moneyness-filtered metric come back empty while
    the unfiltered ones still looked correct.
    """
    from algo import data as data_mod
    idx = pd.bdate_range("2026-08-01", periods=5)
    panel = pd.DataFrame({"AAA": [1.0, 2.0, 3.0, 4.0, np.nan],
                          "BBB": [5.0, 6.0, 7.0, 8.0, np.nan]}, index=idx)
    monkeypatch.setattr(data_mod, "load_panel", lambda *a, **k: panel)
    out = data_mod.latest_prices(["AAA", "BBB"])
    assert out["AAA"] == 4.0 and out["BBB"] == 8.0
    assert not out.isna().any()


def test_refresh_does_not_truncate_cache(tmp_path, monkeypatch):
    """A short refresh must not destroy a long cached history.

    Regression test. `latest_prices()` refetches ~30 days with refresh=True. With
    a replace-on-write cache that silently overwrote years of daily bars with one
    month, for every ticker the options collector touched -- corrupting every
    backtest downstream while looking like it worked.
    """
    from algo import data as data_mod

    monkeypatch.setattr(data_mod, "CACHE_DIR", tmp_path)
    long_idx = pd.bdate_range("2020-01-01", periods=500)
    long_hist = pd.DataFrame({c: np.arange(500, dtype=float) for c in data_mod.BAR_COLUMNS},
                             index=long_idx)
    long_hist.index.name = "date"
    long_hist.to_parquet(tmp_path / "ZZZ_1d.parquet")

    short_idx = pd.bdate_range("2026-08-05", periods=23)
    short = pd.DataFrame({c.capitalize(): np.arange(23, dtype=float)
                          for c in data_mod.BAR_COLUMNS}, index=short_idx)
    monkeypatch.setattr(data_mod.yf, "download", lambda *a, **k: short)

    data_mod.load_bars("ZZZ", start="2026-08-01", refresh=True)
    merged = pd.read_parquet(tmp_path / "ZZZ_1d.parquet")
    assert len(merged) > 500, f"cache truncated to {len(merged)} rows"
    assert merged.index.min() == long_idx[0]
    assert merged.index.max() == short_idx[-1]
