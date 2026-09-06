"""Broker-layer tests. No credentials or network required.

The safety tests matter most: they assert that the module cannot send an order
by accident.
"""
import pandas as pd
import pytest

from algo.broker import OrderPlan, RiskLimits, plan_orders, submit_orders


def w(**kw):
    return pd.Series(kw, dtype="float64")


# Toy 2-asset portfolios exceed the realistic 35% concentration cap, so tests
# exercising the band logic pass permissive limits explicitly. The concentration
# cap gets its own test below.
LOOSE = RiskLimits(max_order_pct=1.0, max_position_pct=1.0)


def test_no_orders_when_inside_band():
    plan = plan_orders(w(SPY=0.5, TLT=0.5), 10_000, w(SPY=0.52, TLT=0.48), band=0.10, limits=LOOSE)
    assert plan.orders == []
    assert plan.ok


def test_orders_generated_outside_band():
    plan = plan_orders(w(SPY=0.5, TLT=0.5), 10_000, w(SPY=0.20, TLT=0.80), band=0.10, limits=LOOSE)
    assert plan.ok
    sides = {o.symbol: o.side for o in plan.orders}
    assert sides == {"SPY": "buy", "TLT": "sell"}
    assert all(o.notional == pytest.approx(3000, rel=0.01) for o in plan.orders)


def test_opening_from_empty_portfolio():
    plan = plan_orders(w(SPY=0.6, TLT=0.4), 1_000, None, band=0.10, limits=LOOSE)
    assert {o.side for o in plan.orders} == {"buy"}
    assert sum(o.notional for o in plan.orders) == pytest.approx(1000, rel=0.01)


# ---------------------------------------------------------------- safety ----

def test_leverage_is_refused():
    """Target weights summing above 1.0 must block, not lever."""
    plan = plan_orders(w(SPY=0.8, TLT=0.8), 10_000, None, band=0.10)
    assert not plan.ok
    assert any("gross exposure" in b for b in plan.blocked)
    assert plan.orders == []


def test_oversized_single_position_is_refused():
    plan = plan_orders(w(SPY=0.9, TLT=0.1), 10_000, None, band=0.10)
    assert not plan.ok
    assert any("max_position_pct" in b for b in plan.blocked)


def test_runaway_order_count_is_refused():
    many = {f"S{i}": 0.02 for i in range(40)}
    plan = plan_orders(pd.Series(many), 100_000, None, band=0.001,
                       limits=RiskLimits(max_orders_per_run=15, max_order_pct=1.0))
    assert not plan.ok
    assert any("runaway" in b for b in plan.blocked)


def test_order_value_is_capped():
    plan = plan_orders(w(SPY=1.0), 100_000, None, band=0.10,
                       limits=RiskLimits(max_order_pct=0.25, max_position_pct=1.0))
    assert plan.orders[0].notional == 25_000   # 25% of equity, not an absolute
    assert any("capped" in n for n in plan.notes)


def test_zero_equity_is_refused():
    plan = plan_orders(w(SPY=1.0), 0.0, None)
    assert not plan.ok


def test_dry_run_submits_nothing():
    """The default path must never send an order."""
    plan = plan_orders(w(SPY=0.5, TLT=0.5), 10_000, None, band=0.10, limits=LOOSE)
    assert plan.orders

    class ExplodingClient:
        def submit_order(self, *a, **k):
            raise AssertionError("submit_order called during a dry run!")

    assert submit_orders(ExplodingClient(), plan) == []          # default
    assert submit_orders(ExplodingClient(), plan, live=False) == []


def test_blocked_plan_cannot_be_submitted_even_when_live():
    plan = plan_orders(w(SPY=0.8, TLT=0.8), 10_000, None)
    assert not plan.ok

    class ExplodingClient:
        def submit_order(self, *a, **k):
            raise AssertionError("submitted a blocked plan!")

    with pytest.raises(RuntimeError, match="blocked"):
        submit_orders(ExplodingClient(), plan, live=True)


def test_connect_requires_explicit_paper_keys(monkeypatch):
    """Missing credentials must raise, not fail obscurely later.

    NOTE: connect() loads algo/.env, which on a configured machine repopulates
    the very vars this test clears. We neutralize load_dotenv so the test
    asserts the code path rather than the developer's local file state.
    """
    import dotenv
    monkeypatch.setattr(dotenv, "load_dotenv", lambda *a, **k: False)
    monkeypatch.delenv("ALPACA_PAPER_KEY", raising=False)
    monkeypatch.delenv("ALPACA_PAPER_SECRET", raising=False)
    from algo import broker
    with pytest.raises(RuntimeError, match="ALPACA_PAPER_KEY"):
        broker.connect(paper=True)


def test_cold_start_opens_positions_below_the_band():
    """A 3% target against a 10% band must still be opened from empty.

    The band suppresses pointless rebalancing; it must not prevent the portfolio
    from being constructed. Without the cold-start rule, small sleeves would
    never be established.
    """
    target = w(SPY=0.30, TLT=0.20, GLD=0.03, IEF=0.03)
    plan = plan_orders(target, 10_000, None, band=0.10, limits=LOOSE)
    opened = {o.symbol for o in plan.orders}
    assert opened == {"SPY", "TLT", "GLD", "IEF"}, "small sleeves were not opened"
    assert all(o.side == "buy" for o in plan.orders)


def test_band_still_suppresses_small_rebalances_once_held():
    """Once a position exists, sub-band drift must NOT generate an order."""
    target = w(SPY=0.30, GLD=0.06)
    current = w(SPY=0.28, GLD=0.03)     # both drifts under a 10% band
    plan = plan_orders(target, 10_000, current, band=0.10, limits=LOOSE)
    assert plan.orders == []


def test_order_cap_scales_with_equity():
    """The same limits must behave sensibly at $1k and at $100k.

    Regression test: an absolute $250 cap left a $100,000 account 2.5% invested.
    """
    lim = RiskLimits(max_position_pct=1.0)
    fractions = []
    for equity in (1_000, 100_000):
        plan = plan_orders(w(SPY=0.5, TLT=0.5), equity, None, band=0.10, limits=lim)
        fractions.append(sum(o.notional for o in plan.orders) / equity)
    # The point is scale-invariance: the same limits must deploy the same
    # FRACTION of a $1k and a $100k account. An absolute cap did not.
    assert fractions[0] == pytest.approx(fractions[1], rel=1e-6), fractions

    # And with a cap above the target weights, the full portfolio deploys.
    generous = RiskLimits(max_order_pct=1.0, max_position_pct=1.0)
    for equity in (1_000, 100_000):
        plan = plan_orders(w(SPY=0.5, TLT=0.5), equity, None, band=0.10, limits=generous)
        assert sum(o.notional for o in plan.orders) == pytest.approx(equity, rel=0.01)
