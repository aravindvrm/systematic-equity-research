"""Transaction cost models, per venue.

This module is the main place where asset class actually changes the answer.
The strategy code above it is asset-agnostic; the cost model below it is not.

Round-trip cost is what determines which strategy families are viable at all:

    IBKR Pro US equities   ~2-3 bp   -> intraday and multi-day both viable
    IBKR ZEROHASH crypto  ~38 bp     -> multi-day holds only

A signal with a 20bp average edge is a business in equities and a guaranteed
loss in crypto. Same signal, same code, different venue.
"""
from __future__ import annotations

from dataclasses import dataclass

BPS = 1e-4


@dataclass(frozen=True)
class CostModel:
    """Per-trade cost, expressed as a fraction of traded notional.

    commission_bps: broker commission, one side.
    spread_bps:     half-spread paid crossing the book, one side.
    slippage_bps:   market impact / adverse selection beyond the quote, one side.
    min_commission: absolute floor per order, in account currency.
    """
    name: str
    commission_bps: float
    spread_bps: float
    slippage_bps: float
    min_commission: float = 0.0

    @property
    def one_way_bps(self) -> float:
        return self.commission_bps + self.spread_bps + self.slippage_bps

    @property
    def round_trip_bps(self) -> float:
        return 2 * self.one_way_bps

    def cost(self, notional: float) -> float:
        """Cost in currency for trading `notional` (absolute value) on one side.

        Commission is floored at min_commission; spread and slippage are pure
        percentages of notional and have no floor.
        """
        notional = abs(notional)
        if notional == 0:
            return 0.0
        commission = max(notional * self.commission_bps * BPS, self.min_commission)
        impact = notional * (self.spread_bps + self.slippage_bps) * BPS
        return commission + impact

    def breakeven_move_bps(self) -> float:
        """How far the asset must move in your favour just to cover a round trip."""
        return self.round_trip_bps


# IBKR Pro TIERED US equities: $0.0035/share, $0.35 minimum per order, capped at
# 1% of trade value. The $0.35 per-order minimum -- not the per-share rate -- is
# what binds for a small account: it is 14bp on a $250 order and 1.4bp on a
# $2,500 one. See position_sizing.py for the full grid.
#
# NOTE: min_commission is in account currency, so its bp impact depends on order
# size, which this fractional model cannot see. Backtests here assume orders are
# large enough (>~$2,000) that the per-share rate dominates. Below that, add the
# extra drag from position_sizing.py by hand.
IBKR_US_EQUITY = CostModel(
    name="IBKR Pro US equity",
    commission_bps=0.7,
    spread_bps=1.0,
    slippage_bps=0.5,
    min_commission=0.35,
)

# Same venue, but sized for a small account where the $0.35 per-order minimum
# dominates -- roughly a $1,000 order. Use this to check whether a strategy
# survives at the size you will actually trade it.
IBKR_US_EQUITY_SMALL = CostModel(
    name="IBKR Pro US equity (small orders)",
    commission_bps=3.5,
    spread_bps=1.0,
    slippage_bps=0.5,
    min_commission=0.35,
)

# IBKR crypto via ZEROHASH: 0.18% per side = 18bp commission, no markup on the
# spread but crypto books are thinner, so spread+slippage is not negligible.
IBKR_CRYPTO = CostModel(
    name="IBKR ZEROHASH crypto",
    commission_bps=18.0,
    spread_bps=1.0,
    slippage_bps=1.0,
    min_commission=1.75,
)

# For sanity checks only -- never report a headline number from this one.
ZERO_COST = CostModel(name="zero cost (DEBUG ONLY)", commission_bps=0, spread_bps=0, slippage_bps=0)

REGISTRY = {c.name: c for c in (IBKR_US_EQUITY, IBKR_US_EQUITY_SMALL, IBKR_CRYPTO, ZERO_COST)}
