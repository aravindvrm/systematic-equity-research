"""Alpaca broker integration.

SAFETY MODEL
------------
Everything here is DRY RUN by default. `plan_orders()` computes what should be
traded and returns it; `submit_orders()` refuses to send anything unless it is
passed live=True explicitly. There is no code path where a typo results in an
order being sent.

Layered limits, checked before anything is submitted:
    - max_order_value      no single order above this
    - max_orders_per_run   circuit breaker against runaway loops
    - max_gross_exposure   total target weight cannot exceed this
    - min_trade_value      skip trades too small to be worth the spread

PAPER vs LIVE
-------------
Paper and live use different base URLs AND different API keys. The `paper` flag
selects both. Alpaca's paper environment does NOT simulate: market impact, queue
position, price improvement, regulatory fees, or DIVIDENDS. That last one
matters here -- our backtest uses dividend-adjusted prices, so paper will
understate returns for any dividend-paying ETF. Expect a reconciliation gap and
do not treat it as a bug.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field

import pandas as pd

log = logging.getLogger(__name__)

PAPER_URL = "https://paper-api.alpaca.markets"
LIVE_URL = "https://api.alpaca.markets"


@dataclass
class RiskLimits:
    """Guardrails, expressed as FRACTIONS OF EQUITY wherever possible.

    An absolute dollar cap does not survive a change of account size: a $250
    limit that is a sensible circuit breaker on a $1,000 account silently
    truncates every order on a $100,000 one, leaving the portfolio 2.5%
    invested instead of 97%. Percentages scale; dollars do not.
    """
    max_order_pct: float = 0.25           # no single order above 25% of equity
    max_order_value: float | None = None  # optional ABSOLUTE cap, usually None
    max_orders_per_run: int = 15
    max_gross_exposure: float = 1.0
    min_trade_value: float = 5.0          # Alpaca supports fractional shares
    max_position_pct: float = 0.35        # no single name above this

    def order_cap(self, equity: float) -> float:
        pct_cap = self.max_order_pct * equity
        return min(pct_cap, self.max_order_value) if self.max_order_value else pct_cap


@dataclass
class PlannedOrder:
    symbol: str
    side: str              # "buy" | "sell"
    notional: float        # dollars (fractional shares)
    current_weight: float
    target_weight: float
    reason: str = ""

    def __str__(self) -> str:
        return (f"  {self.side.upper():4s} {self.symbol:6s} ${self.notional:>9,.2f}   "
                f"{self.current_weight*100:5.1f}% -> {self.target_weight*100:5.1f}%  {self.reason}")


@dataclass
class OrderPlan:
    orders: list[PlannedOrder] = field(default_factory=list)
    equity: float = 0.0
    blocked: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.blocked

    def __str__(self) -> str:
        out = [f"account equity: ${self.equity:,.2f}", ""]
        if self.orders:
            out.append(f"{len(self.orders)} order(s) planned:")
            out += [str(o) for o in self.orders]
            out.append(f"\n  total notional: ${sum(o.notional for o in self.orders):,.2f}")
        else:
            out.append("no orders -- nothing drifted past the band")
        if self.notes:
            out += ["", "notes:"] + [f"  - {n}" for n in self.notes]
        if self.blocked:
            out += ["", "BLOCKED:"] + [f"  !! {b}" for b in self.blocked]
        return "\n".join(out)


def connect(paper: bool = True):
    """Create an Alpaca trading client. Reads keys from the environment.

    Paper:  ALPACA_PAPER_KEY / ALPACA_PAPER_SECRET
    Live:   ALPACA_LIVE_KEY  / ALPACA_LIVE_SECRET

    Deliberately separate variable names so a paper script cannot accidentally
    pick up live credentials.
    """
    from alpaca.trading.client import TradingClient

    # Load algo/.env if present. That file is gitignored -- keep credentials
    # there, never in source, and never paste them into a chat or a commit.
    try:
        from dotenv import load_dotenv
        load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env"))
    except ImportError:
        pass

    prefix = "ALPACA_PAPER" if paper else "ALPACA_LIVE"
    key, secret = os.getenv(f"{prefix}_KEY"), os.getenv(f"{prefix}_SECRET")
    if not key or not secret:
        raise RuntimeError(
            f"missing {prefix}_KEY / {prefix}_SECRET in the environment. "
            "Paper and live keys are different -- generate paper keys from the "
            "Alpaca dashboard's paper account."
        )
    return TradingClient(key, secret, paper=paper)


def current_state(client) -> tuple[float, pd.Series]:
    """Return (equity, current weights by symbol)."""
    acct = client.get_account()
    equity = float(acct.equity)
    positions = client.get_all_positions()
    if not positions:
        return equity, pd.Series(dtype="float64")
    vals = {p.symbol: float(p.market_value) for p in positions}
    return equity, pd.Series(vals) / equity


def plan_orders(target_weights: pd.Series, equity: float,
                current_weights: pd.Series | None = None,
                band: float = 0.10,
                limits: RiskLimits | None = None) -> OrderPlan:
    """Diff target vs current weights and produce the orders needed.

    Pure function -- no network, no side effects. Testable without credentials.
    """
    limits = limits or RiskLimits()
    current_weights = (pd.Series(dtype="float64") if current_weights is None
                       else current_weights)
    plan = OrderPlan(equity=equity)

    symbols = sorted(set(target_weights.index) | set(current_weights.index))
    tgt = target_weights.reindex(symbols).fillna(0.0).astype(float)
    cur = current_weights.reindex(symbols).fillna(0.0).astype(float)

    # --- pre-flight checks on the TARGET, before any order is built ---------
    gross = float(tgt.sum())
    if gross > limits.max_gross_exposure + 1e-9:
        plan.blocked.append(
            f"target gross exposure {gross*100:.1f}% exceeds limit "
            f"{limits.max_gross_exposure*100:.0f}% -- refusing to lever")
    over = tgt[tgt > limits.max_position_pct]
    for sym, wgt in over.items():
        plan.blocked.append(
            f"{sym} target {wgt*100:.1f}% exceeds max_position_pct "
            f"{limits.max_position_pct*100:.0f}%")
    if equity <= 0:
        plan.blocked.append(f"account equity is ${equity:,.2f}")
    if plan.blocked:
        return plan

    # --- build orders for anything outside the band -------------------------
    for sym in symbols:
        drift = float(tgt[sym] - cur[sym])

        # Cold start: a position that does not exist yet is opened regardless of
        # the band. Without this, any target weight persistently below the band
        # (a 3% sleeve against a 10% band) would never be established at all --
        # the band is meant to suppress pointless REBALANCING, not to prevent
        # the portfolio from being built in the first place.
        opening = cur[sym] == 0.0 and tgt[sym] > 0.0
        if not opening and abs(drift) <= band:
            continue

        notional = abs(drift) * equity
        if notional < limits.min_trade_value:
            plan.notes.append(f"{sym}: ${notional:.2f} below min_trade_value, skipped")
            continue
        cap = limits.order_cap(equity)
        if notional > cap:
            plan.notes.append(f"{sym}: ${notional:,.2f} capped to ${cap:,.2f}")
            notional = cap
        plan.orders.append(PlannedOrder(
            symbol=sym,
            side="buy" if drift > 0 else "sell",
            notional=round(notional, 2),
            current_weight=float(cur[sym]),
            target_weight=float(tgt[sym]),
            reason=("opening position" if opening
                    else f"drift {drift*100:+.1f}% > band {band*100:.0f}%"),
        ))

    if len(plan.orders) > limits.max_orders_per_run:
        plan.blocked.append(
            f"{len(plan.orders)} orders exceeds max_orders_per_run "
            f"{limits.max_orders_per_run} -- looks like a runaway, refusing")
    return plan


def submit_orders(client, plan: OrderPlan, live: bool = False) -> list:
    """Submit a plan. Does NOTHING unless live=True is passed explicitly."""
    from alpaca.trading.enums import OrderSide, TimeInForce
    from alpaca.trading.requests import MarketOrderRequest

    if not plan.ok:
        raise RuntimeError(f"refusing to submit a blocked plan: {plan.blocked}")
    if not live:
        log.warning("DRY RUN -- %d order(s) NOT submitted. Pass live=True to send.",
                    len(plan.orders))
        return []

    submitted = []
    for o in plan.orders:
        req = MarketOrderRequest(
            symbol=o.symbol,
            notional=o.notional,
            side=OrderSide.BUY if o.side == "buy" else OrderSide.SELL,
            time_in_force=TimeInForce.DAY,
        )
        submitted.append(client.submit_order(req))
        log.info("submitted %s %s $%.2f", o.side, o.symbol, o.notional)
    return submitted
