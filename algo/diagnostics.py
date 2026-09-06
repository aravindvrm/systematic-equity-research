"""Automatic sanity checks on a backtest result.

Every backtest gets these. The point is not to produce more numbers -- it is to
make the failure modes *loud*, because they are otherwise invisible: a bad
strategy and a good one produce equally attractive equity curves.

The checks encode the things that turned an apparent 0.98 Sharpe crypto
strategy into "worse than just holding BTC":

    1. Does it beat the dumb benchmark, on return AND on drawdown?
    2. Does the edge survive splitting the sample in half?
    3. Are the tails fat enough that Sharpe is the wrong metric?
    4. Does it survive a haircut for the number of variants you tried?
    5. Is the whole thing an artifact of costs being ignored?
    6. MARGINAL: added to the portfolio you already own, does it help at all --
       and does it beat simply holding less equity?

CHECK 6 EXISTS BECAUSE CHECKS 1-5 ASKED THE WRONG QUESTION (2026-09-05 audit).
Standalone Sharpe against an equity benchmark is close to irrelevant for a sleeve
held ALONGSIDE an existing portfolio. What matters is correlation and marginal
contribution. Worse, a low-beta sleeve raises portfolio Sharpe simply by
de-risking -- which you can do for free by holding less equity -- so any gain
must be measured against a BETA-MATCHED core/cash mix, not against the core.
Pass `core=` to enable it. See notes/Methodology Audit.md.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import metrics


@dataclass
class Warning_:
    level: str      # "FAIL" | "WARN" | "NOTE"
    check: str
    message: str

    def __str__(self) -> str:
        icon = {"FAIL": "[FAIL]", "WARN": "[WARN]", "NOTE": "[note]"}[self.level]
        return f"  {icon} {self.check}: {self.message}"


@dataclass
class Diagnostics:
    stats: dict
    warnings: list[Warning_] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not any(w.level == "FAIL" for w in self.warnings)

    def __str__(self) -> str:
        lines = []
        s = self.stats
        lines.append(f"  {'Sharpe (net)':<26} {s['sharpe']:>8.2f}")
        lines.append(f"  {'Sharpe (gross)':<26} {s['gross_sharpe']:>8.2f}"
                     f"   cost drag {s['sharpe_cost_drag']:.2f}")
        lines.append(f"  {'deflated (n_trials='+str(s['n_trials'])+')':<26} {s['deflated_sharpe']:>8.2f}")
        lines.append(f"  {'CAGR':<26} {s['cagr']*100:>7.2f}%")
        lines.append(f"  {'volatility':<26} {s['vol']*100:>7.2f}%")
        lines.append(f"  {'max drawdown':<26} {s['max_drawdown']*100:>7.2f}%")
        lines.append(f"  {'Calmar':<26} {s['calmar']:>8.2f}")
        lines.append(f"  {'avg turnover':<26} {s['avg_turnover']:>8.3f}")
        lines.append("")
        lines.append(f"  {'1st half Sharpe':<26} {s['first_half']:>8.2f}")
        lines.append(f"  {'2nd half Sharpe':<26} {s['second_half']:>8.2f}"
                     f"   retention {s['retention']*100:.0f}%")
        lines.append(f"  {'skew':<26} {s['skew']:>8.2f}")
        lines.append(f"  {'excess kurtosis':<26} {s['excess_kurtosis']:>8.2f}")
        lines.append(f"  {'worst / best day':<26} {s['worst']*100:>7.2f}% / {s['best']*100:.2f}%")
        if "marg_corr_to_core" in s:
            lines.append("")
            lines.append("  MARGINAL vs your existing portfolio")
            lines.append(f"  {'correlation to core':<26} {s['marg_corr_to_core']:>8.2f}")
            lines.append(f"  {'beta to core':<26} {s['marg_beta']:>8.2f}")
            lines.append(f"  {'alpha (annualised)':<26} {s['marg_alpha_ann']*100:>7.2f}%"
                         f"   NW t {s['marg_alpha_t']:.2f}")
            lines.append(f"  {'Sharpe @ 10% weight':<26} {s['marg_sharpe_at_10pct']:>8.2f}"
                         f"   delta {s['marg_delta_at_10pct']:+.3f}")
            if "marg_genuine_gain" in s:
                lines.append(f"  {'gain over beta-matched':<26} "
                             f"{s['marg_genuine_gain']:>+8.3f}"
                             "   <- the number that matters")
        if "benchmark_sharpe" in s:
            lines.append("")
            lines.append(f"  {'benchmark ('+s['benchmark_name']+')':<26} "
                         f"Sharpe {s['benchmark_sharpe']:.2f}  "
                         f"CAGR {s['benchmark_cagr']*100:.2f}%  "
                         f"MaxDD {s['benchmark_maxdd']*100:.1f}%")
            lines.append(f"  {'excess vs benchmark':<26} "
                         f"Sharpe {s['excess_sharpe']:+.2f}  CAGR {s['excess_cagr']*100:+.2f}%  "
                         f"Calmar {s['excess_calmar']:+.2f}")
        if self.warnings:
            lines.append("")
            lines.extend(str(w) for w in self.warnings)
        else:
            lines.append("\n  all checks passed")
        return "\n".join(lines)


def marginal_checks(returns, core, stats: dict, w: list, cash_annual: float = 0.02):
    """Check 6: marginal contribution to an existing portfolio.

    `core` is the return series of the portfolio the user already holds. The
    de-risking control is the important half: a sleeve with beta b is compared
    against b*core + (1-b)*cash, which requires no strategy at all.
    """
    from . import evaluation

    ev = evaluation.evaluate(returns, core, label="strategy")
    stats.update({f"marg_{k}": v for k, v in ev.items() if k != "label"})

    s, c = evaluation._align(returns, core)
    beta = ev["beta"]
    if not np.isfinite(beta):
        return
    cash = pd.Series(cash_annual / 252, index=c.index)
    matched = beta * c + (1 - beta) * cash
    gain = (metrics.sharpe(0.9 * c + 0.1 * s)
            - metrics.sharpe(0.9 * c + 0.1 * matched))
    stats["marg_genuine_gain"] = gain

    if ev["corr_to_core"] > 0.7:
        w.append(Warning_("NOTE", "marginal",
                          f"correlation to core is {ev['corr_to_core']:.2f} -- this is "
                          "not a diversifier, it is more of what you already own."))
    # The beta-matched control is NECESSARY BUT NOT SUFFICIENT. Random signals
    # through a vol-targeted stack score +0.029 on it (measured, n=40), because
    # vol targeting gives the sleeve a time-varying beta that a static match
    # cannot represent. Treat anything below ~+0.035 as indistinguishable from
    # noise unless an explicit null floor says otherwise.
    NULL_GAIN_MEAN, NULL_GAIN_P95 = 0.029, 0.035
    NULL_ALPHA_T_MEAN = 3.14
    if gain < NULL_GAIN_P95:
        w.append(Warning_("FAIL", "null floor",
                          f"genuine gain {gain:+.4f} is inside the measured noise "
                          f"floor for this stack (random signals: mean "
                          f"{NULL_GAIN_MEAN:+.3f}, p95 {NULL_GAIN_P95:+.3f}). "
                          "Run algo.evaluation.null_floor with YOUR stack to "
                          "calibrate properly -- these constants are from a "
                          "top-30% inverse-vol 10%-vol-target book."))
    if np.isfinite(ev["alpha_t"]) and ev["alpha_t"] < NULL_ALPHA_T_MEAN + 1.0:
        w.append(Warning_("FAIL", "alpha vs null",
                          f"alpha t={ev['alpha_t']:.2f}. Random signals through a "
                          f"vol-targeted stack average t={NULL_ALPHA_T_MEAN:.2f}. "
                          "A t of 3 is NOT a bar here -- noise clears it."))
    if gain <= 0:
        w.append(Warning_("FAIL", "de-risking control",
                          f"at 10% weight this adds {gain:+.3f} Sharpe over simply "
                          f"holding beta={beta:.2f} of the core and the rest in cash. "
                          "The sleeve is doing nothing a lower equity weight would "
                          "not do for free."))
    elif gain < 0.02:
        w.append(Warning_("WARN", "de-risking control",
                          f"genuine gain over a beta-matched core/cash mix is only "
                          f"{gain:+.3f} Sharpe at 10% weight."))



def analyze(result, benchmark=None, benchmark_name: str = "benchmark",
            n_trials: int = 1, core=None) -> Diagnostics:
    """Run every sanity check against a BacktestResult.

    benchmark: another BacktestResult to compare against -- normally the dumbest
               thing that would have worked (buy and hold).
    n_trials:  how many strategy variants you tried before settling on this one.
               Be honest. Every parameter you swept counts.
    """
    ppy = result.periods_per_year
    r = result.returns
    stats = dict(result.summary())
    stats.update(metrics.tail_stats(r))
    stats.update(metrics.split_stability(r, ppy))
    stats["n_trials"] = n_trials
    stats["deflated_sharpe"] = metrics.deflated_sharpe(stats["sharpe"], n_trials, len(r))
    stats["sharpe_cost_drag"] = stats["gross_sharpe"] - stats["sharpe"]

    w: list[Warning_] = []

    # 1. benchmark comparison -- the check that matters most
    if benchmark is not None:
        b = benchmark.summary()
        stats.update(benchmark_name=benchmark_name,
                     benchmark_sharpe=b["sharpe"], benchmark_cagr=b["cagr"],
                     benchmark_maxdd=b["max_drawdown"],
                     benchmark_calmar=b["calmar"],
                     excess_sharpe=stats["sharpe"] - b["sharpe"],
                     excess_cagr=stats["cagr"] - b["cagr"],
                     excess_calmar=stats["calmar"] - b["calmar"])
        if stats["excess_sharpe"] <= 0:
            # Losing on Sharpe while winning on Calmar is a real and common
            # outcome for a de-risked portfolio: same reward per unit of
            # volatility, much better reward per unit of drawdown. Say so
            # rather than reporting a bare failure.
            if stats["excess_calmar"] > 0:
                w.append(Warning_("WARN", "benchmark",
                                  f"loses to {benchmark_name} on Sharpe "
                                  f"({stats['sharpe']:.2f} vs {b['sharpe']:.2f}) but WINS on "
                                  f"Calmar ({stats['calmar']:.2f} vs {b['calmar']:.2f}). "
                                  "It is a different risk profile, not a better Sharpe. "
                                  "Judge it on which risk you actually care about."))
            else:
                w.append(Warning_("FAIL", "benchmark",
                                  f"does not beat {benchmark_name} on Sharpe "
                                  f"({stats['sharpe']:.2f} vs {b['sharpe']:.2f}) or Calmar "
                                  f"({stats['calmar']:.2f} vs {b['calmar']:.2f}). "
                                  "The simpler thing is better."))
        elif stats["excess_sharpe"] < 0.1:
            w.append(Warning_("WARN", "benchmark",
                              f"only {stats['excess_sharpe']:+.2f} Sharpe over {benchmark_name} "
                              "-- inside the noise for this sample length."))
        if stats["max_drawdown"] < b["max_drawdown"]:
            w.append(Warning_("WARN", "benchmark risk",
                              f"drawdown is WORSE than {benchmark_name} "
                              f"({stats['max_drawdown']*100:.1f}% vs {b['max_drawdown']*100:.1f}%)."))

    # 2. regime dependence
    if stats["first_half"] > 0 and stats["retention"] < 0.5:
        w.append(Warning_("FAIL", "stability",
                          f"2nd-half Sharpe is only {stats['retention']*100:.0f}% of 1st-half "
                          f"({stats['second_half']:.2f} vs {stats['first_half']:.2f}). "
                          "This describes a regime, not a rule."))
    elif stats["first_half"] > 0 and stats["retention"] < 0.7:
        w.append(Warning_("WARN", "stability",
                          f"edge decayed to {stats['retention']*100:.0f}% in the 2nd half."))

    # 3. fat tails make Sharpe misleading
    if stats["excess_kurtosis"] > 8:
        w.append(Warning_("WARN", "tails",
                          f"excess kurtosis {stats['excess_kurtosis']:.1f}, worst day "
                          f"{stats['worst']*100:.1f}%. Sharpe understates tail risk here; "
                          "look at max drawdown and Calmar instead."))

    # 4. multiple testing
    if n_trials > 1 and stats["deflated_sharpe"] < 0.2:
        w.append(Warning_("FAIL", "multiple testing",
                          f"deflated Sharpe {stats['deflated_sharpe']:.2f} after {n_trials} "
                          "trials. Consistent with having found nothing."))
    if n_trials == 1:
        w.append(Warning_("NOTE", "multiple testing",
                          "n_trials=1 assumed. If you swept parameters to get here, "
                          "pass the real count -- the haircut is not optional."))

    # 5. cost sensitivity
    if stats["gross_sharpe"] > 0 and stats["sharpe_cost_drag"] / stats["gross_sharpe"] > 0.3:
        w.append(Warning_("WARN", "costs",
                          f"costs eat {stats['sharpe_cost_drag']/stats['gross_sharpe']*100:.0f}% "
                          "of gross Sharpe. Very sensitive to fill quality and order size."))

    # 6. marginal contribution -- the check that should have existed first
    if core is not None:
        marginal_checks(r, core, stats, w)

    # 7. sample adequacy
    years = stats["n_periods"] / ppy
    if years < 5:
        w.append(Warning_("WARN", "sample",
                          f"only {years:.1f} years of data. Too short to distinguish "
                          "skill from luck."))
    return Diagnostics(stats=stats, warnings=w)
