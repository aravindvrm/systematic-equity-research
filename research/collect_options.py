#!/usr/bin/env python
"""Daily options snapshot. Run once per trading day, ideally ~15:45 ET.

    ./.venv/bin/python collect_options.py            # snapshot the universe
    ./.venv/bin/python collect_options.py --coverage # report what we have
"""
import argparse, logging, sys
import pandas as pd
from algo import data, optionsdb, universe

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
p = argparse.ArgumentParser()
p.add_argument("--coverage", action="store_true")
p.add_argument("--force", action="store_true",
               help="snapshot even if a good RTH capture already exists today")
p.add_argument("--expiries", type=int, default=12)
args = p.parse_args()

# The COLLECTION universe is deliberately wide. Options history cannot be
# backfilled -- a name not snapshotted today is gone for that day forever.
# Narrowing later costs nothing; widening retroactively is impossible.
UNIVERSE = universe.collection_universe()

if args.coverage:
    cov = optionsdb.coverage()
    if cov.empty:
        sys.exit("no data collected yet")
    print(cov.to_string())
    print(f"\n  {len(cov)} underlyings, "
          f"{cov['count'].min()}-{cov['count'].max()} days each")
    sys.exit(0)

session = optionsdb.market_session()
if optionsdb.have_good_snapshot_today() and not args.force:
    print(f"already have an RTH snapshot for today; nothing to do (use --force to override)")
    sys.exit(0)
print(f"session: {session}")
if session in ("WEEKEND",):
    # Exit 0, not 1. sys.exit(<string>) exits with status 1, which made the
    # launchd wrapper log every Saturday and Sunday as "FAIL" -- and therefore
    # made a real failure indistinguishable from a normal weekend.
    print("market closed; skipping")
    sys.exit(0)
if session != "RTH":
    print("  NOTE: outside regular hours -- quotes will be stale/wide. "
          "Volume and open interest are still valid.")
print(f"snapshotting {len(UNIVERSE)} underlyings, {args.expiries} expiries each ...")
spot = data.latest_prices(UNIVERSE)
chain = optionsdb.snapshot(UNIVERSE, max_expiries=args.expiries, spot=spot)
summ = optionsdb.summarize(chain)
optionsdb.append_summary(summ)

print(f"\n{len(chain):,} contracts stored for {chain['underlying'].nunique()} names")
print(f"\nDAILY SUMMARY (the signal inputs):\n")
print(summ[["underlying","atm_iv","iv_spread","skew","term_slope","pc_volume","pc_oi"]]
      .round(4).to_string(index=False))
