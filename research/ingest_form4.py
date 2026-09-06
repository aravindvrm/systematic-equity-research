"""Ingest SEC Form 4 history for the full S&P 500."""
import logging
from algo import form4, universe as U

logging.basicConfig(level=logging.INFO, format="%(message)s")
cons = U.fetch_constituents()
tickers = set(cons.ticker.str.upper()) | set(U.DIVERSIFIERS)
print(f"ingesting Form 4 for {len(tickers)} tickers, 2018q1..2026q1")
df = form4.ingest(tickers=tickers, start_year=2018, end="2026q1")
print(f"\nDONE: {len(df):,} transactions -> {form4.TRANS_PATH}")
