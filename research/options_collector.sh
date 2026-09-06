#!/bin/bash
# Launcher for the daily options snapshot. Invoked by launchd.
#
# This file lives on the INTERNAL drive deliberately. The project sits on an
# exFAT USB volume mounted `noowners,nosuid`, and launchd refuses to execute a
# program whose ownership it cannot verify there (EX_CONFIG / exit 78). Keeping
# the launcher on APFS also means it still runs -- and still LOGS -- when the
# USB is unplugged, which would otherwise be a silent no-op.

set -uo pipefail

PROJECT="/Volumes/SD3.2_256/Repos/Trading/algo"
VENV="$PROJECT/.venv/bin/python"
LOG="$HOME/Library/Logs/options_collector.log"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S %Z')] $*" >> "$LOG"; }

if [ ! -d "$PROJECT" ]; then
    log "SKIP: USB volume not mounted"
    exit 0
fi
if [ ! -x "$VENV" ]; then
    log "FAIL: venv missing at $VENV"
    exit 1
fi

# Market-holiday guard: cron/launchd know weekends, not holidays.
FRESH=$("$VENV" - <<'PY' 2>/dev/null
import warnings, yfinance as yf, pandas as pd
warnings.filterwarnings("ignore")
try:
    d = yf.download("SPY", period="5d", progress=False, auto_adjust=True)
    last = pd.to_datetime(d.dropna().index[-1]).date()
    print("STALE" if (pd.Timestamp.today().date() - last).days > 1 else "FRESH")
except Exception:
    print("UNKNOWN")
PY
)
if [ "$FRESH" = "STALE" ]; then
    log "SKIP: market data stale -- likely a holiday"
    exit 0
fi

log "START"
cd "$PROJECT" || { log "FAIL: cd"; exit 1; }
if "$VENV" collect_options.py >> "$LOG" 2>&1; then
    N=$(find "$PROJECT/data/options/raw" -name "20*.parquet" 2>/dev/null | wc -l | tr -d ' ')
    log "OK ($N daily files on disk)"
else
    log "FAIL: collect_options.py exit $?"
    exit 1
fi

if [ "$(wc -c < "$LOG")" -gt 5000000 ]; then
    tail -n 2000 "$LOG" > "$LOG.tmp" && mv "$LOG.tmp" "$LOG"
fi
