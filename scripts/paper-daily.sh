#!/usr/bin/env bash
# Daily paper-trading run, before the US open (the routine runs it at 12:30 UTC on
# weekdays). Always runs the dry-run book; also runs the Alpaca paper book when
# ALPACA_API_KEY and ALPACA_SECRET_KEY are set. Safe to rerun: each book has a lock and
# client order ids stop duplicate orders. Output is appended to $ROOT/daily.log.
set -uo pipefail
ROOT=${TT_PAPER_ROOT:-/mnt/project-files/paper}
REPO=${TT_REPO_DIR:-/home/user/trade-tracker}
BRANCH=${TT_PAPER_BRANCH:-m5-paper-trading}
MODEL=${TT_PAPER_MODEL:-$REPO/models/two-inputs-60-days-cap-floor.json}
SHARES=${TT_SHARES_SEED:-/mnt/project-files/fundamentals/shares-outstanding.csv}
status=0
{
  echo "=== $(date -u +%FT%TZ)"
  if ! git -C "$REPO" rev-parse HEAD >/dev/null 2>&1; then
    git clone -q --depth 1 -b "$BRANCH" https://github.com/AvikBe/Trade-Tracker "$REPO" \
      || git clone -q --depth 1 https://github.com/AvikBe/Trade-Tracker "$REPO"
  fi
  git -C "$REPO" pull -q --ff-only 2>/dev/null || true
  python3 -c "import tradetracker.paper.engine" 2>/dev/null \
    || python3 -m pip install -q -e "$REPO" >/dev/null 2>&1
  books="dry-run"
  if [ -n "${ALPACA_API_KEY:-}" ] && [ -n "${ALPACA_SECRET_KEY:-}" ]; then
    books="dry-run alpaca"
  fi
  for b in $books; do
    echo "--- $b"
    out=$(tt paper-trade --state "$ROOT/$b" --broker "$b" --model "$MODEL" --shares "$SHARES" \
      --filings "$ROOT/filings" 2>&1)
    rc=$?
    printf '%s\n' "$out" | grep -v -e "HTTP Request" -e "dry run: would"
    [ $rc -eq 0 ] || { echo "!!! $b book failed (exit $rc)"; status=1; }
  done
} >> "$ROOT/daily.log" 2>&1
tail -n 40 "$ROOT/daily.log"
exit $status
