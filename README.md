# Trade Tracker

Ingests corporate-insider (SEC Form 4) trades, and later congressional PTRs, and ranks
the ones still worth acting on, weighted by how promptly each was disclosed. Every
signal is measured from the filing date, never the trade date.

This is milestone 1, the data foundation, on a free stack:

| Data | Source | Cost |
| --- | --- | --- |
| Form 4 history, 2015 onward | SEC quarterly Insider Transactions Data Sets | free |
| Form 4 live | EDGAR latest-filings feed + filing XML | free |
| CIK to ticker | ticker printed on each filing, then SEC `company_tickers.json` | free |
| Daily prices incl. delisted | Tiingo free tier (50 req/hour, 1,000/day) | free |
| Congress trades | deferred until the insider backtest shows the signal holds | n/a |

## Setup

```sh
cp .env.example .env        # fill in EDGAR_USER_AGENT and TIINGO_API_KEY; never commit .env
make install
make db                     # local Postgres 16 in Docker
set -a; . ./.env; set +a
tt migrate
```

## Commands

```sh
tt load-bulk --quarters 2015q1:2026q2   # download and load SEC insider data sets
tt poll-edgar                           # new Form 4s from the live feed (run every 10 min)
tt map-tickers                          # fill tickers missing on filings from today's SEC map
tt load-prices                          # price backfill/update, stays inside Tiingo's quota
tt validate                             # spec validation rules -> rejects table
tt report                               # exit check: coverage per year, lag histogram
tt features [--all]                     # milestone 2: fill trade_features (see below)
tt feature-report                       # feature coverage and distributions
```

`load-prices` spends at most the remaining hourly and daily Tiingo quota per run and
queues backfills before updates, so run it hourly until the backlog clears.

## What gets stored

- Only Form 4 open-market purchases (`P`) and sales (`S`). Grants, exercises, gifts and
  tax withholding are dropped at parse time.
- 10b5-1 plan trades are stored with `is_10b5_1 = true` (checkbox, footnote or remarks)
  so the backtest can compare them; ranking will exclude them, as the spec says.
- Amendments (4/A) are stored as their own filings with `amends_filing_id` pointing at
  the original; nothing is overwritten.
- `filed_at` is the filing date (midnight Eastern) for bulk history, and `accepted_at`
  the exact EDGAR acceptance time when the live feed supplies it.

## Features (milestone 2)

`tt features` writes one `trade_features` row per trade (`latest_trade_features` is the
newest per trade). Everything is point in time: history features only see trades whose
filing date is before (z-score, size) or on (clusters) this trade's filing date.

| Column | Definition |
| --- | --- |
| `lag_days` | SEC business days from trade date to filing date (federal holidays skipped, Good Friday counted) |
| `lag_ratio` | `lag_days` / deadline: 2 for Form 4, 31 for PTRs |
| `drift_pct` | % change from the trade-date close to the filing-date close, sign flipped for sells; the last close within 5 days stands in for non-trading days |
| `days_since_filing` | NYSE trading days from filing date to `--as-of` |
| `filer_lag_zscore` | lag vs the filer's earlier filings (one lag per filing, 3+ needed, std floored at 0.5) |
| `role_weight` | CEO/CFO 1.0, officer 0.75, director 0.5, 10% owner 0.25 |
| `size_score` | dollar size percentile vs the filer's own earlier same-side trades (5+), else vs all same-side trades of the past year (`size_basis`) |
| `cluster_count` | distinct insiders trading the same ticker and side within 14 days, known by this filing date |

Amendment lines that repeat or correct an original line keep the original filing date,
point at it through `duplicate_of_trade_id`, and never count twice in any history.
`flags` records why a value is missing (`no_prices`, `no_trade_price`, `trade_after_filing`, ...)
and marks `10b5_1` and amendment trades.

## Tests

```sh
pytest                                          # parser tests
TT_TEST_DATABASE_URL=postgresql://... pytest    # plus database tests (drops the schema!)
```

Fixtures are synthetic files in SEC's formats, not real filings.
