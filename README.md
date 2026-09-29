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
| Daily prices incl. delisted | Tiingo free tier (50 req/hour, 1,000/day, 500 symbols/month) | free |
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

## Price cache (hourly backfill without a shared database)

Tiingo's free plan allows only **500 unique symbols a month**, besides 50 requests an
hour and 1,000 a day, so a quarter's ~2,700 tickers take months, not days. The backfill
therefore runs as an hourly job that keeps its state in a directory (`--cache` or
`$TT_PRICE_CACHE`) rather than in Postgres, so any session can resume it:

```sh
tt price-universe --cache DIR     # from the trades table: quarter,ticker,buys,trades
tt fetch-prices --cache DIR       # hourly: fetch what the quotas allow, write coverage.txt
tt price-coverage --cache DIR     # tickers and trades covered per quarter
tt import-prices --cache DIR      # copy the cache into daily_prices (idempotent)
```

- Each ticker is fetched once, as CSV from 2014-01-01 (about 280 KB), and serves every
  quarter it appears in.
- Order: SPY and the sector ETFs, then quarters by `--order` (default 2024q1 back to
  2023q1, then 2026q2, 2022q4, 2020q1, 2018q3, 2015q1), and within a quarter by buy
  count, so the monthly symbol quota covers as many buys as possible.
- Budgets keep headroom: 45 calls an hour, 950 a day and 480 new symbols per calendar
  month (UTC), counted from `calls.csv` (a paid plan raises them with `TT_TIINGO_HOURLY`,
  `TT_TIINGO_DAILY` and `TT_TIINGO_SYMBOLS`). A quota refusal from Tiingo stops the run and
  sets a cooldown (55 minutes, 6 hours, or 24 hours for the symbol quota).
- Unknown tickers are marked `not_found`, tickers with no bars since 2014 `empty`, and
  network or server errors are retried an hour apart, three attempts in all.
- Coverage counts a ticker for a quarter only when its bars span that quarter, which
  catches tickers delisted earlier or reused by a later company.
- `fetch-prices` and `load-prices` share one API key but not one call log, so don't run
  `load-prices` while the hourly job is active.

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
pytest                                          # parser, client and CLI tests
TT_TEST_DATABASE_URL=postgresql://... pytest    # plus database tests (drops the schema!)
TT_LIVE=1 pytest -m live                        # smoke tests against live SEC and Tiingo
```

CI runs everything except the live tests, against Postgres 16.

`tests/fixtures/real/` holds real SEC filings: an eight-filing cut of the 2024q1 data set
and the matching XML documents. The bulk and XML parsers are tested to agree on them.
The other fixtures are synthetic.

## Known data quirks

Found by loading 2015q1, 2018q3, 2020q1, 2022q4, 2023q4, 2024q1 and 2026q2 and
cross-checking 260 filings against their XML:

- The data sets round shares and prices to two decimals (half up). The XML keeps
  full precision, so live-feed rows are more exact than bulk rows.
- Within a data set, an amendment can be listed before its original. `load-bulk` and
  `poll-edgar` finish with a relink pass that matches on owner, issuer and period,
  or on the original's filing date that each 4/A states. With 2023q4 and 2024q1
  loaded, 1,109 of 1,122 2024q1 amendments are linked or amend a pre-October 2023
  filing. Amendments of filings older than the loaded history stay unlinked, and
  `tt report` shows how many.
- Joint filings list their owners in a different order in the two sources, so both
  parsers pick the most senior owner, then the lowest CIK.
- Row order in the data sets is not document order, and the ownership nature
  (spouse, trust) is sometimes on the wrong row. Side, date, shares and price agree.
- Filers type tickers like `NONE`, `(SIRI)`, `NYSE: SCS` or `Z AND ZG`; they are
  normalized, and placeholders are left for `tt map-tickers`.
