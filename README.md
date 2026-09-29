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
