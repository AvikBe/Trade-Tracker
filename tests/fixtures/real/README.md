Real SEC filings, used to test the parsers against what the SEC actually publishes.

- `2024q1_subset_form345.zip`: rows for eight accessions cut from the SEC's 2024q1
  Insider Transactions Data Set (SUBMISSION, REPORTINGOWNER, NONDERIV_TRANS, FOOTNOTES),
  unchanged apart from the row filter.
- `*.xml`: the matching ownership documents from EDGAR, as filed.

Built by filtering the full 2024q1 ZIP on ACCESSION_NUMBER and fetching each XML via `feed.ownership_xml_url`.

Feature tests (`tests/test_features_real.py`):

- `2024q1_features_form345.zip`: 39 accessions from the same 2024q1 data set, chosen for
  real feature cases: the CTBI board buying together (cluster of 11), PLCE's February 2024
  run-up (132.9% drift before filing) and its 4/As, NBIX 10b5-1 sales, and RMCF joint
  filers who filed the same buys twice plus two 4/As that correct one line.
- `2024q1_features_prices.csv`: Tiingo adjusted closes for those four tickers over the
  same weeks, cross-checked against Yahoo Finance.
