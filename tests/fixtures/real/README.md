Real SEC filings, used to test the parsers against what the SEC actually publishes.

- `2024q1_subset_form345.zip`: rows for eight accessions cut from the SEC's 2024q1
  Insider Transactions Data Set (SUBMISSION, REPORTINGOWNER, NONDERIV_TRANS, FOOTNOTES),
  unchanged apart from the row filter.
- `*.xml`: the matching ownership documents from EDGAR, as filed.

Built by filtering the full 2024q1 ZIP on ACCESSION_NUMBER and fetching each XML via `feed.ownership_xml_url`.
