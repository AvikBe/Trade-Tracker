-- Supports linking amendments to their original filing (store.link_amendments).
CREATE INDEX filings_owner_period_idx ON filings (filer_id, issuer_cik, period_of_report);
