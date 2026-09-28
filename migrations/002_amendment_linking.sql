-- Linking amendments to their original filing (store.link_amendments).
ALTER TABLE filings ADD COLUMN original_filed_on DATE;  -- from the 4/A itself
CREATE INDEX filings_owner_period_idx ON filings (filer_id, issuer_cik, period_of_report);
