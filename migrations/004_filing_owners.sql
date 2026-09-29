-- Every reporting owner on a filing, not just the primary one in filings.filer_id.
-- A joint 4/A can list a different set of owners than the filing it amends
-- (e.g. Mithaq Capital's Children's Place filings in Feb 2024), so amendments
-- are matched to originals on any shared owner.
CREATE TABLE filing_owners (
    filing_id  BIGINT NOT NULL REFERENCES filings(filing_id) ON DELETE CASCADE,
    owner_cik  TEXT   NOT NULL,
    PRIMARY KEY (filing_id, owner_cik)
);
CREATE INDEX filing_owners_cik_idx ON filing_owners (owner_cik);

-- Existing rows: the primary owner is the only one known until the data is reloaded.
INSERT INTO filing_owners (filing_id, owner_cik)
SELECT f.filing_id, fl.source_key FROM filings f JOIN filers fl USING (filer_id)
WHERE fl.source_key <> ''
ON CONFLICT DO NOTHING;
