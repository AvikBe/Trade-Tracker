-- Milestone 2: columns the feature job needs beyond the spec's trade_features.
ALTER TABLE trade_features
    ADD COLUMN feature_version       INT NOT NULL DEFAULT 1,
    ADD COLUMN filing_date           DATE,     -- Eastern date the trade became public
    ADD COLUMN trade_value           NUMERIC,  -- dollars: shares x price, or PTR range midpoint
    ADD COLUMN size_basis            TEXT,     -- 'filer' (own history) | 'market' (past year)
    ADD COLUMN duplicate_of_trade_id BIGINT REFERENCES trades ON DELETE SET NULL,
    ADD COLUMN flags                 TEXT[] NOT NULL DEFAULT '{}';

CREATE INDEX trade_features_latest_idx ON trade_features (trade_id, computed_at DESC);

-- The newest feature row per trade.
CREATE VIEW latest_trade_features AS
SELECT DISTINCT ON (trade_id) *
FROM trade_features
ORDER BY trade_id, computed_at DESC;
