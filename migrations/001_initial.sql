-- Milestone 1 schema: the spec's four core tables plus prices, ticker history,
-- rejects and an API call log. Every filing keeps filed_at and first_seen_at so a
-- backtest can replay only what was knowable at the time.

CREATE TABLE filers (
    filer_id     BIGSERIAL PRIMARY KEY,
    source       TEXT NOT NULL,              -- 'edgar' | 'house' | 'senate' | vendor
    source_key   TEXT NOT NULL,              -- reporting-owner CIK for EDGAR
    name         TEXT NOT NULL,
    kind         TEXT NOT NULL CHECK (kind IN ('insider', 'house', 'senate')),
    party        TEXT,
    state        TEXT,
    committees   TEXT[],
    UNIQUE (source, source_key)
);

CREATE TABLE filings (
    filing_id         BIGSERIAL PRIMARY KEY,
    filer_id          BIGINT NOT NULL REFERENCES filers,
    source            TEXT NOT NULL,
    source_filing_id  TEXT NOT NULL,         -- EDGAR accession number
    source_url        TEXT,
    document_type     TEXT,                  -- '4' | '4/A' | 'PTR' ...
    issuer_cik        TEXT,
    issuer_name       TEXT,
    issuer_ticker     TEXT,                  -- symbol as reported on the filing (point in time)
    period_of_report  DATE,
    filed_at          TIMESTAMPTZ NOT NULL,  -- official filing date (midnight ET when only a date is known)
    accepted_at       TIMESTAMPTZ,           -- exact EDGAR acceptance time; NULL when unknown
    first_seen_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    amends_filing_id  BIGINT REFERENCES filings,
    raw               JSONB,
    UNIQUE (source, source_filing_id)
);
CREATE INDEX filings_filed_at_idx ON filings (filed_at);
CREATE INDEX filings_issuer_idx ON filings (issuer_cik);

CREATE TABLE trades (
    trade_id     BIGSERIAL PRIMARY KEY,
    filing_id    BIGINT NOT NULL REFERENCES filings ON DELETE CASCADE,
    line_no      INT NOT NULL,              -- position within the filing, for idempotent re-ingest
    ticker       TEXT,
    asset_type   TEXT NOT NULL DEFAULT 'stock',
    side         TEXT NOT NULL CHECK (side IN ('buy', 'sell')),
    txn_code     TEXT,
    is_10b5_1    BOOLEAN NOT NULL DEFAULT false,
    trade_date   DATE NOT NULL,
    amount_low   NUMERIC,
    amount_high  NUMERIC,
    shares       NUMERIC,
    price        NUMERIC,
    owner        TEXT,                      -- self | spouse | child | trust | indirect
    role         TEXT,                      -- CEO | CFO | officer | director | 10% owner | member
    UNIQUE (filing_id, line_no)
);
CREATE INDEX trades_ticker_date_idx ON trades (ticker, trade_date);

CREATE TABLE trade_features (
    trade_id          BIGINT NOT NULL REFERENCES trades ON DELETE CASCADE,
    computed_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
    lag_days          INT,
    lag_ratio         NUMERIC,
    drift_pct         NUMERIC,
    days_since_filing INT,
    filer_lag_zscore  NUMERIC,
    cluster_count     INT,
    role_weight       NUMERIC,
    size_score        NUMERIC,
    committee_match   BOOLEAN,
    score             NUMERIC,
    score_version     INT,
    PRIMARY KEY (trade_id, computed_at)
);

CREATE TABLE daily_prices (
    ticker     TEXT NOT NULL,
    date       DATE NOT NULL,
    open       NUMERIC,
    high       NUMERIC,
    low        NUMERIC,
    close      NUMERIC,
    adj_open   NUMERIC,
    adj_close  NUMERIC,
    volume     BIGINT,
    source     TEXT NOT NULL DEFAULT 'tiingo',
    PRIMARY KEY (ticker, date)
);

-- CIK -> ticker as observed over time, so the backtest never uses a future mapping.
CREATE TABLE ticker_history (
    cik         TEXT NOT NULL,
    ticker      TEXT NOT NULL,
    first_seen  DATE NOT NULL,
    last_seen   DATE NOT NULL,
    source      TEXT NOT NULL,              -- 'filing' | 'company_tickers'
    PRIMARY KEY (cik, ticker, source)
);

CREATE TABLE rejects (
    reject_id    BIGSERIAL PRIMARY KEY,
    trade_id     BIGINT REFERENCES trades ON DELETE CASCADE,
    rule         TEXT NOT NULL,
    detail       TEXT,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (trade_id, rule)
);

-- Tracks vendor calls so loaders stay inside free-tier quotas (Tiingo: 50/hour, 1000/day).
CREATE TABLE api_calls (
    vendor     TEXT NOT NULL,
    called_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    endpoint   TEXT
);
CREATE INDEX api_calls_vendor_time_idx ON api_calls (vendor, called_at);

-- Per-ticker price load state, so tickers the vendor doesn't know stop costing quota.
CREATE TABLE price_status (
    ticker        TEXT PRIMARY KEY,
    status        TEXT NOT NULL,            -- 'ok' | 'not_found' | 'error'
    last_attempt  TIMESTAMPTZ NOT NULL DEFAULT now(),
    detail        TEXT
);

CREATE TABLE schema_migrations (
    version     TEXT PRIMARY KEY,
    applied_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);
