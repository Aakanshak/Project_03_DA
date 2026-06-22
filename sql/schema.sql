-- Dynamic Pricing and Promotion Optimization Platform
-- Normalized PostgreSQL schema for the Rossmann Store Sales dataset.

BEGIN;

CREATE TABLE IF NOT EXISTS stores (
    store_id INTEGER PRIMARY KEY CHECK (store_id > 0)
);

CREATE TABLE IF NOT EXISTS store_metadata (
    store_id INTEGER PRIMARY KEY
        REFERENCES stores (store_id)
        ON UPDATE CASCADE
        ON DELETE CASCADE,
    store_type VARCHAR(1) NOT NULL,
    assortment VARCHAR(1) NOT NULL,
    competition_distance NUMERIC(12, 2),
    competition_open_since_month SMALLINT
        CHECK (competition_open_since_month BETWEEN 1 AND 12),
    competition_open_since_year SMALLINT
        CHECK (competition_open_since_year >= 1900),
    promo2 BOOLEAN NOT NULL DEFAULT FALSE,
    promo2_since_week SMALLINT
        CHECK (promo2_since_week BETWEEN 1 AND 53),
    promo2_since_year SMALLINT
        CHECK (promo2_since_year >= 1900),
    promo_interval VARCHAR(64),
    CONSTRAINT promo2_dates_consistent CHECK (
        promo2
        OR (
            promo2_since_week IS NULL
            AND promo2_since_year IS NULL
            AND promo_interval IS NULL
        )
    )
);

CREATE TABLE IF NOT EXISTS daily_sales (
    store_id INTEGER NOT NULL
        REFERENCES stores (store_id)
        ON UPDATE CASCADE
        ON DELETE RESTRICT,
    sales_date DATE NOT NULL,
    sales NUMERIC(14, 2) NOT NULL CHECK (sales >= 0),
    customers INTEGER NOT NULL CHECK (customers >= 0),
    open BOOLEAN NOT NULL,
    promo BOOLEAN NOT NULL,
    state_holiday VARCHAR(1) NOT NULL DEFAULT '0',
    school_holiday BOOLEAN NOT NULL,
    PRIMARY KEY (store_id, sales_date)
);

CREATE INDEX IF NOT EXISTS idx_daily_sales_date
    ON daily_sales (sales_date);

CREATE INDEX IF NOT EXISTS idx_daily_sales_promo_date
    ON daily_sales (promo, sales_date);

CREATE INDEX IF NOT EXISTS idx_store_metadata_type_assortment
    ON store_metadata (store_type, assortment);

CREATE OR REPLACE VIEW vw_store_profile AS
SELECT
    s.store_id,
    m.store_type,
    m.assortment,
    m.competition_distance,
    m.competition_open_since_month,
    m.competition_open_since_year,
    m.promo2,
    m.promo2_since_week,
    m.promo2_since_year,
    m.promo_interval
FROM stores AS s
JOIN store_metadata AS m USING (store_id);

COMMIT;

