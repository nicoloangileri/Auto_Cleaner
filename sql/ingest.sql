-- =============================================================================
-- auto_cleaner :: DuckDB ingestion recipes
-- -----------------------------------------------------------------------------
-- DuckDB powers the SQL ingestion path. These statements show how to pull any
-- common source into a table, then hand it to the Python/polars engine (which
-- reads the result zero-copy through Arrow) or export it straight to Parquet.
--
-- Run interactively:   duckdb mydb.duckdb < sql/ingest.sql
-- or from Python:      auto_cleaner.ingest.read_sql("mydb.duckdb", table="raw")
-- =============================================================================

-- Use all cores for ingestion.
PRAGMA threads=4;

-- 1) CSV with automatic schema, delimiter and type sniffing ------------------
--    `read_csv_auto` handles messy delimiters, quoting and nulls.
CREATE OR REPLACE TABLE raw_csv AS
SELECT * FROM read_csv_auto(
    'raw_data.csv',
    sample_size = -1,         -- scan the whole file for robust type inference
    ignore_errors = true,     -- skip unparseable rows instead of aborting
    nullstr = ['', 'NA', 'N/A', 'null', 'None', '?']
);

-- 2) Parquet (single file or a glob — partitioned datasets welcome) ----------
CREATE OR REPLACE TABLE raw_parquet AS
SELECT * FROM read_parquet('data/*.parquet', union_by_name = true);

-- 3) Newline-delimited JSON --------------------------------------------------
CREATE OR REPLACE TABLE raw_json AS
SELECT * FROM read_json_auto('events.ndjson');

-- 4) Hand off to the polars engine ------------------------------------------
--    The Python reader will SELECT * FROM this table and clean it.
CREATE OR REPLACE TABLE raw AS SELECT * FROM raw_csv;

-- 5) (Optional) export a cleaned table back to compressed Parquet ------------
-- COPY clean TO 'clean_data.parquet' (FORMAT PARQUET, COMPRESSION ZSTD);
