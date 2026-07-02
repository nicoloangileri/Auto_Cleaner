-- =============================================================================
-- auto_cleaner :: DuckDB in-database profiling
-- -----------------------------------------------------------------------------
-- Push the heavy EDA aggregation down into DuckDB so profiling scales to data
-- far larger than RAM. Replace `raw` with your table name (see ingest.sql).
--
-- Run:  duckdb mydb.duckdb < sql/profiling.sql
-- =============================================================================

PRAGMA threads=4;

-- 0) The fastest possible overview -------------------------------------------
--    DuckDB's built-in SUMMARIZE returns per-column count, nulls, distinct,
--    min, max, avg, std and approximate quartiles in a single pass.
SUMMARIZE SELECT * FROM raw;

-- 1) Dataset shape & duplicate detection -------------------------------------
SELECT
    (SELECT COUNT(*)               FROM raw)               AS n_rows,
    (SELECT COUNT(*) FROM (SELECT DISTINCT * FROM raw))    AS n_unique_rows,
    (SELECT COUNT(*) FROM raw)
        - (SELECT COUNT(*) FROM (SELECT DISTINCT * FROM raw)) AS n_duplicate_rows;

-- 2) Per-column missingness (UNPIVOT keeps it column-agnostic) ----------------
--    Edit the column list to match your schema.
WITH nulls AS (
    SELECT
        COUNT(*)                                   AS n,
        COUNT(*) - COUNT("Miles_per_Gallon")       AS "Miles_per_Gallon",
        COUNT(*) - COUNT("Horsepower")             AS "Horsepower",
        COUNT(*) - COUNT("Origin")                 AS "Origin"
    FROM raw
)
SELECT feature, missing,
       ROUND(100.0 * missing / n, 2) AS missing_pct
FROM nulls
UNPIVOT (missing FOR feature IN ("Miles_per_Gallon", "Horsepower", "Origin"))
ORDER BY missing_pct DESC;

-- 3) Numeric distribution diagnostics (incl. skewness & excess kurtosis) -----
SELECT
    'Horsepower'                       AS feature,
    COUNT("Horsepower")                AS n,
    ROUND(AVG("Horsepower"), 3)        AS mean,
    ROUND(STDDEV_SAMP("Horsepower"),3) AS std,
    MIN("Horsepower")                  AS min,
    ROUND(QUANTILE_CONT("Horsepower", 0.25), 3) AS q25,
    ROUND(MEDIAN("Horsepower"), 3)     AS median,
    ROUND(QUANTILE_CONT("Horsepower", 0.75), 3) AS q75,
    MAX("Horsepower")                  AS max,
    ROUND(SKEWNESS("Horsepower"), 4)   AS skewness,
    ROUND(KURTOSIS("Horsepower"), 4)   AS excess_kurtosis
FROM raw;

-- 4) Categorical frequency (top levels + cardinality) ------------------------
SELECT "Origin" AS level,
       COUNT(*) AS freq,
       ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 2) AS pct
FROM raw
GROUP BY "Origin"
ORDER BY freq DESC;

-- 5) Pairwise correlation for collinearity screening -------------------------
SELECT
    ROUND(CORR("Horsepower", "Weight_in_lbs"), 3)  AS hp_vs_weight,
    ROUND(CORR("Displacement", "Cylinders"), 3)    AS disp_vs_cyl,
    ROUND(CORR("Horsepower", "Displacement"), 3)   AS hp_vs_disp
FROM raw;
