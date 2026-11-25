-- Business question: Which store-family series are hard to plan (high volatility, zero-sales days)?
-- Output: one row per store x family over open days in [$start, $cutoff]:
--   mean       mean daily observed sales
--   cv         coefficient of variation of DAILY sales (stddev / mean; includes the weekday pattern)
--   cv_weekly  CV of complete-week totals (weekday pattern removed, closer to a planning view)
--   zero_share share of open days with sales = 0
-- Views expected: panel. Params: $start, $cutoff.
WITH d AS (
    SELECT store_nbr, family, date, sales
    FROM panel
    WHERE is_future = 0 AND closed = 0 AND sales IS NOT NULL
      AND date BETWEEN CAST($start AS DATE) AND CAST($cutoff AS DATE)
), wk AS (
    SELECT store_nbr, family, date_trunc('week', date) AS wk, SUM(sales) AS s, COUNT(*) AS n
    FROM d GROUP BY ALL
), wcv AS (
    SELECT store_nbr, family, STDDEV_SAMP(s) / AVG(s) AS cv_weekly
    FROM wk WHERE n = 7 GROUP BY ALL
)
SELECT d.store_nbr, d.family,
       AVG(sales)                          AS mean,
       STDDEV_SAMP(sales) / AVG(sales)     AS cv,
       ANY_VALUE(wcv.cv_weekly)            AS cv_weekly,
       AVG(CAST(sales = 0 AS DOUBLE))      AS zero_share,
       COUNT(*)                            AS n_open_days
FROM d JOIN wcv USING (store_nbr, family)
GROUP BY d.store_nbr, d.family
ORDER BY d.store_nbr, d.family;
