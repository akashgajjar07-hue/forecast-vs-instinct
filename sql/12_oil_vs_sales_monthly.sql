-- Business question: What did the oil price (Ecuador is an oil exporter) do over the window, next to sales?
-- Output: per month in [$start, $cutoff]: mean WTI daily close (oil.csv, missing days ignored) and the
-- 12-store, 8-family mean daily sales. Context only: oil is not a model feature here, and two trending
-- series will correlate whether or not one drives the other.
-- Views expected: panel, oil. Params: $start, $cutoff.
WITH s AS (
    SELECT CAST(date_trunc('month', date) AS DATE) AS month, date, SUM(sales) AS sales
    FROM panel
    WHERE is_future = 0 AND closed = 0 AND sales IS NOT NULL
      AND date BETWEEN CAST($start AS DATE) AND CAST($cutoff AS DATE)
    GROUP BY ALL
), sm AS (SELECT month, AVG(sales) AS mean_daily_sales, COUNT(*) AS n_days FROM s GROUP BY month),
o AS (
    SELECT CAST(date_trunc('month', CAST(date AS DATE)) AS DATE) AS month,
           AVG(dcoilwtico) AS oil_mean, COUNT(dcoilwtico) AS n_oil_days
    FROM oil
    WHERE CAST(date AS DATE) BETWEEN CAST($start AS DATE) AND CAST($cutoff AS DATE)
    GROUP BY 1
)
SELECT sm.month, sm.mean_daily_sales, sm.n_days, o.oil_mean, o.n_oil_days
FROM sm LEFT JOIN o USING (month)
ORDER BY sm.month;
