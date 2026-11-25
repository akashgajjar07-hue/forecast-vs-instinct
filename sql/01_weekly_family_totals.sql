-- Business question: How big is each product family across the 12 stores, and how is it trending week by week?
-- Output: one row per family x complete Monday-Sunday week inside [$start, $cutoff].
-- Sales = observed sales summed over the 12 stores (closed store-days are NULL and contribute nothing).
-- Views expected (registered by src/eda.py): panel.
-- Params: $start, $cutoff (default 2015-06-01 .. 2017-06-27, the last calibration target date).
WITH w AS (
    SELECT family,
           CAST(date_trunc('week', date) AS DATE) AS week_start,
           date, sales, closed
    FROM panel
    WHERE is_future = 0
      AND date BETWEEN CAST($start AS DATE) AND CAST($cutoff AS DATE)
)
SELECT family,
       week_start,
       SUM(sales)                         AS sales,
       COUNT(sales)                       AS n_open_store_days,
       SUM(closed)                        AS n_closed_store_days,
       COUNT(DISTINCT date)               AS n_dates
FROM w
GROUP BY family, week_start
HAVING COUNT(DISTINCT date) = 7          -- complete weeks only
ORDER BY family, week_start;
