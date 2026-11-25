-- Business question: What exactly is in the analysed panel (size, closures, volume share per family)?
-- Output: per family: number of series, store-days, closed store-days, open days with zero sales,
-- total observed sales and its share of the 8-family total, in [$start, $cutoff].
-- Views expected: panel. Params: $start, $cutoff.
WITH p AS (
    SELECT * FROM panel
    WHERE is_future = 0 AND date BETWEEN CAST($start AS DATE) AND CAST($cutoff AS DATE)
)
SELECT family,
       COUNT(DISTINCT store_nbr)                         AS n_stores,
       COUNT(*)                                          AS n_store_days,
       SUM(closed)                                       AS n_closed_store_days,
       COUNT(*) FILTER (WHERE closed = 0 AND sales = 0)  AS n_open_zero_days,
       SUM(sales)                                        AS total_sales,
       SUM(sales) / SUM(SUM(sales)) OVER ()              AS volume_share,
       MIN(date)                                         AS first_date,
       MAX(date)                                         AS last_date
FROM p
GROUP BY family
ORDER BY total_sales DESC;
