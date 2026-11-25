-- Business question: How do observed sales vary with the NUMBER of promoted items in a family/store/day?
-- onpromotion is a COUNT of promoted items (no discount depth, no price). This is an ASSOCIATION, not causal uplift:
-- bigger stores promote more items and promotion counts trend over time.
-- Output: per family x promo-count bin: number of open store-days, mean and median sales, and
--   mean_index = mean of sales / (same store-family's mean for that calendar month), which removes store size
--   and slow trend so bins are comparable. n_stores = how many stores contribute to the bin.
-- Views expected: panel. Params: $start, $cutoff.
WITH base AS (
    SELECT store_nbr, family, date, sales, onpromotion,
           AVG(sales) OVER (PARTITION BY store_nbr, family, date_trunc('month', date)) AS month_mean
    FROM panel
    WHERE is_future = 0 AND closed = 0 AND sales IS NOT NULL
      AND date BETWEEN CAST($start AS DATE) AND CAST($cutoff AS DATE)
), b AS (
    SELECT *,
           CASE WHEN onpromotion = 0 THEN 0 WHEN onpromotion <= 5 THEN 1 WHEN onpromotion <= 15 THEN 2
                WHEN onpromotion <= 30 THEN 3 WHEN onpromotion <= 60 THEN 4 ELSE 5 END AS bin_order
    FROM base
)
SELECT family,
       bin_order,
       ['0', '1-5', '6-15', '16-30', '31-60', '61+'][bin_order + 1] AS bin,
       COUNT(*)                               AS n_days,
       AVG(sales)                             AS mean_sales,
       MEDIAN(sales)                          AS median_sales,
       AVG(sales / NULLIF(month_mean, 0))     AS mean_index,
       COUNT(DISTINCT store_nbr)              AS n_stores
FROM b
GROUP BY family, bin_order
ORDER BY family, bin_order;
