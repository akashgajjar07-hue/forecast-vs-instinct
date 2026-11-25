-- Business question: How often is each store-family on promotion, and how many items are typically promoted?
-- Output: per store x family over open days in [$start, $cutoff]: share of days with >= 1 promoted item,
-- mean and max promoted-item count, and the within-series correlation between promo count and sales
-- (an association only: no price/discount data, no causal claim).
-- Views expected: panel. Params: $start, $cutoff.
SELECT store_nbr, family,
       AVG(CAST(onpromotion > 0 AS DOUBLE))   AS promo_day_share,
       AVG(onpromotion)                       AS mean_promo_count,
       MAX(onpromotion)                       AS max_promo_count,
       CORR(CAST(onpromotion AS DOUBLE), sales) AS corr_promo_sales,
       COUNT(*)                               AS n_open_days
FROM panel
WHERE is_future = 0 AND closed = 0 AND sales IS NOT NULL
  AND date BETWEEN CAST($start AS DATE) AND CAST($cutoff AS DATE)
GROUP BY store_nbr, family
ORDER BY store_nbr, family;
