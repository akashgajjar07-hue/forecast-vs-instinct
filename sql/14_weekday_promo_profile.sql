-- Business question: Do promotion counts follow a weekday rhythm that could explain a family's weekday pattern?
-- Output: per family x ISO weekday (1 = Mon .. 7 = Sun): mean promoted-item count and mean observed sales per
-- open store-day, for the full window and for the last 12 months before $cutoff. Association only.
-- Views expected: panel. Params: $start, $cutoff.
SELECT family, isodow(date) AS dow,
       AVG(onpromotion)                                                     AS mean_promo_count,
       AVG(sales)                                                           AS mean_sales,
       AVG(onpromotion) FILTER (WHERE date > CAST($cutoff AS DATE) - INTERVAL 365 DAY) AS mean_promo_count_last_12m,
       AVG(sales)       FILTER (WHERE date > CAST($cutoff AS DATE) - INTERVAL 365 DAY) AS mean_sales_last_12m,
       COUNT(*)                                                             AS n_store_days
FROM panel
WHERE is_future = 0 AND closed = 0 AND sales IS NOT NULL
  AND date BETWEEN CAST($start AS DATE) AND CAST($cutoff AS DATE)
GROUP BY family, isodow(date)
ORDER BY family, dow;
