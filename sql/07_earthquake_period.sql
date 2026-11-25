-- Business question: What happened to sales after the 16 April 2016 earthquake (M7.8, Manabi province), in the
-- Manabi stores versus the rest of the chain?
-- Periods (30 days each): pre = the 30 days before $quake_start; quake = $quake_start .. $quake_end;
-- post = the 30 days after $quake_end. Mean daily sales per store = SUM(sales) / open store-days.
-- Group: stores in Manabi state (53 Manta, 54 El Carmen) vs the other 10 stores.
-- Descriptive only (one event, no control group); some holidays (Viernes Santo, 1 May, Mother's Day) fall inside.
-- Views expected: panel. Params: $quake_start, $quake_end.
WITH d AS (
    SELECT family,
           CASE WHEN state = 'Manabi' THEN 'Manabi' ELSE 'Other' END AS store_group,
           CASE WHEN date BETWEEN CAST($quake_start AS DATE) - INTERVAL 30 DAY AND CAST($quake_start AS DATE) - INTERVAL 1 DAY THEN 'pre'
                WHEN date BETWEEN CAST($quake_start AS DATE) AND CAST($quake_end AS DATE) THEN 'quake'
                WHEN date BETWEEN CAST($quake_end AS DATE) + INTERVAL 1 DAY AND CAST($quake_end AS DATE) + INTERVAL 30 DAY THEN 'post'
           END AS period,
           store_nbr, date, sales
    FROM panel
    WHERE is_future = 0 AND closed = 0 AND sales IS NOT NULL
)
SELECT family, store_group, period,
       SUM(sales) / COUNT(*)       AS mean_daily_sales_per_store,
       SUM(sales)                  AS total_sales,
       COUNT(*)                    AS n_open_store_days,
       COUNT(DISTINCT store_nbr)   AS n_stores
FROM d
WHERE period IS NOT NULL
GROUP BY family, store_group, period
ORDER BY family, store_group, period;
