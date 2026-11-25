-- Business question: Do sales rise around public-sector paydays (the 15th and the last day of the month)?
-- Method: index = sales / mean sales of the SAME store-family, SAME calendar month and SAME weekday on non-payday
-- days. This removes store size, trend and the weekday pattern. Index 1.00 = a normal day.
-- Days used: open days in [$start, $cutoff] with no day off, no Work Day, no Event, outside the earthquake month.
-- day_class: payday (15th or month end), day_after (16th or 1st), other.
-- Output uses GROUPING SETS: rows with family/day_class set = payday comparison; rows with day_of_month set
-- = chain-wide (all families pooled) day-of-month profile.
-- Views expected: panel. Params: $start, $cutoff.
WITH d AS (
    SELECT store_nbr, family, date, sales, payday,
           CASE WHEN payday = 1 THEN 'payday'
                WHEN day(date) IN (1, 16) THEN 'day_after'
                ELSE 'other' END AS day_class
    FROM panel
    WHERE is_future = 0 AND closed = 0 AND sales IS NOT NULL
      AND is_holiday = 0 AND is_workday = 0 AND is_event = 0 AND quake = 0
      AND date BETWEEN CAST($start AS DATE) AND CAST($cutoff AS DATE)
), r AS (
    SELECT *,
           AVG(CASE WHEN payday = 0 THEN sales END)
               OVER (PARTITION BY store_nbr, family, date_trunc('month', date), isodow(date)) AS ref
    FROM d
)
SELECT family, day_class, day(date) AS day_of_month,
       AVG(sales / ref)  AS mean_index,
       COUNT(*)          AS n_store_days
FROM r
WHERE ref > 0
GROUP BY GROUPING SETS ((family, day_class), (day(date)))
ORDER BY family NULLS LAST, day_class, day_of_month;
