-- Business question: Which weekdays drive volume in each family?
-- Output: per family x ISO weekday (1 = Mon .. 7 = Sun): mean chain-level daily sales and an index
-- (weekday mean / mean of the 7 weekday means, so the 7 indices average exactly 1).
-- Days used: open days in [$start, $cutoff] that are NOT a national day off (holiday, transfer, bridge, additional),
-- so that holidays do not distort the weekday pattern.
-- Views expected: panel. Params: $start, $cutoff.
WITH daily AS (
    SELECT family, date, SUM(sales) AS sales
    FROM panel
    WHERE is_future = 0 AND closed = 0 AND sales IS NOT NULL
      AND date BETWEEN CAST($start AS DATE) AND CAST($cutoff AS DATE)
    GROUP BY family, date
    HAVING MAX(is_national_holiday) = 0
), dow AS (
    SELECT family, isodow(date) AS dow, AVG(sales) AS mean_sales, COUNT(*) AS n_days
    FROM daily
    GROUP BY family, isodow(date)
)
SELECT family, dow, mean_sales, n_days,
       mean_sales / AVG(mean_sales) OVER (PARTITION BY family) AS dow_index
FROM dow
ORDER BY family, dow;
