-- Business question: On average, how much do sales of each family move on a national holiday in Ecuador,
-- compared with the same weekday in the surrounding weeks? (Used as a real-data analogue for event scenarios.)
-- Definition (precise):
--   Holiday dates H = National rows of holidays_events with (type = 'Holiday' AND NOT transferred) OR
--     type IN ('Transfer', 'Bridge'), dated in [$start, $cutoff]. 'Additional' days (e.g. Christmas run-up),
--     Events and Work Days are NOT holiday dates.
--   Target date t = h + $offset days (offset 0 = the holiday itself, -1 = the day before). For offset <> 0
--     the target must not itself be a day off for that store.
--   Baseline for store s = mean of sales on t-14, t-7, t+7, t+14 (same weekday) keeping only days that are open,
--     in [$start, $cutoff], not a day off / Work Day / Event of any locale for that store, outside the earthquake
--     month. A store needs >= $min_baseline valid baseline days and to be open on t.
--   Per holiday x family: change_pct = 100 * (SUM over valid stores of sales on t
--                                            / SUM over the same stores of baseline mean - 1).
--   The family figure (computed in src/eda.py) = simple mean of the per-holiday change_pct (each holiday counts once).
--   Holidays on which the chain is closed (25 Dec, 1 Jan) have no sales and drop out.
-- Associations from observed sales, not unconstrained demand.
-- Views expected: panel, holidays. Params: $start, $cutoff, $offset, $min_baseline.
WITH h AS (
    SELECT CAST(date AS DATE) AS holiday_date,
           string_agg(DISTINCT description, ' / ') AS description,
           string_agg(DISTINCT type, ' / ')        AS holiday_type
    FROM holidays
    WHERE locale = 'National'
      AND ((type = 'Holiday' AND NOT transferred) OR type IN ('Transfer', 'Bridge'))
      AND CAST(date AS DATE) BETWEEN CAST($start AS DATE) AND CAST($cutoff AS DATE)
    GROUP BY 1
), t AS (
    SELECT h.holiday_date, h.description, h.holiday_type,
           CAST(h.holiday_date + to_days(CAST($offset AS INTEGER)) AS DATE) AS target_date
    FROM h
), tgt AS (
    SELECT t.*, p.store_nbr, p.family, p.sales AS target_sales
    FROM t JOIN panel p ON p.date = t.target_date
    WHERE p.closed = 0 AND p.sales IS NOT NULL AND p.is_future = 0
      AND (CAST($offset AS INTEGER) = 0 OR p.is_holiday = 0)
), base AS (
    SELECT tgt.holiday_date, tgt.store_nbr, tgt.family,
           AVG(b.sales) AS baseline_mean, COUNT(b.sales) AS n_baseline
    FROM tgt JOIN panel b
      ON b.store_nbr = tgt.store_nbr AND b.family = tgt.family
     AND b.date IN (tgt.target_date - INTERVAL 14 DAY, tgt.target_date - INTERVAL 7 DAY,
                    tgt.target_date + INTERVAL 7 DAY, tgt.target_date + INTERVAL 14 DAY)
    WHERE b.closed = 0 AND b.sales IS NOT NULL AND b.is_future = 0
      AND b.is_holiday = 0 AND b.is_workday = 0 AND b.is_event = 0 AND b.quake = 0
      AND b.date BETWEEN CAST($start AS DATE) AND CAST($cutoff AS DATE)
    GROUP BY ALL
), j AS (
    SELECT tgt.*, base.baseline_mean, base.n_baseline
    FROM tgt JOIN base USING (holiday_date, store_nbr, family)
    WHERE base.n_baseline >= CAST($min_baseline AS INTEGER)
)
SELECT holiday_date, ANY_VALUE(description) AS description, ANY_VALUE(holiday_type) AS holiday_type,
       ANY_VALUE(target_date) AS target_date, isodow(ANY_VALUE(target_date)) AS target_dow, family,
       SUM(target_sales)                        AS target_sales,
       SUM(baseline_mean)                       AS baseline_sales,
       100 * (SUM(target_sales) / SUM(baseline_mean) - 1) AS change_pct,
       COUNT(*)                                 AS n_stores
FROM j
GROUP BY holiday_date, family
ORDER BY holiday_date, family;
