-- Business question: Do sales differ on national holiday days versus normal days, by family?
-- Holiday flags in the panel are already matched to each store: National -> every store, Regional -> store state,
-- Local -> store city. Day-off rules: a 'Holiday' that was transferred is NOT a day off (the day it moved to, type
-- 'Transfer', is); 'Bridge' and 'Additional' days are days off; 'Work Day' (a weekend made a working day) is neither.
--   holiday day = open store-day with is_national_holiday = 1
--   normal day  = open store-day with no day off of ANY locale for that store, not a Work Day, not an Event
--                 (Black Friday, Mother's Day, earthquake aftermath...) and outside the earthquake month.
-- change_pct = 100 * (holiday_mean / normal_mean - 1). Raw means do not control for weekday mix: see
-- 10_ecuador_holiday_analogue.sql for the weekday-matched version.
-- index_change_pct uses each store-family's normal-day mean as the base (store-mix neutral).
-- change_pct_excl_additional drops national 'Additional' days (mostly the 21-26 Dec / 31 Dec Christmas run-up,
-- plus the eve of Mother's Day), which are shopping peaks rather than days off in the usual sense.
-- Closed days (25 Dec, 1 Jan: the chain closes) have no sales and drop out.
-- Views expected: panel, holidays. Params: $start, $cutoff.
WITH d AS (
    SELECT store_nbr, family, date, sales,
           date IN (SELECT CAST(date AS DATE) FROM holidays WHERE locale = 'National' AND type = 'Additional') AS is_additional,
           CASE WHEN is_national_holiday = 1 THEN 'holiday'
                WHEN is_holiday = 0 AND is_workday = 0 AND is_event = 0 AND quake = 0 THEN 'normal' END AS day_class
    FROM panel
    WHERE is_future = 0 AND closed = 0 AND sales IS NOT NULL
      AND date BETWEEN CAST($start AS DATE) AND CAST($cutoff AS DATE)
), s AS (
    SELECT *, AVG(CASE WHEN day_class = 'normal' THEN sales END) OVER (PARTITION BY store_nbr, family) AS series_normal_mean
    FROM d WHERE day_class IS NOT NULL
)
SELECT family,
       AVG(sales) FILTER (WHERE day_class = 'holiday')                    AS holiday_mean,
       AVG(sales) FILTER (WHERE day_class = 'normal')                     AS normal_mean,
       100 * (AVG(sales) FILTER (WHERE day_class = 'holiday')
              / AVG(sales) FILTER (WHERE day_class = 'normal') - 1)      AS change_pct,
       100 * (AVG(sales / series_normal_mean) FILTER (WHERE day_class = 'holiday') - 1) AS index_change_pct,
       100 * (AVG(sales) FILTER (WHERE day_class = 'holiday' AND NOT is_additional)
              / AVG(sales) FILTER (WHERE day_class = 'normal') - 1)      AS change_pct_excl_additional,
       CAST(COUNT(DISTINCT date) FILTER (WHERE day_class = 'holiday' AND NOT is_additional) AS INTEGER)
                                                                          AS n_holiday_days_excl_additional,
       CAST(COUNT(DISTINCT date) FILTER (WHERE day_class = 'holiday') AS INTEGER) AS n_holiday_days,
       CAST(COUNT(*) FILTER (WHERE day_class = 'holiday') AS INTEGER)     AS n_holiday_store_days,
       CAST(COUNT(*) FILTER (WHERE day_class = 'normal') AS INTEGER)      AS n_normal_store_days
FROM s
GROUP BY family
ORDER BY family;
