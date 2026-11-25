-- Business question: Why does the modelling window start on 2015-06-01? (PRODUCE recording gap.)
-- Output: per family x month, the share of OPEN store-days with zero recorded sales in the 12 scope stores,
-- from the raw train.csv (2013-01 .. $cutoff). Open store-day = the store sold something across all 33 families.
-- A high zero share on open days means the family was not being recorded, not that demand was zero.
-- Descriptive data-quality check; uses pre-window history on purpose.
-- Views expected: train (raw). Params: $cutoff, $stores (list of store numbers), $families (list).
WITH st AS (
    SELECT store_nbr, date, SUM(sales) AS tot
    FROM train
    WHERE store_nbr IN (SELECT UNNEST($stores))
    GROUP BY ALL
)
SELECT t.family,
       CAST(date_trunc('month', t.date) AS DATE) AS month,
       AVG(CAST(t.sales = 0 AS DOUBLE))          AS zero_share,
       COUNT(*)                                  AS n_open_store_days
FROM train t JOIN st ON st.store_nbr = t.store_nbr AND st.date = t.date
WHERE st.tot > 0
  AND t.family IN (SELECT UNNEST($families))
  AND CAST(t.date AS DATE) <= CAST($cutoff AS DATE)
GROUP BY ALL
ORDER BY t.family, month;
