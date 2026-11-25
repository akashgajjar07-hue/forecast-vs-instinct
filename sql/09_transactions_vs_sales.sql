-- Business question: How closely does store footfall (transactions = receipts) track sales in our 8 families?
-- EDA ONLY: transactions are NOT a model feature, because future transactions are unknown at forecast time.
-- Output: per store (and pooled, store_nbr NULL): Pearson correlation of daily transactions vs daily sales
-- summed over the 8 families, plus sales per transaction. Open days in [$start, $cutoff]; days with no
-- transactions record are dropped (n_missing_txn counts them).
-- Views expected: panel, transactions. Params: $start, $cutoff.
WITH s AS (
    SELECT store_nbr, date, SUM(sales) AS sales8
    FROM panel
    WHERE is_future = 0 AND closed = 0 AND sales IS NOT NULL
      AND date BETWEEN CAST($start AS DATE) AND CAST($cutoff AS DATE)
    GROUP BY store_nbr, date
), j AS (
    SELECT s.store_nbr, s.date, s.sales8, CAST(t.transactions AS DOUBLE) AS transactions
    FROM s LEFT JOIN transactions t ON t.store_nbr = s.store_nbr AND CAST(t.date AS DATE) = s.date
)
SELECT store_nbr,
       CORR(sales8, transactions)                          AS corr_daily,
       AVG(sales8 / transactions)                          AS sales_per_txn,
       AVG(transactions)                                   AS mean_transactions,
       COUNT(transactions)                                 AS n_days,
       COUNT(*) - COUNT(transactions)                      AS n_missing_txn
FROM j
GROUP BY GROUPING SETS ((store_nbr), ())
ORDER BY store_nbr NULLS LAST;
