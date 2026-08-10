-- Degradation-Proxy: Median-Rundenzeit je Mischung und Strecke.
-- Kein Fuel-korrigierter Degradations-Koeffizient (siehe f1lab.degradation()
-- fuer den echten Fit) - hier bewusst die einfache Warehouse-Kennzahl, die
-- sich direkt aus fact_lap ergibt, ohne die pandas-Regression von f1lab.core
-- in SQL nachzubauen.
SELECT
    EventName, Compound,
    count(*) AS runden,
    round(median(LapTime_s), 3) AS median_s
FROM stg_fact_lap_clean
WHERE Compound <> 'UNKNOWN'
GROUP BY 1, 2
HAVING count(*) > 50
