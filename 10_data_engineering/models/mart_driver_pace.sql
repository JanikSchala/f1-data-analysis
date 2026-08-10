-- Pace-Ranking: bereinigte Median-Rundenzeit je Fahrer und Event, plus die
-- relative Pace (Anteil ueber der schnellsten Median-Runde desselben
-- Events). Rohe Sekunden ueber Events hinweg zu mitteln waere blind fuer
-- Streckenlaenge (Monaco ~75s, Spa ~105s) - erst die relative Spalte macht
-- einen saison-weiten Vergleich moeglich.
WITH je_event AS (
    SELECT
        Season, EventName, Driver,
        count(*) AS runden,
        median(LapTime_s) AS median_pace_s
    FROM stg_fact_lap_clean
    GROUP BY 1, 2, 3
    HAVING count(*) >= 5
)
SELECT
    *,
    median_pace_s / min(median_pace_s) OVER (PARTITION BY Season, EventName)
        AS rel_pace
FROM je_event
