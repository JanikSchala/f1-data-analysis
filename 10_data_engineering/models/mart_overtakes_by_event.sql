-- Ueberholungen je Rennen (P38/P39-Baustein): zaehlt fact_overtake, gejoint
-- auf dim_event fuer den Streckennamen. Kein DRS-Zonen-Anteil hier - das
-- braucht Telemetrie (siehe P39), fact_overtake bewusst telemetriefrei
-- gehalten wie fact_lap/fact_pitstop.
SELECT
    e.event_name,
    o.Season,
    o.Round,
    count(*) AS ueberholungen,
    count(DISTINCT o.Gainer) AS fahrer_mit_mindestens_einer_ueberholung
FROM fact_overtake o
JOIN dim_event e ON e.season = o.Season AND e.round = o.Round
GROUP BY 1, 2, 3
