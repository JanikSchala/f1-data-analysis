-- Saubere Runden: gewertete Zeit, gruene Flagge, nicht gestrichen.
-- Basis fuer alle mart_-Modelle, damit dieselbe Definition von "sauber"
-- ueberall gilt statt in jeder Analyse neu erfunden zu werden.
SELECT *
FROM fact_lap
WHERE IsAccurate
  AND TrackStatus = '1'
  AND NOT Deleted
  AND LapTime_s > 0
