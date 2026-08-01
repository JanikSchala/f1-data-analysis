"""
P06 - Sektor-Analyse: Wo genau geht die Zeit verloren?
======================================================

Sektorzeiten und Speed-Traps zerlegen die Rundenzeit in Aussagen ueber Abtrieb, Topspeed und Traktion.

Kategorie:   Timing & Rundenanalyse
Niveau:      Einsteiger
Aufwand:     2 h
Rollen:      DS, STRAT

WARUM DAS ZAEHLT
Sektor-Deltas sind die schnellste Diagnose fuer Auto-Charakteristik. Perfekt, um analytisches Denken zu demonstrieren, ohne dass es kompliziert wird.

VORGEHEN
  1. Beste Sektorzeit je Fahrer ermitteln
  2. Theoretisch beste Runde je Fahrer bilden (Summe der Bestsektoren)
  3. Delta zur tatsaechlich gefahrenen Bestzeit = ungenutztes Potenzial
  4. Speed-Traps gegen Sektorzeiten plotten

GENUTZTE FASTF1-BAUSTEINE
  - Laps.pick_fastest
  - Sector1Time/2/3
  - SpeedI1
  - SpeedI2
  - SpeedFL
  - SpeedST

AUSBAUSTUFE
Mache daraus eine 'Mini-Sektor'-Analyse: teile die Runde per Telemetrie-Distanz in 25 Abschnitte und faerbe die Strecke nach schnellstem Fahrer.
"""

import pandas as pd
import fastf1

fastf1.Cache.enable_cache("~/f1_cache")
ses = fastf1.get_session(2024, "Monaco", "Q")
ses.load(telemetry=False)

laps = ses.laps.pick_accurate().pick_wo_box()

sectors = laps.groupby("Driver")[["Sector1Time", "Sector2Time", "Sector3Time"]].min()
sectors = sectors.apply(lambda s: s.dt.total_seconds())
sectors["Theoretisch"] = sectors.sum(axis=1)

real = laps.groupby("Driver")["LapTime"].min().dt.total_seconds()
sectors["Real"] = real
sectors["Ungenutzt"] = sectors["Real"] - sectors["Theoretisch"]
sectors = sectors.sort_values("Real")

# Delta zur Bestzeit je Sektor
for i in (1, 2, 3):
    col = f"Sector{i}Time"
    sectors[f"dS{i}"] = sectors[col] - sectors[col].min()

print(sectors[["Real", "Theoretisch", "Ungenutzt",
               "dS1", "dS2", "dS3"]].round(3).to_string())

speeds = laps.groupby("Driver")[["SpeedI1", "SpeedI2", "SpeedFL", "SpeedST"]].max()
print("\n" + "Hoechstgeschwindigkeiten [km/h]:")
print(speeds.sort_values("SpeedST", ascending=False).round(1).to_string())
