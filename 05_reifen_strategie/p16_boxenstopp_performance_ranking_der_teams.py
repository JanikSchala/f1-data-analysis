"""
P16 - Boxenstopp-Performance-Ranking der Teams
==============================================

Standzeiten ueber eine ganze Saison aus der Ergast/jolpica-API - Median, Streuung und Konsistenz je Team.

Kategorie:   Reifen & Strategie
Niveau:      Fortgeschritten
Aufwand:     3 h
Rollen:      DS, ENG

WARUM DAS ZAEHLT
Boxenstopps sind messbare Teamleistung. Der Wechsel zwischen zwei Datenquellen (Timing-API und Ergast) zeigt, dass du mit heterogenen Quellen umgehen kannst.

VORGEHEN
  1. Ergast-Client mit result_type='pandas' aufsetzen
  2. Pitstops aller Rennen einer Saison holen (Paging beachten)
  3. Auf Konstrukteur joinen und Ausreisser (Reparaturen) kappen
  4. Ranking nach Median und Interquartilsabstand

GENUTZTE FASTF1-BAUSTEINE
  - fastf1.ergast.Ergast
  - Ergast.get_pit_stops
  - Ergast.get_race_schedule

AUSBAUSTUFE
Korreliere die Stopp-Zeiten mit der Position im Rennen - unter Druck sind Teams messbar langsamer.
"""

import pandas as pd
from fastf1.ergast import Ergast

erg = Ergast(result_type="pandas", auto_cast=True)
YEAR = 2024

sched = erg.get_race_schedule(season=YEAR)
frames = []

for _, race in sched.iterrows():
    rnd = int(race["round"])
    resp = erg.get_pit_stops(season=YEAR, round=rnd, limit=200)
    stops = resp.content[0] if resp.content else None
    if stops is None or stops.empty:
        continue
    res = erg.get_race_results(season=YEAR, round=rnd).content[0]
    stops = stops.merge(res[["driverId", "constructorName"]],
                        on="driverId", how="left")
    stops["round"] = rnd
    stops["raceName"] = race["raceName"]
    frames.append(stops)

pit = pd.concat(frames, ignore_index=True)
pit["dur"] = pd.to_timedelta(pit["duration"], errors="coerce").dt.total_seconds()
pit = pit[(pit["dur"] > 1.5) & (pit["dur"] < 6.0)]   # Reparaturen raus

rank = (pit.groupby("constructorName")["dur"]
        .agg(Median="median", Bester="min",
             IQR=lambda s: s.quantile(0.75) - s.quantile(0.25),
             Stopps="count")
        .sort_values("Median").round(3))

print(rank.to_string())
print("\nSchnellster Stopp der Saison:")
print(pit.nsmallest(5, "dur")[["raceName", "driverId",
                               "constructorName", "dur"]].to_string(index=False))
