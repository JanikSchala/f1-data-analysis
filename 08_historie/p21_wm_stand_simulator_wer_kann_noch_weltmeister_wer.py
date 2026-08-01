"""
P21 - WM-Stand-Simulator: Wer kann noch Weltmeister werden?
===========================================================

Aktuelle Standings holen, alle verbleibenden Punkte-Szenarien durchrechnen und Titelchancen simulieren.

Kategorie:   Historie & Ergast-API
Niveau:      Fortgeschritten
Aufwand:     3-4 h
Rollen:      DS, STRAT

WARUM DAS ZAEHLT
Kombinatorik plus Monte-Carlo auf echten Daten. Die Frage 'wann ist der Titel entschieden' beantwortet jeder F1-Sender - du machst es reproduzierbar.

VORGEHEN
  1. Aktuellen Fahrer- und Teamstand laden
  2. Verbleibende Rennen und Sprints zaehlen, Maximalpunkte bestimmen
  3. Rechnerische Titelchance (worst/best case) pruefen
  4. Monte-Carlo mit historischer Ergebnisverteilung je Fahrer

GENUTZTE FASTF1-BAUSTEINE
  - Ergast.get_driver_standings
  - Ergast.get_constructor_standings
  - get_events_remaining

AUSBAUSTUFE
Ergaenze Ausfallwahrscheinlichkeiten aus get_finishing_status() - DNFs veraendern die Chancen deutlich.
"""

import numpy as np
import pandas as pd
import fastf1
from fastf1.ergast import Ergast

fastf1.Cache.enable_cache("~/f1_cache")
erg = Ergast(result_type="pandas", auto_cast=True)
YEAR = 2024

standings = erg.get_driver_standings(season=YEAR).content[0]
standings = standings[["position", "points", "wins", "familyName", "constructorNames"]]
print(standings.head(10).to_string(index=False))

remaining = fastf1.get_events_remaining(include_testing=False)
n_races = len(remaining)
n_sprints = int(remaining["EventFormat"].str.contains("sprint", case=False).sum())
max_points = n_races * 25 + n_sprints * 8
print(f"\nVerbleibend: {n_races} Rennen, {n_sprints} Sprints -> max {max_points} Punkte")

leader = standings["points"].max()
standings["Rechnerisch moeglich"] = standings["points"] + max_points >= leader
print(standings[["familyName", "points", "Rechnerisch moeglich"]].to_string(index=False))

# --- Monte-Carlo auf Basis der bisherigen Saisonform ---
race_frames = []
for r in range(1, 25):
    try:
        race_frames.append(erg.get_race_results(season=YEAR, round=r).content[0])
    except Exception:
        break
res = pd.concat(race_frames, ignore_index=True)
PTS = {1: 25, 2: 18, 3: 15, 4: 12, 5: 10, 6: 8, 7: 6, 8: 4, 9: 2, 10: 1}
hist = res.groupby("driverId")["position"].apply(list).to_dict()

rng = np.random.default_rng(7)
titles = {d: 0 for d in hist}
N = 20000
base = res.groupby("driverId")["points"].sum()

for _ in range(N):
    tot = base.copy()
    for _ in range(n_races):
        for d, positions in hist.items():
            p = rng.choice(positions)
            tot[d] = tot.get(d, 0) + PTS.get(int(p), 0)
    titles[tot.idxmax()] += 1

chances = pd.Series({d: c / N * 100 for d, c in titles.items()})
print("\nTitelchance [%] (Monte-Carlo):")
print(chances[chances > 0.1].sort_values(ascending=False).round(1).to_string())
