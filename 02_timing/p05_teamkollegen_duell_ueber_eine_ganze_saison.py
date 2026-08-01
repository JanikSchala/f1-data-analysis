"""
P05 - Teamkollegen-Duell ueber eine ganze Saison
================================================

Der fairste Leistungsvergleich in der F1: gleiches Auto, gleiche Strecke. Quali-Delta und Race-Delta ueber 24 Rennen.

Kategorie:   Timing & Rundenanalyse
Niveau:      Fortgeschritten
Aufwand:     4-5 h
Schwerpunkt: Datenanalyse, Strategie

WARUM DAS LOHNT
Teamintern ist genau das die Kernfrage bei Fahrerbewertung. Zeigt, dass du ueber Einzelrennen hinaus denkst und mit Batch-Verarbeitung umgehen kannst.

VORGEHEN
  1. Alle Rennen einer Saison iterieren (mit try/except pro Event)
  2. Pro Team die zwei Fahrer identifizieren
  3. Quali: bestes Q-Ergebnis vergleichen, Prozent-Delta bilden
  4. Rennen: Median saubere Runde vergleichen
  5. Heatmap Team x Rennen

GENUTZTE FASTF1-BAUSTEINE
  - Session.results
  - Laps.split_qualifying_sessions
  - Laps.pick_drivers
  - get_event_schedule

AUSBAUSTUFE
Ergaenze das Ganze um ein Elo-artiges Rating, das nach jedem Wochenende aktualisiert wird.
"""

import numpy as np
import pandas as pd
import fastf1

fastf1.Cache.enable_cache("~/f1_cache")
YEAR = 2024

sched = fastf1.get_event_schedule(YEAR, include_testing=False)
records = []

for _, ev in sched.iterrows():
    rnd = int(ev["RoundNumber"])
    try:
        q = fastf1.get_session(YEAR, rnd, "Q")
        q.load(telemetry=False, weather=False, messages=False)
    except Exception as exc:
        print(f"R{rnd} uebersprungen: {exc}")
        continue

    laps = q.laps.pick_accurate().pick_wo_box()
    best = (laps.groupby(["Team", "Driver"])["LapTime"].min()
                .dt.total_seconds().reset_index())

    for team, grp in best.groupby("Team"):
        if len(grp) != 2:
            continue
        grp = grp.sort_values("LapTime")
        fast, slow = grp.iloc[0], grp.iloc[1]
        records.append({
            "Round": rnd,
            "Event": ev["EventName"],
            "Team": team,
            "Faster": fast["Driver"],
            "Slower": slow["Driver"],
            "DeltaPct": (slow["LapTime"] / fast["LapTime"] - 1) * 100,
        })

duel = pd.DataFrame(records)

summary = (duel.groupby(["Team", "Faster"]).size()
           .rename("Quali-Siege").reset_index()
           .sort_values(["Team", "Quali-Siege"], ascending=[True, False]))
print(summary.to_string(index=False))

pivot = duel.pivot_table(index="Team", columns="Round",
                         values="DeltaPct", aggfunc="first")
print("\n" + "Mittleres Quali-Delta je Team [%]:")
print(pivot.mean(axis=1).sort_values().round(3))
