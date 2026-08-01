"""
P14 - Stint- und Strategie-Uebersicht als Gantt-Chart
=====================================================

Die klassische Strategiegrafik: jeder Fahrer eine Zeile, jeder Stint ein Balken in Compound-Farbe.

Kategorie:   Reifen & Strategie
Niveau:      Einsteiger
Aufwand:     2-3 h
Rollen:      STRAT, DS

WARUM DAS ZAEHLT
Diese Grafik hat jedes Team an der Boxenmauer. Schnell gebaut, sofort verstaendlich, ideal als Einstiegsbild fuer dein Portfolio.

VORGEHEN
  1. Stints je Fahrer aggregieren: Start-Runde, Laenge, Compound
  2. Nach Endposition sortieren
  3. Gestapelte horizontale Balken mit Compound-Farben
  4. Compound-Kuerzel und Stintlaenge beschriften

GENUTZTE FASTF1-BAUSTEINE
  - Laps groupby Stint
  - Laps Compound/FreshTyre
  - fastf1.plotting.get_compound_color

AUSBAUSTUFE
Ergaenze eine zweite Spalte mit dem Positionsgewinn/-verlust je Stint, damit man sieht, welche Strategie sich ausgezahlt hat.
"""

import pandas as pd
import matplotlib.pyplot as plt
import fastf1
import fastf1.plotting as f1plt

fastf1.Cache.enable_cache("~/f1_cache")
ses = fastf1.get_session(2024, "Hungary", "R")
ses.load(telemetry=False)

laps = ses.laps
stints = (laps.groupby(["Driver", "Stint", "Compound"])
          .agg(Start=("LapNumber", "min"),
               Ende=("LapNumber", "max"),
               Laenge=("LapNumber", "count"))
          .reset_index())

order = ses.results.sort_values("Position")["Abbreviation"].tolist()
order = [d for d in order if d in stints["Driver"].unique()]

fig, ax = plt.subplots(figsize=(11, 9))
for drv in order:
    prev = 0
    for _, s in stints[stints["Driver"] == drv].sort_values("Stint").iterrows():
        color = f1plt.get_compound_color(s["Compound"], session=ses)
        ax.barh(drv, s["Laenge"], left=prev, color=color,
                edgecolor="black", height=0.7)
        if s["Laenge"] > 4:
            ax.text(prev + s["Laenge"] / 2, drv, s["Compound"][0],
                    ha="center", va="center", fontsize=8, fontweight="bold")
        prev += s["Laenge"]

ax.invert_yaxis()
ax.set_xlabel("Runde")
ax.set_title(f"{ses.event['EventName']} {ses.event.year} - Reifenstrategien")
ax.grid(axis="x", alpha=0.3)
plt.tight_layout()
plt.show()

print(stints.groupby("Compound")["Laenge"].agg(["mean", "max", "count"]).round(1))
