"""
P20 - Positionsverlauf und Ueberholmatrix
=========================================

Position ueber Runden fuer alle Fahrer, plus eine Matrix wer wen wie oft ueberholt hat.

Kategorie:   Timing & Rundenanalyse
Niveau:      Fortgeschritten
Aufwand:     3 h
Rollen:      DS, STRAT

WARUM DAS ZAEHLT
Der Positionsgraph ist das meistgenutzte Rennbild ueberhaupt. Die Ueberholmatrix zeigt, dass du ueber die Standardgrafik hinausdenkst.

VORGEHEN
  1. Position je Runde und Fahrer pivotieren
  2. Linienplot in Fahrer-Styles, y-Achse invertiert
  3. Positionswechsel zwischen aufeinanderfolgenden Runden erkennen
  4. Ueberholungen ohne Boxenstopp-Runden zaehlen

GENUTZTE FASTF1-BAUSTEINE
  - Laps Position/LapNumber
  - Session.results
  - fastf1.plotting.get_driver_style
  - add_sorted_driver_legend

AUSBAUSTUFE
Kombiniere die Matrix mit DistanceToDriverAhead aus der Telemetrie, um versuchte gegen erfolgreiche Ueberholmanoever zu trennen.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import fastf1
import fastf1.plotting as f1plt

fastf1.Cache.enable_cache("~/f1_cache")
f1plt.setup_mpl(mpl_timedelta_support=False, color_scheme="fastf1")

ses = fastf1.get_session(2024, "Brazil", "R")
ses.load(telemetry=False)

laps = ses.laps
pos = laps.pivot_table(index="LapNumber", columns="Driver",
                       values="Position", aggfunc="first")

fig, ax = plt.subplots(figsize=(13, 8))
for drv in pos.columns:
    style = f1plt.get_driver_style(drv, ["color", "linestyle"], session=ses)
    ax.plot(pos.index, pos[drv], label=drv, lw=2, **style)
ax.set_ylim(20.5, 0.5)
ax.set_yticks(range(1, 21))
ax.set_xlabel("Runde"); ax.set_ylabel("Position")
ax.set_title(f"{ses.event['EventName']} {ses.event.year} - Positionsverlauf")
f1plt.add_sorted_driver_legend(ax, ses)
plt.tight_layout()
plt.show()

# --- Ueberholmatrix (ohne Box-Runden) ---
box_laps = set(zip(laps[laps["PitInTime"].notna()]["Driver"],
                   laps[laps["PitInTime"].notna()]["LapNumber"]))
drivers = list(pos.columns)
mat = pd.DataFrame(0, index=drivers, columns=drivers)

for lap in pos.index[1:]:
    prev, cur = pos.loc[lap - 1], pos.loc[lap]
    for a in drivers:
        for b in drivers:
            if a == b or pd.isna(prev[a]) or pd.isna(cur[a]):
                continue
            if pd.isna(prev[b]) or pd.isna(cur[b]):
                continue
            if prev[a] > prev[b] and cur[a] < cur[b]:
                if (b, lap) in box_laps or (a, lap - 1) in box_laps:
                    continue
                mat.loc[a, b] += 1

print("\nUeberholungen (Zeile ueberholt Spalte), Top-Werte:")
stacked = mat.stack().sort_values(ascending=False)
print(stacked[stacked > 0].head(15).to_string())
print("\nUeberholungen gesamt je Fahrer:")
print(mat.sum(axis=1).sort_values(ascending=False).to_string())
