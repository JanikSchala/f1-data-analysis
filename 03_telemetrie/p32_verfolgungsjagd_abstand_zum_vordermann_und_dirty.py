"""
P32 - Verfolgungsjagd: Abstand zum Vordermann und Dirty Air
===========================================================

DistanceToDriverAhead auswerten: Wie viel Pace kostet es, im Windschatten zu haengen?

Kategorie:   Telemetrie
Niveau:      Profi
Aufwand:     4-5 h
Rollen:      STRAT, DS

WARUM DAS ZAEHLT
Dirty Air ist DAS Thema seit der Regelaenderung 2022. Der add_driver_ahead()-Kanal ist rechenintensiv - dass du ihn nutzt, zeigt Tiefe.

VORGEHEN
  1. Telemetrie eines Fahrers mit add_driver_ahead() anreichern
  2. Je Runde den Medianabstand zum Vordermann berechnen
  3. Rundenzeit gegen Abstand regressieren (nur gruene Runden)
  4. Schwellenwert finden, ab dem Dirty Air spuerbar wird

GENUTZTE FASTF1-BAUSTEINE
  - Telemetry.add_driver_ahead
  - DistanceToDriverAhead
  - DriverAhead
  - Laps.get_telemetry

AUSBAUSTUFE
Wiederhole die Analyse fuer mehrere Strecken und pruefe, wo Dirty Air am staerksten wirkt - Highspeed- vs. Winkelkurse.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import fastf1

fastf1.Cache.enable_cache("~/f1_cache")
ses = fastf1.get_session(2024, "Spain", "R")
ses.load()

DRIVER = "SAI"
laps = (ses.laps.pick_drivers(DRIVER)
        .pick_wo_box().pick_accurate().pick_track_status("1"))

rows = []
for _, lap in laps.iterlaps():
    try:
        tel = lap.get_telemetry().add_driver_ahead()
    except Exception:
        continue
    if tel is None or tel.empty:
        continue
    gap = tel["DistanceToDriverAhead"].replace(0, np.nan)
    rows.append({
        "Lap": lap["LapNumber"],
        "Sec": lap["LapTime"].total_seconds(),
        "Gap_median_m": gap.median(),
        "Gap_min_m": gap.min(),
        "Anteil_unter_50m": 100 * (gap < 50).mean(),
        "Compound": lap["Compound"],
        "TyreLife": lap["TyreLife"],
    })

df = pd.DataFrame(rows).dropna(subset=["Gap_median_m"])
df = df[df["Gap_median_m"] < 500]

# Reifenalter herausrechnen, dann Effekt des Abstands isolieren
A = np.column_stack([df["TyreLife"], np.ones(len(df))])
coef, *_ = np.linalg.lstsq(A, df["Sec"], rcond=None)
df["Sec_corr"] = df["Sec"] - coef[0] * df["TyreLife"]

slope, inter = np.polyfit(df["Anteil_unter_50m"], df["Sec_corr"], 1)
print(f"Dirty-Air-Effekt: {slope:+.3f} s je 1% Zeitanteil unter 50 m Abstand")
print(df.round(2).to_string(index=False))

fig, ax = plt.subplots(figsize=(10, 6))
sc = ax.scatter(df["Anteil_unter_50m"], df["Sec_corr"],
                c=df["TyreLife"], cmap="viridis", s=70)
xs = np.linspace(df["Anteil_unter_50m"].min(), df["Anteil_unter_50m"].max(), 50)
ax.plot(xs, slope * xs + inter, color="red", lw=2)
fig.colorbar(sc, label="Reifenalter [Runden]")
ax.set_xlabel("Zeitanteil unter 50 m hinter Vordermann [%]")
ax.set_ylabel("Reifenkorrigierte Rundenzeit [s]")
ax.set_title(f"{DRIVER} - {ses.event['EventName']}: Dirty-Air-Effekt")
plt.tight_layout()
plt.show()
