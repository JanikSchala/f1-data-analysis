"""
P04 - Race Pace Ranking: Wer war wirklich am schnellsten?
=========================================================

Endergebnis != Pace. Rangliste nach bereinigter Median-Pace pro Fahrer und Team, mit Konfidenzintervall.

Kategorie:   Timing & Rundenanalyse
Niveau:      Fortgeschritten
Aufwand:     3-4 h
Schwerpunkt: Datenanalyse, Strategie

WARUM DAS LOHNT
Genau diese Frage stellt sich an der Boxenmauer nach jedem Rennen. Ein Ranking ohne Unsicherheitsangabe ist nur eine Meinung mit Nachkommastellen.

VORGEHEN
  1. Saubere Runden filtern
  2. Median-Pace je Fahrer berechnen, Delta zum Schnellsten
  3. Bootstrap-Konfidenzintervall (1000 Resamples)
  4. Horizontaler Barplot in Team-Farben

GENUTZTE FASTF1-BAUSTEINE
  - Session.laps
  - Laps.pick_quicklaps
  - Laps.groupby
  - fastf1.plotting.get_team_color

AUSBAUSTUFE
Rechne den Treibstoffeffekt heraus (ca. 0.03 s/Runde/kg) und normalisiere alle Runden auf gleiche Tankfuellung.
"""

import numpy as np
import pandas as pd
import fastf1
import fastf1.plotting as f1plt
import matplotlib.pyplot as plt

fastf1.Cache.enable_cache("~/f1_cache")
f1plt.setup_mpl(mpl_timedelta_support=False, color_scheme="fastf1")

ses = fastf1.get_session(2024, "Silverstone", "R")
ses.load(telemetry=False)

laps = (ses.laps.pick_wo_box().pick_accurate()
        .pick_track_status("1").pick_quicklaps())
laps = laps.copy()
laps["Sec"] = laps["LapTime"].dt.total_seconds()

rng = np.random.default_rng(42)

rows = []
for drv, grp in laps.groupby("Driver"):
    vals = grp["Sec"].to_numpy()
    if len(vals) < 8:
        continue
    boots = [np.median(rng.choice(vals, len(vals), replace=True))
             for _ in range(1000)]
    rows.append({
        "Driver": drv,
        "Team": grp["Team"].iloc[0],
        "Laps": len(vals),
        "Median": np.median(vals),
        "Lo": np.percentile(boots, 2.5),
        "Hi": np.percentile(boots, 97.5),
    })

pace = pd.DataFrame(rows).sort_values("Median").reset_index(drop=True)
best = pace["Median"].iloc[0]
for c in ["Median", "Lo", "Hi"]:
    pace[c + "Delta"] = pace[c] - best

print(pace[["Driver", "Team", "Laps", "MedianDelta"]].round(3).to_string(index=False))

fig, ax = plt.subplots(figsize=(9, 8))
colors = [f1plt.get_team_color(t, session=ses) for t in pace["Team"]]
err = [pace["MedianDelta"] - pace["LoDelta"], pace["HiDelta"] - pace["MedianDelta"]]
ax.barh(pace["Driver"], pace["MedianDelta"], xerr=err, color=colors)
ax.invert_yaxis()
ax.set_xlabel("Delta zur besten Race Pace [s/Runde]")
ax.set_title(f"{ses.event['EventName']} {ses.event.year} - bereinigte Race Pace")
plt.tight_layout()
plt.show()
