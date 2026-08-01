"""
P25 - Anomalie-Erkennung: Technische Probleme aus Telemetrie erkennen
=====================================================================

Isolation Forest auf Rundenprofilen, um Runden zu markieren, in denen etwas nicht stimmte - noch bevor der Fahrer funkt.

Kategorie:   Machine Learning
Niveau:      Profi
Aufwand:     5 h
Schwerpunkt: Datenanalyse, Engineering
Zusaetzliche Pakete: scikit-learn

WARUM DAS LOHNT
Predictive Maintenance in F1-Kontext. Anomalieerkennung ohne Labels ist genau das Problem, das Teams bei Zuverlaessigkeit tatsaechlich haben.

VORGEHEN
  1. Fuer jede Runde eines Fahrers ein Feature-Profil bauen
  2. Isolation Forest auf allen Runden trainieren
  3. Auffaellige Runden markieren und mit Race-Control-Messages abgleichen
  4. Anomalie-Score ueber den Rennverlauf plotten

GENUTZTE FASTF1-BAUSTEINE
  - Lap.get_car_data
  - Telemetry RPM/Speed/Throttle
  - sklearn IsolationForest

AUSBAUSTUFE
Vergleiche die Flags mit den Ausfallgruenden aus Session.results['Status'] - wie frueh haette das Modell gewarnt?
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import fastf1
from sklearn.ensemble import IsolationForest

fastf1.Cache.enable_cache("~/f1_cache")
ses = fastf1.get_session(2024, "Baku", "R")
ses.load()

rows = []
for _, lap in ses.laps.pick_wo_box().pick_accurate().iterlaps():
    tel = lap.get_car_data()
    if tel is None or len(tel) < 50:
        continue
    rows.append({
        "Driver": lap["Driver"],
        "Lap": lap["LapNumber"],
        "LapTime": lap["LapTime"].total_seconds(),
        "vmax": tel["Speed"].max(),
        "vmean": tel["Speed"].mean(),
        "rpm_max": tel["RPM"].max(),
        "rpm_mean": tel["RPM"].mean(),
        "throttle_mean": tel["Throttle"].mean(),
        "brake_pct": 100 * tel["Brake"].astype(bool).mean(),
        "gear_mean": tel["nGear"].mean(),
    })

df = pd.DataFrame(rows).dropna()

# Pro Fahrer normalisieren -> Abweichung vom eigenen Normalverhalten
feat = ["LapTime", "vmax", "vmean", "rpm_max", "rpm_mean",
        "throttle_mean", "brake_pct", "gear_mean"]
z = df.groupby("Driver")[feat].transform(lambda s: (s - s.mean()) / (s.std() + 1e-9))

iso = IsolationForest(contamination=0.04, random_state=0)
df["Anomalie"] = iso.fit_predict(z)
df["Score"] = -iso.score_samples(z)

flags = df[df["Anomalie"] == -1].sort_values("Score", ascending=False)
print("Auffaellige Runden:")
print(flags[["Driver", "Lap", "LapTime", "vmax", "rpm_max", "Score"]]
      .head(20).round(2).to_string(index=False))

fig, ax = plt.subplots(figsize=(12, 6))
for drv, g in df.groupby("Driver"):
    ax.plot(g["Lap"], g["Score"], lw=0.8, alpha=0.5)
ax.scatter(flags["Lap"], flags["Score"], color="red", s=30, zorder=3,
           label="Anomalie")
ax.set_xlabel("Runde"); ax.set_ylabel("Anomalie-Score")
ax.set_title(f"{ses.event['EventName']} - Anomalie-Erkennung")
ax.legend()
plt.tight_layout()
plt.show()
