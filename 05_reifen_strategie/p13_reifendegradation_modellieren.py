"""
P13 - Reifendegradation modellieren
===================================

Pro Stint eine Regression Rundenzeit ~ TyreLife. Die Steigung ist die Degradation in s pro Runde.

Kategorie:   Reifen & Strategie
Niveau:      Fortgeschritten
Aufwand:     4-5 h
Rollen:      STRAT, DS

WARUM DAS ZAEHLT
Degradation ist die zentrale Groesse jeder Strategieentscheidung. Wenn du sie sauber schaetzt und Fuel-Effekt trennst, denkst du wie ein Strategie-Ingenieur.

VORGEHEN
  1. Stints ueber Driver+Stint gruppieren, In-/Out-Laps entfernen
  2. Fuel-Korrektur anwenden (ca. 0.03 s/Runde/kg, ~1.8 kg Verbrauch/Runde)
  3. Lineare Regression je Stint, Steigung = Degradation
  4. Aggregation je Compound und Vergleich der Teams

GENUTZTE FASTF1-BAUSTEINE
  - Laps Compound/TyreLife/Stint/FreshTyre
  - Laps.pick_compounds
  - numpy polyfit

AUSBAUSTUFE
Ersetze die lineare Regression durch ein Modell mit Cliff-Effekt (stueckweise linear) und finde den Knickpunkt je Compound.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import fastf1

fastf1.Cache.enable_cache("~/f1_cache")
ses = fastf1.get_session(2024, "Bahrain", "R")
ses.load(telemetry=False)

FUEL_EFFECT = 0.03      # s pro Runde pro kg
FUEL_PER_LAP = 1.8      # kg

laps = (ses.laps.pick_wo_box().pick_accurate()
        .pick_track_status("1").pick_quicklaps(1.10)).copy()
laps["Sec"] = laps["LapTime"].dt.total_seconds()

total = ses.total_laps
laps["FuelCorr"] = laps["Sec"] - (total - laps["LapNumber"]) * FUEL_PER_LAP * FUEL_EFFECT

rows = []
for (drv, stint), g in laps.groupby(["Driver", "Stint"]):
    g = g.sort_values("TyreLife")
    if len(g) < 6:
        continue
    slope, intercept = np.polyfit(g["TyreLife"], g["FuelCorr"], 1)
    rows.append({
        "Driver": drv,
        "Team": g["Team"].iloc[0],
        "Stint": int(stint),
        "Compound": g["Compound"].iloc[0],
        "Laps": len(g),
        "Deg_s_pro_Runde": round(slope, 4),
        "Basiszeit": round(intercept, 3),
    })

deg = pd.DataFrame(rows)
print(deg.sort_values("Deg_s_pro_Runde").to_string(index=False))

print("\nMittlere Degradation je Compound:")
print(deg.groupby("Compound")["Deg_s_pro_Runde"]
      .agg(["mean", "std", "count"]).round(4).to_string())

fig, ax = plt.subplots(figsize=(10, 6))
colors = {"SOFT": "#da291c", "MEDIUM": "#ffd12e", "HARD": "#f0f0ec",
          "INTERMEDIATE": "#43b02a", "WET": "#0067ad"}
for (drv, stint), g in laps.groupby(["Driver", "Stint"]):
    if len(g) < 6:
        continue
    c = colors.get(g["Compound"].iloc[0], "grey")
    ax.plot(g["TyreLife"], g["FuelCorr"], marker=".", lw=0.8, alpha=0.6, color=c)
ax.set_xlabel("Reifenalter [Runden]")
ax.set_ylabel("Fuel-korrigierte Rundenzeit [s]")
ax.set_title(f"{ses.event['EventName']} - Degradationskurven je Stint")
plt.tight_layout()
plt.show()
