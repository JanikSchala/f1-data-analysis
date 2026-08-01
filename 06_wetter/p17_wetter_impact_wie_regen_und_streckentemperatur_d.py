"""
P17 - Wetter-Impact: Wie Regen und Streckentemperatur die Pace veraendern
=========================================================================

Wetterdaten im Minutentakt auf jede Runde joinen und den Effekt von TrackTemp auf die Rundenzeit quantifizieren.

Kategorie:   Wetter & Bedingungen
Niveau:      Fortgeschritten
Aufwand:     3-4 h
Rollen:      DS, STRAT

WARUM DAS ZAEHLT
Streckentemperatur steuert das Reifenfenster. Ein sauberer Zeit-Join zwischen zwei unterschiedlich getakteten Streams ist eine klassische Data-Engineering-Uebung.

VORGEHEN
  1. weather_data laden und den Verlauf plotten
  2. get_weather_data() je Runde zuordnen (naechster Messpunkt)
  3. Rundenzeit gegen TrackTemp regressieren, Fuel-Effekt vorher rausrechnen
  4. Trocken-/Nass-Phasen ueber Rainfall-Flag segmentieren

GENUTZTE FASTF1-BAUSTEINE
  - Session.weather_data
  - Laps.get_weather_data
  - AirTemp/TrackTemp/Humidity/Rainfall/WindSpeed

AUSBAUSTUFE
Baue einen Klassifikator, der allein aus Rundenzeit-Streuung und Speed-Traps erkennt, ob gerade Intermediates gefahren werden.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import fastf1

fastf1.Cache.enable_cache("~/f1_cache")
ses = fastf1.get_session(2024, "Brazil", "R")
ses.load(telemetry=False)

w = ses.weather_data
print(w[["Time", "AirTemp", "TrackTemp", "Humidity",
         "Rainfall", "WindSpeed"]].describe().round(2).to_string())

laps = (ses.laps.pick_wo_box().pick_accurate().pick_track_status("1")).copy()
wl = laps.get_weather_data().reset_index(drop=True)
laps = laps.reset_index(drop=True)
merged = pd.concat([laps, wl.loc[:, ~wl.columns.isin(laps.columns)]], axis=1)

merged["Sec"] = merged["LapTime"].dt.total_seconds()
total = ses.total_laps
merged["Corr"] = merged["Sec"] - (total - merged["LapNumber"]) * 1.8 * 0.03

dry = merged[~merged["Rainfall"]]
if len(dry) > 30:
    slope, inter = np.polyfit(dry["TrackTemp"], dry["Corr"], 1)
    print(f"\nEffekt Streckentemperatur: {slope:+.3f} s je Grad C")

print("\nPace nach Bedingung:")
print(merged.groupby("Rainfall")["Sec"].agg(["median", "std", "count"]).round(3))

fig, ax = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
ax[0].plot(w["Time"].dt.total_seconds() / 60, w["TrackTemp"], label="Strecke")
ax[0].plot(w["Time"].dt.total_seconds() / 60, w["AirTemp"], label="Luft")
ax[0].set_ylabel("Temperatur [C]"); ax[0].legend()
ax[1].scatter(merged["Time"].dt.total_seconds() / 60, merged["Sec"],
              c=merged["TrackTemp"], cmap="inferno", s=12)
ax[1].set_ylabel("Rundenzeit [s]"); ax[1].set_xlabel("Sessionzeit [min]")
fig.suptitle(f"{ses.event['EventName']} - Wetter und Pace")
plt.tight_layout()
plt.show()
