"""
P33 - Streckenvergleich ueber Jahre: Hat sich das Layout geaendert?
===================================================================

Dieselbe Strecke in verschiedenen Saisons: Streckenlaenge aus Telemetrie, Speed-Profile und Rundenzeitentwicklung.

Kategorie:   Strecke & Position
Niveau:      Fortgeschritten
Aufwand:     3-4 h
Rollen:      DS, ENG

WARUM DAS ZAEHLT
Zeigt, dass du Daten ueber Zeit hinweg vergleichbar machen kannst - Normalisierung ist eine der schwersten Aufgaben in echten Datenprojekten.

VORGEHEN
  1. Pole-Runde derselben Strecke fuer 5 Saisons laden
  2. Speed ueber RelativeDistance interpolieren, damit die Kurven vergleichbar sind
  3. Speed-Profile uebereinanderlegen
  4. Rundenzeit und Kurvenanzahl je Jahr gegenueberstellen

GENUTZTE FASTF1-BAUSTEINE
  - Telemetry.add_distance
  - Telemetry.add_relative_distance
  - get_circuit_info
  - mehrere Sessions

AUSBAUSTUFE
Ergaenze die Regelaera als Kategorie (2017-2021 vs. ab 2022) und quantifiziere, wo die Ground-Effect-Autos schneller sind.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import fastf1

fastf1.Cache.enable_cache("~/f1_cache")
GP = "Spain"
YEARS = [2019, 2021, 2022, 2023, 2024]

grid = np.linspace(0, 1, 1000)
profiles, meta = {}, []

for y in YEARS:
    try:
        ses = fastf1.get_session(y, GP, "Q")
        ses.load()
    except Exception as exc:
        print(f"{y}: {exc}")
        continue

    lap = ses.laps.pick_fastest()
    tel = lap.get_telemetry().add_distance().add_relative_distance()
    rel = tel["RelativeDistance"].to_numpy()
    v = tel["Speed"].to_numpy()
    ok = ~np.isnan(rel) & ~np.isnan(v)
    profiles[y] = np.interp(grid, rel[ok], v[ok])

    ci = ses.get_circuit_info()
    meta.append({
        "Jahr": y,
        "Pole": lap["Driver"],
        "Zeit_s": round(lap["LapTime"].total_seconds(), 3),
        "Laenge_m": round(tel["Distance"].max(), 0),
        "Kurven": len(ci.corners),
        "vmax": round(v.max(), 1),
        "vmean": round(np.average(v), 1),
    })

info = pd.DataFrame(meta)
print(info.to_string(index=False))

fig, ax = plt.subplots(2, 1, figsize=(13, 9), sharex=True)
for y, prof in profiles.items():
    ax[0].plot(grid * 100, prof, label=str(y), lw=1.4)
ax[0].set_ylabel("Speed [km/h]"); ax[0].legend(ncol=len(profiles))
ax[0].set_title(f"{GP} - Speed-Profil der Pole-Runde ueber die Jahre")

ref = profiles[max(profiles)]
for y, prof in profiles.items():
    if y == max(profiles):
        continue
    ax[1].plot(grid * 100, prof - ref, label=f"{y} vs {max(profiles)}", lw=1.2)
ax[1].axhline(0, color="grey", lw=0.8)
ax[1].set_ylabel("Delta Speed [km/h]"); ax[1].set_xlabel("Rundenanteil [%]")
ax[1].legend(ncol=4)
plt.tight_layout()
plt.show()
