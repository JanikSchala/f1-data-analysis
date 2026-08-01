"""
P12 - Kurvengeschwindigkeits-Profil je Fahrer
=============================================

Apex-Speed in jeder einzelnen Kurve, fuer alle Fahrer, als Heatmap. Zeigt Abtriebs- vs. Topspeed-Setups.

Kategorie:   Strecke & Position
Niveau:      Profi
Aufwand:     4 h
Schwerpunkt: Datenanalyse, Strategie

WARUM DAS LOHNT
Aus Kurvenspeeds liest ein Aerodynamiker das Setup ab. Die Verbindung Circuit-Info x Telemetrie ist genau die Art Feature-Engineering, die F1-Teams suchen.

VORGEHEN
  1. Kurvenpositionen in Distanz entlang der Runde umrechnen
  2. Fuer jede Kurve ein Distanzfenster definieren
  3. Minimalgeschwindigkeit im Fenster je Fahrer bestimmen
  4. Heatmap Fahrer x Kurve, normiert auf den Schnellsten

GENUTZTE FASTF1-BAUSTEINE
  - CircuitInfo.corners
  - Lap.get_telemetry
  - Telemetry.add_distance
  - numpy searchsorted

AUSBAUSTUFE
Cluster die Fahrer per k-Means anhand ihres Kurvenprofils - langsame Kurven vs. schnelle Kurven trennt Setups sauber.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import fastf1

fastf1.Cache.enable_cache("~/f1_cache")
ses = fastf1.get_session(2024, "Hungary", "Q")
ses.load()

ci = ses.get_circuit_info()
ref = ses.laps.pick_fastest().get_telemetry().add_distance()

# Kurven-XY auf die Referenzlinie projizieren -> Distanz
ref_xy = ref[["X", "Y"]].to_numpy(dtype=float)
corner_dist = []
for _, c in ci.corners.iterrows():
    d = np.hypot(ref_xy[:, 0] - c["X"], ref_xy[:, 1] - c["Y"])
    corner_dist.append(ref["Distance"].to_numpy()[np.argmin(d)])
ci_corners = ci.corners.assign(Distance=corner_dist)

WINDOW = 60   # +/- Meter um den Apex

rows = {}
for drv in ses.drivers:
    abb = ses.get_driver(drv)["Abbreviation"]
    try:
        tel = ses.laps.pick_drivers(drv).pick_fastest().get_telemetry().add_distance()
    except Exception:
        continue
    d = tel["Distance"].to_numpy()
    v = tel["Speed"].to_numpy()
    speeds = {}
    for _, c in ci_corners.iterrows():
        m = (d > c["Distance"] - WINDOW) & (d < c["Distance"] + WINDOW)
        if m.any():
            speeds[f"T{int(c['Number'])}{c['Letter']}"] = v[m].min()
    rows[abb] = speeds

mat = pd.DataFrame(rows).T
rel = mat.sub(mat.max(axis=0), axis=1)     # Delta zum Kurvenbesten

order = mat.mean(axis=1).sort_values(ascending=False).index
rel = rel.loc[order]

fig, ax = plt.subplots(figsize=(14, 8))
im = ax.imshow(rel.to_numpy(), cmap="RdYlGn", aspect="auto", vmin=-25, vmax=0)
ax.set_xticks(range(len(rel.columns)))
ax.set_xticklabels(rel.columns, rotation=90, fontsize=8)
ax.set_yticks(range(len(rel.index)))
ax.set_yticklabels(rel.index, fontsize=9)
fig.colorbar(im, label="Delta Apex-Speed zum Besten [km/h]")
ax.set_title(f"{ses.event['EventName']} - Kurvengeschwindigkeits-Profil")
plt.tight_layout()
plt.show()
