"""
P09 - Gangwechsel-Karte der Strecke
===================================

Die Ideallinie in XY-Koordinaten, eingefaerbt nach eingelegtem Gang - das ikonische FastF1-Bild.

Kategorie:   Telemetrie
Niveau:      Fortgeschritten
Aufwand:     2-3 h
Schwerpunkt: Datenanalyse, Engineering

WARUM DAS LOHNT
Optisch stark und technisch lehrreich zugleich: LineCollection, Positionsdaten und diskrete Farbskalen. Das Bild, das man sich an die Wand haengt.

VORGEHEN
  1. Telemetrie der schnellsten Runde holen (get_telemetry mergt Car+Pos)
  2. X/Y zu Segment-Paaren umbauen
  3. LineCollection mit diskreter Colormap ueber nGear
  4. Achsen gleich skalieren, Colorbar mit Gaengen 1-8

GENUTZTE FASTF1-BAUSTEINE
  - Lap.get_telemetry
  - Telemetry X/Y/nGear
  - matplotlib LineCollection

AUSBAUSTUFE
Erzeuge dieselbe Karte fuer Speed, Throttle und DRS und stelle die vier Varianten als 2x2-Grid nebeneinander.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import fastf1

fastf1.Cache.enable_cache("~/f1_cache")
ses = fastf1.get_session(2024, "Belgium", "Q")
ses.load()

lap = ses.laps.pick_fastest()
tel = lap.get_telemetry()

x = tel["X"].to_numpy(dtype=float)
y = tel["Y"].to_numpy(dtype=float)
gear = tel["nGear"].to_numpy(dtype=float)

points = np.array([x, y]).T.reshape(-1, 1, 2)
segments = np.concatenate([points[:-1], points[1:]], axis=1)

cmap = plt.get_cmap("viridis", 8)
lc = LineCollection(segments, norm=plt.Normalize(1, 8), cmap=cmap)
lc.set_array(gear[:-1])
lc.set_linewidth(5)

fig, ax = plt.subplots(figsize=(10, 9))
ax.add_collection(lc)
ax.axis("equal")
ax.axis("off")
ax.set_title(f"{ses.event['EventName']} {ses.event.year}"
             f"\nGangwechsel - schnellste Runde {lap['Driver']}")

cbar = fig.colorbar(lc, ax=ax, boundaries=np.arange(0.5, 9.5))
cbar.set_ticks(np.arange(1, 9))
cbar.set_label("Gang")
plt.tight_layout()
plt.show()
