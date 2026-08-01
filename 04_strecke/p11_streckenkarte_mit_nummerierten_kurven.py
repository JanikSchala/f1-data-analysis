"""
P11 - Streckenkarte mit nummerierten Kurven
===========================================

get_circuit_info() liefert offizielle Kurvenpositionen und Marshal-Sektoren - daraus wird eine beschriftete Streckenkarte.

Kategorie:   Strecke & Position
Niveau:      Einsteiger
Aufwand:     2 h
Rollen:      DS, ENG

WARUM DAS ZAEHLT
Kurvennummern sind die Sprache im Team ('T7 Untersteuern'). Wer Telemetrie auf Kurven mappen kann, macht Analysen fuer Ingenieure lesbar.

VORGEHEN
  1. Circuit-Info und Positionsdaten der schnellsten Runde laden
  2. Track um circuit_info.rotation drehen, damit die Karte richtig liegt
  3. Kurvennummern mit Versatz ausserhalb der Strecke annotieren
  4. Marshal-Sektoren farbig markieren

GENUTZTE FASTF1-BAUSTEINE
  - Session.get_circuit_info
  - CircuitInfo.corners
  - CircuitInfo.marshal_sectors
  - CircuitInfo.rotation

AUSBAUSTUFE
Ordne jedem Telemetriepunkt die naechstgelegene Kurve zu und baue eine Tabelle 'Minimalgeschwindigkeit je Kurve' pro Fahrer.
"""

import numpy as np
import matplotlib.pyplot as plt
import fastf1

fastf1.Cache.enable_cache("~/f1_cache")
ses = fastf1.get_session(2024, "Zandvoort", "Q")
ses.load()

lap = ses.laps.pick_fastest()
pos = lap.get_pos_data()
ci = ses.get_circuit_info()


def rotate(xy, angle_deg):
    a = np.deg2rad(angle_deg)
    rot = np.array([[np.cos(a), np.sin(a)], [-np.sin(a), np.cos(a)]])
    return xy @ rot


track = pos.loc[:, ("X", "Y")].to_numpy(dtype=float)
track_r = rotate(track, ci.rotation)

fig, ax = plt.subplots(figsize=(10, 10))
ax.plot(track_r[:, 0], track_r[:, 1], color="black", lw=3)

OFFSET = 700
for _, corner in ci.corners.iterrows():
    label = f"{corner['Number']}{corner['Letter']}"
    ang = np.deg2rad(corner["Angle"])
    off = rotate(np.array([[OFFSET * np.cos(ang), OFFSET * np.sin(ang)]]),
                 ci.rotation)[0]
    base = rotate(np.array([[corner["X"], corner["Y"]]]), ci.rotation)[0]
    tip = base + off
    ax.plot([base[0], tip[0]], [base[1], tip[1]], color="grey", lw=1)
    ax.scatter(tip[0], tip[1], s=180, color="crimson", zorder=3)
    ax.text(tip[0], tip[1], label, color="white", va="center", ha="center",
            fontsize=8, zorder=4)

ax.set_title(f"{ses.event['EventName']} {ses.event.year} - Kurvenlayout")
ax.axis("equal")
ax.axis("off")
plt.tight_layout()
plt.show()
