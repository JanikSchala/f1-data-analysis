"""
P08 - Bremspunkt-Detektor und Bremsphasen-Report
================================================

Aus dem Brake-Kanal automatisch alle Bremszonen extrahieren: Einsatzpunkt, Dauer, Delta-v, Verzoegerung.

Kategorie:   Telemetrie
Niveau:      Profi
Aufwand:     4-5 h
Schwerpunkt: Datenanalyse, Strategie

WARUM DAS LOHNT
Bremspunkte sind die wertvollste Einzelinformation im Fahrervergleich. Signal-Segmentierung ist echte Engineering-Arbeit, nicht nur Plotting.

VORGEHEN
  1. Brake-Kanal als bool auslesen, Flanken per diff finden
  2. Zusammenhaengende Bremszonen zu Intervallen gruppieren
  3. Je Zone Eintrittsspeed, Minimalspeed, Laenge, mittlere Verzoegerung berechnen
  4. Zonen zweier Fahrer nebeneinanderstellen

GENUTZTE FASTF1-BAUSTEINE
  - Telemetry Brake/Speed/Distance
  - Telemetry.add_distance
  - numpy diff

AUSBAUSTUFE
Matche die Zonen beider Fahrer ueber die Distanz und gib eine Tabelle 'wer bremst wie viele Meter spaeter' aus.
"""

import numpy as np
import pandas as pd
import fastf1

fastf1.Cache.enable_cache("~/f1_cache")
ses = fastf1.get_session(2024, "Austria", "Q")
ses.load()


def braking_zones(lap, min_len=20):
    tel = lap.get_car_data().add_distance()
    brake = tel["Brake"].astype(bool).to_numpy()
    dist = tel["Distance"].to_numpy()
    speed = tel["Speed"].to_numpy()
    t = tel["Time"].dt.total_seconds().to_numpy()

    edges = np.diff(brake.astype(int))
    starts = np.where(edges == 1)[0] + 1
    ends = np.where(edges == -1)[0] + 1
    if brake[0]:
        starts = np.r_[0, starts]
    if brake[-1]:
        ends = np.r_[ends, len(brake) - 1]

    out = []
    for s, e in zip(starts, ends):
        length = dist[e] - dist[s]
        if length < min_len:
            continue
        dt = max(t[e] - t[s], 1e-3)
        dv = (speed[s] - speed[e]) / 3.6          # m/s
        out.append({
            "Start_m": round(dist[s], 1),
            "Ende_m": round(dist[e], 1),
            "Laenge_m": round(length, 1),
            "v_ein": round(speed[s], 1),
            "v_min": round(speed[e], 1),
            "Dauer_s": round(dt, 2),
            "a_mittel_g": round(dv / dt / 9.81, 2),
        })
    return pd.DataFrame(out)


for drv in ("VER", "LEC"):
    lap = ses.laps.pick_drivers(drv).pick_fastest()
    zones = braking_zones(lap)
    print(f"\n=== {drv} - {len(zones)} Bremszonen ===")
    print(zones.to_string(index=False))
