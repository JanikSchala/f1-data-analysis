"""
P31 - Startphasen-Analyse: Wer gewinnt die ersten 500 Meter?
============================================================

Beschleunigung von der Startaufstellung bis Kurve 1 - Reaktion, Traktion und Positionsgewinn.

Kategorie:   Telemetrie
Niveau:      Fortgeschritten
Aufwand:     3-4 h
Schwerpunkt: Datenanalyse, Strategie

WARUM DAS LOHNT
Der Start ist die groesste Einzelchance im Rennen. Positionsdaten der ersten Runde sauber zu isolieren ist knifflig und zeigt Datensinn.

VORGEHEN
  1. Erste Runde aller Fahrer selektieren
  2. Telemetrie auf die ersten Sekunden nach Startfreigabe schneiden
  3. Zeit bis 100/200 km/h und Distanz nach 5 s berechnen
  4. Positionsgewinn Grid -> Ende Runde 1 gegenueberstellen

GENUTZTE FASTF1-BAUSTEINE
  - Laps.pick_lap
  - Lap.get_telemetry
  - Telemetry.slice_by_time
  - Session.results GridPosition

AUSBAUSTUFE
Vergleiche Starts derselben Fahrer ueber eine Saison und pruefe, ob Kupplungs-Updates messbar etwas gebracht haben.
"""

import pandas as pd
import matplotlib.pyplot as plt
import fastf1

fastf1.Cache.enable_cache("~/f1_cache")
ses = fastf1.get_session(2024, "Austria", "R")
ses.load()

rows = []
for drv in ses.drivers:
    info = ses.get_driver(drv)
    try:
        lap1 = ses.laps.pick_drivers(drv).pick_laps(1).iloc[0]
        tel = lap1.get_car_data().add_distance()
    except Exception:
        continue
    if tel is None or tel.empty:
        continue

    t = tel["Time"].dt.total_seconds().to_numpy()
    v = tel["Speed"].to_numpy()
    d = tel["Distance"].to_numpy()
    t0 = t[0]

    def t_at(target):
        idx = (v >= target).argmax() if (v >= target).any() else None
        return round(t[idx] - t0, 2) if idx is not None else None

    end_pos = lap1["Position"]
    grid = ses.results.loc[ses.results["DriverNumber"] == drv,
                           "GridPosition"].squeeze()

    rows.append({
        "Driver": info["Abbreviation"],
        "Grid": grid,
        "Ende_L1": end_pos,
        "Gewinn": (grid - end_pos) if pd.notna(end_pos) else None,
        "t_100": t_at(100),
        "t_200": t_at(200),
        "m_nach_5s": round(float(d[(t - t0) <= 5].max()), 1),
    })

df = pd.DataFrame(rows).sort_values("m_nach_5s", ascending=False)
print(df.to_string(index=False))

fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(df["m_nach_5s"], df["Gewinn"], s=80, color="#e10600")
for _, r in df.iterrows():
    ax.annotate(r["Driver"], (r["m_nach_5s"], r["Gewinn"]),
                xytext=(5, 3), textcoords="offset points", fontsize=8)
ax.axhline(0, color="grey", lw=0.8)
ax.set_xlabel("Distanz nach 5 s [m]")
ax.set_ylabel("Positionsgewinn Runde 1")
ax.set_title(f"{ses.event['EventName']} - Startperformance")
plt.tight_layout()
plt.show()
