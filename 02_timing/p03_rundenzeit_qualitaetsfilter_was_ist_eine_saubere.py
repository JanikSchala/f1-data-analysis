"""
P03 - Rundenzeit-Qualitaetsfilter: Was ist eine 'saubere' Runde?
================================================================

Rohdaten luegen. Out-Laps, In-Laps, Safety Car und geloeschte Runden verzerren jede Pace-Analyse.

Kategorie:   Timing & Rundenanalyse
Niveau:      Einsteiger
Aufwand:     2-3 h
Rollen:      DS, STRAT

WARUM DAS ZAEHLT
Das ist der haeufigste Anfaengerfehler in F1-Analysen. Wenn du im Gespraech erklaeren kannst, warum du track_status '1' filterst, klingst du wie jemand aus dem Fach.

VORGEHEN
  1. Alle Runden eines Rennens laden
  2. Schrittweise filtern und nach jedem Schritt zaehlen, wie viel wegfaellt
  3. Verteilung vorher/nachher als Boxplot vergleichen
  4. Eine wiederverwendbare clean_laps()-Funktion schreiben

GENUTZTE FASTF1-BAUSTEINE
  - Laps.pick_quicklaps
  - Laps.pick_accurate
  - Laps.pick_wo_box
  - Laps.pick_track_status
  - Laps.pick_not_deleted

AUSBAUSTUFE
Baue einen Ausreisser-Detektor auf Basis des Median Absolute Deviation und vergleiche ihn mit der 107%-Regel.
"""

import fastf1
import matplotlib.pyplot as plt

fastf1.Cache.enable_cache("~/f1_cache")
ses = fastf1.get_session(2024, "Spain", "R")
ses.load(telemetry=False)

laps = ses.laps
stages = {"roh": laps}
stages["ohne Box"]      = stages["roh"].pick_wo_box()
stages["nur akkurat"]   = stages["ohne Box"].pick_accurate()
stages["gruene Flagge"] = stages["nur akkurat"].pick_track_status("1")
stages["quicklaps"]     = stages["gruene Flagge"].pick_quicklaps(threshold=1.07)

for name, df in stages.items():
    sec = df["LapTime"].dt.total_seconds()
    print(f"{name:<15} n={len(df):>4}  median={sec.median():.3f}s  std={sec.std():.3f}s")


def clean_laps(session, threshold=1.07):
    """Wiederverwendbarer Filter fuer Pace-Analysen."""
    return (session.laps
            .pick_wo_box()
            .pick_accurate()
            .pick_track_status("1")
            .pick_not_deleted()
            .pick_quicklaps(threshold=threshold))


clean = clean_laps(ses)
fig, ax = plt.subplots(figsize=(9, 5))
ax.boxplot([laps["LapTime"].dt.total_seconds().dropna(),
            clean["LapTime"].dt.total_seconds().dropna()])
ax.set_xticks([1, 2], ["roh", "gefiltert"])
ax.set_ylabel("Rundenzeit [s]")
ax.set_title(f"{ses.event['EventName']} - Effekt der Datenbereinigung")
plt.tight_layout()
plt.show()
