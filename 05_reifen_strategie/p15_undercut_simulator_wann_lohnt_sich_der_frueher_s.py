"""
P15 - Undercut-Simulator: Wann lohnt sich der frueher Stopp?
============================================================

Ein Modell aus Degradation, Pitloss und Out-Lap-Performance, das den Undercut-Gewinn in Sekunden ausrechnet.

Kategorie:   Reifen & Strategie
Niveau:      Profi
Aufwand:     5-6 h
Schwerpunkt: Strategie

WARUM DAS LOHNT
Das ist Race Strategy in Reinform. Ein funktionierender Simulator mit aus Daten geschaetzten Parametern ist das staerkste Einzelprojekt fuer eine Strategie-Rolle.

VORGEHEN
  1. Pitloss aus echten Daten schaetzen (Delta In-/Out-Lap zur Normalrunde)
  2. Degradation je Compound aus Projekt 13 uebernehmen
  3. Out-Lap-Malus und Warm-up-Verhalten quantifizieren
  4. Simulation ueber Undercut-Fenster von 1-5 Runden
  5. Sensitivitaets-Heatmap: Deg x Pitloss

GENUTZTE FASTF1-BAUSTEINE
  - Laps PitInTime/PitOutTime
  - TyreLife
  - Laps.pick_box_laps
  - eigene Simulation

AUSBAUSTUFE
Erweitere um Verkehr: modelliere, wie viele Sekunden der Undercut kostet, wenn du hinter einem langsameren Auto herauskommst.
"""

import numpy as np
import pandas as pd
import fastf1

fastf1.Cache.enable_cache("~/f1_cache")
ses = fastf1.get_session(2024, "Spain", "R")
ses.load(telemetry=False)

laps = ses.laps.copy()
laps["Sec"] = laps["LapTime"].dt.total_seconds()

# --- 1) Pitloss empirisch schaetzen ---
base = (laps.pick_wo_box().pick_accurate().pick_track_status("1")
        .groupby("Driver")["LapTime"].median().dt.total_seconds())

in_laps = laps[laps["PitInTime"].notna()]
out_laps = laps[laps["PitOutTime"].notna()]
loss_in = (in_laps["Sec"] - in_laps["Driver"].map(base)).dropna()
loss_out = (out_laps["Sec"] - out_laps["Driver"].map(base)).dropna()
PITLOSS = float(loss_in.median() + loss_out.median())
print(f"Geschaetzter Pitloss: {PITLOSS:.2f} s  "
      f"(in {loss_in.median():.2f} / out {loss_out.median():.2f})")


def undercut_gain(deg_alt, deg_neu, pitloss, out_lap_malus=0.6, n_laps=3):
    """Zeitgewinn (positiv = Undercut lohnt) fuer n Runden frueher stoppen."""
    gain = 0.0
    for i in range(n_laps):
        # Verfolger: neue Reifen, Alter i
        t_new = deg_neu * i + (out_lap_malus if i == 0 else 0.0)
        # Vorausfahrender: alte Reifen, altern weiter
        t_old = deg_alt * (i + 1)
        gain += t_old - t_new
    return gain - 0.0   # Pitloss faellt fuer beide an, kuerzt sich raus


DEG_ALT, DEG_NEU = 0.09, 0.05     # s/Runde, aus Projekt 13
for n in range(1, 6):
    g = undercut_gain(DEG_ALT, DEG_NEU, PITLOSS, n_laps=n)
    print(f"{n} Runden frueher stoppen -> {g:+.2f} s Vorteil")

# --- Sensitivitaet ---
degs = np.round(np.arange(0.04, 0.16, 0.02), 3)
outs = np.round(np.arange(0.0, 1.6, 0.3), 2)
grid = pd.DataFrame(
    [[round(undercut_gain(d, DEG_NEU, PITLOSS, o, 3), 2) for o in outs]
     for d in degs],
    index=[f"deg={d}" for d in degs],
    columns=[f"out+{o}" for o in outs],
)
print("\nUndercut-Gewinn [s] bei 3 Runden:")
print(grid.to_string())
