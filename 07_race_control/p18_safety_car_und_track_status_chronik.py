"""
P18 - Safety-Car- und Track-Status-Chronik
==========================================

Alle gelben Phasen, VSC- und SC-Perioden exakt als Rundenintervalle - inklusive Auswirkung auf die Feldstreckung.

Kategorie:   Race Control & Regeln
Niveau:      Fortgeschritten
Aufwand:     3 h
Rollen:      STRAT, DS

WARUM DAS ZAEHLT
Safety Cars entscheiden Rennen. Wer sie automatisch aus track_status extrahiert und deren Effekt beziffert, liefert Strategie-Input statt nur Deskription.

VORGEHEN
  1. track_status decodieren (1 gruen, 2 gelb, 4 SC, 5 rot, 6/7 VSC)
  2. Aufeinanderfolgende gleiche Status zu Intervallen zusammenfassen
  3. Intervalle auf Rundennummern mappen
  4. Feldstreckung (Abstand P1 zu Letztem) vor/nach SC vergleichen

GENUTZTE FASTF1-BAUSTEINE
  - Session.track_status
  - Session.session_status
  - Laps.pick_track_status
  - Laps TrackStatus

AUSBAUSTUFE
Berechne pro Fahrer, ob das Safety Car ihm Zeit geschenkt oder gekostet hat - abhaengig davon, ob er direkt davor gestoppt hatte.
"""

import pandas as pd
import fastf1

fastf1.Cache.enable_cache("~/f1_cache")
ses = fastf1.get_session(2024, "Canada", "R")
ses.load(telemetry=False)

CODES = {"1": "Gruen", "2": "Gelb", "3": "?", "4": "Safety Car",
         "5": "Rote Flagge", "6": "VSC", "7": "VSC endet"}

ts = ses.track_status.copy()
ts["Label"] = ts["Status"].map(CODES).fillna(ts["Status"])
ts["grp"] = (ts["Status"] != ts["Status"].shift()).cumsum()

phases = (ts.groupby(["grp", "Label"])
          .agg(Von=("Time", "min"), Bis=("Time", "max"))
          .reset_index().drop(columns="grp"))
phases["Dauer_s"] = (phases["Bis"] - phases["Von"]).dt.total_seconds()

# Rundennummern zuordnen
laps = ses.laps
leader = laps[laps["Position"] == 1][["LapNumber", "LapStartTime"]].dropna()


def lap_at(t):
    m = leader[leader["LapStartTime"] <= t]
    return int(m["LapNumber"].max()) if len(m) else 0


phases["RundeVon"] = phases["Von"].apply(lap_at)
phases["RundeBis"] = phases["Bis"].apply(lap_at)

print(phases[phases["Label"] != "Gruen"].to_string(index=False))

# Feldstreckung je Runde
spread = (laps.dropna(subset=["Position"])
          .groupby("LapNumber")["Time"]
          .agg(lambda s: (s.max() - s.min()).total_seconds()))
sc_laps = set()
for _, p in phases[phases["Label"].isin(["Safety Car", "VSC"])].iterrows():
    sc_laps.update(range(p["RundeVon"], p["RundeBis"] + 1))

print("\nMittlere Feldstreckung:")
print(f"  unter Gruen: {spread[~spread.index.isin(sc_laps)].mean():.1f} s")
print(f"  unter SC/VSC: {spread[spread.index.isin(sc_laps)].mean():.1f} s")
