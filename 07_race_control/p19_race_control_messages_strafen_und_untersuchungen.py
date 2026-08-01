"""
P19 - Race Control Messages: Strafen und Untersuchungen automatisch auswerten
=============================================================================

Die Textmeldungen der Rennleitung parsen: Strafen, Verwarnungen, Track-Limits, gestrichene Runden.

Kategorie:   Race Control & Regeln
Niveau:      Fortgeschritten
Aufwand:     3-4 h
Schwerpunkt: Datenanalyse, Engineering

WARUM DAS LOHNT
Textparsing plus Regex auf einem echten, unsauberen Feed. Zeigt, dass du auch mit unstrukturierten Daten arbeitest, nicht nur mit fertigen DataFrames.

VORGEHEN
  1. race_control_messages laden und Kategorien inspizieren
  2. Regex fuer Fahrer-Nummer, Vergehen und Strafmass bauen
  3. Track-Limit-Verstoesse je Fahrer und Kurve zaehlen
  4. Mit Deleted/DeletedReason aus den Laps gegenpruefen

GENUTZTE FASTF1-BAUSTEINE
  - Session.race_control_messages
  - Laps Deleted/DeletedReason
  - re

AUSBAUSTUFE
Baue daraus einen Saison-Report: welche Strecke produziert die meisten Track-Limit-Verstoesse, und an welcher Kurve genau.
"""

import re
import pandas as pd
import fastf1

fastf1.Cache.enable_cache("~/f1_cache")
ses = fastf1.get_session(2024, "Austria", "R")
ses.load(telemetry=False)

rcm = ses.race_control_messages.copy()
print(rcm["Category"].value_counts().to_string())

PENALTY = re.compile(r"CAR (\d+) \(([A-Z]{3})\).*?(\d+ SECOND|DRIVE THROUGH|"
                     r"STOP AND GO|REPRIMAND|GRID PENALTY)", re.I)
TRACKLIM = re.compile(r"CAR (\d+) \(([A-Z]{3})\).*TRACK LIMITS AT TURN (\d+)", re.I)

penalties, limits = [], []
for _, m in rcm.iterrows():
    txt = str(m["Message"])
    if (p := PENALTY.search(txt)):
        penalties.append({"Lap": m["Lap"], "Nr": p.group(1),
                          "Driver": p.group(2), "Strafe": p.group(3).upper(),
                          "Text": txt})
    if (t := TRACKLIM.search(txt)):
        limits.append({"Lap": m["Lap"], "Driver": t.group(2),
                       "Turn": int(t.group(3))})

pen = pd.DataFrame(penalties)
lim = pd.DataFrame(limits)

print("\n=== Strafen ===")
print(pen[["Lap", "Driver", "Strafe"]].to_string(index=False) if len(pen) else "keine")

if len(lim):
    print("\n=== Track Limits je Fahrer ===")
    print(lim["Driver"].value_counts().to_string())
    print("\n=== Haeufigste Kurven ===")
    print(lim["Turn"].value_counts().to_string())

deleted = ses.laps[ses.laps["Deleted"] == True]
if len(deleted):
    print("\n=== Gestrichene Runden ===")
    print(deleted[["Driver", "LapNumber", "DeletedReason"]].to_string(index=False))
