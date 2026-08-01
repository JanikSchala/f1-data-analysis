"""
P02 - Saison-Kalender & Event-Metadaten als Datenbank
=====================================================

Alle Rennwochenenden seit 1950 in eine saubere Tabelle - inkl. Sprint-Formaten und Session-Zeiten.

Kategorie:   Grundlagen & Datenzugriff
Niveau:      Einsteiger
Aufwand:     2 h
Schwerpunkt: Engineering, DS

WARUM DAS LOHNT
Jedes groessere F1-Datenprodukt braucht eine Dimensionstabelle fuer Events. Zeigt Verstaendnis fuer Datenmodellierung statt nur Ad-hoc-Skripten.

VORGEHEN
  1. Schedule fuer mehrere Jahre laden und konkatenieren
  2. Sprint- vs. konventionelles Format ableiten (Session2Date-Muster)
  3. Session-Zeiten in UTC und lokal ausgeben
  4. Als Parquet-Dimensionstabelle speichern

GENUTZTE FASTF1-BAUSTEINE
  - fastf1.get_event_schedule
  - fastf1.get_events_remaining
  - Event.get_session_date
  - EventSchedule

AUSBAUSTUFE
Ergaenze Streckenlaenge, Anzahl Kurven und Hoehenmeter pro Circuit aus get_circuit_info() und mache daraus eine echte Circuit-Dimension.
"""

import pandas as pd
import fastf1

fastf1.Cache.enable_cache("~/f1_cache")

frames = []
for year in range(2018, 2025):
    sched = fastf1.get_event_schedule(year, include_testing=False)
    sched["Season"] = year
    frames.append(sched)

cal = pd.concat(frames, ignore_index=True)

# Sprint-Wochenenden erkennen
cal["IsSprint"] = cal["EventFormat"].str.contains("sprint", case=False, na=False)

cols = ["Season", "RoundNumber", "EventName", "Country", "Location",
        "EventDate", "EventFormat", "IsSprint"]
print(cal[cols].tail(15).to_string(index=False))

print("\n".join([
    f"Events gesamt:   {len(cal)}",
    f"Sprint-Events:   {int(cal['IsSprint'].sum())}",
    f"Formate:         {sorted(cal['EventFormat'].unique())}",
]))

# Session-Zeiten eines Events
ev = fastf1.get_event(2024, "Brazil")
for i in range(1, 6):
    name = ev.get(f"Session{i}")
    date = ev.get(f"Session{i}DateUtc")
    if isinstance(name, str) and name:
        print(f"{name:<12} {date} UTC")

cal.to_parquet("dim_events.parquet", index=False)
