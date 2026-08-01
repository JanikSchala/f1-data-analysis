"""
P01 - Session-Explorer: Jede Session der F1-Historie laden
==========================================================

Das Fundament. Cache aufsetzen, beliebige Session laden, verstehen was load() eigentlich holt.

Kategorie:   Grundlagen & Datenzugriff
Niveau:      Einsteiger
Aufwand:     1-2 h
Schwerpunkt: Datenanalyse, Engineering

WARUM DAS LOHNT
Ohne sauberes Caching wartest du bei jeder Analyse Minuten. Und wer den Datenfluss einmal durchschaut hat - Live-Timing-API, Parser, DataFrame - weiss bei jedem spaeteren Problem, wo er suchen muss.

VORGEHEN
  1. Cache-Ordner anlegen und global aktivieren
  2. Session ueber (Jahr, GP, Identifier) laden - 'R', 'Q', 'FP1', 'S', 'SQ'
  3. Selektiv laden: laps/telemetry/weather/messages einzeln steuern
  4. session_info, drivers und results inspizieren
  5. Ladezeiten mit und ohne Cache messen

GENUTZTE FASTF1-BAUSTEINE
  - fastf1.Cache.enable_cache
  - fastf1.get_session
  - Session.load
  - Session.session_info

AUSBAUSTUFE
Baue eine Funktion, die eine ganze Saison in einem Rutsch in den Cache zieht und den Fortschritt in der Konsole anzeigt.
"""

import time
from pathlib import Path
import fastf1

CACHE = Path.home() / "f1_cache"
CACHE.mkdir(exist_ok=True)
fastf1.Cache.enable_cache(str(CACHE))

def load_session(year, gp, ident, **kw):
    t0 = time.perf_counter()
    ses = fastf1.get_session(year, gp, ident)
    ses.load(**kw)
    print(f"{year} {gp} {ident} geladen in {time.perf_counter()-t0:.1f}s")
    return ses

# Volles Rennen inkl. Telemetrie
race = load_session(2024, "Monza", "R")

print("Event:      ", race.event["EventName"])
print("Datum:      ", race.event["EventDate"].date())
print("Runden:     ", race.total_laps)
print("Fahrer:     ", len(race.drivers))
print("Lap-Zeilen: ", len(race.laps))

# Nur Timing, ohne Telemetrie -> deutlich schneller
quali = load_session(2024, "Monza", "Q",
                     telemetry=False, weather=False, messages=False)

print(quali.results[["Abbreviation", "TeamName", "Position", "Q1", "Q2", "Q3"]])
