"""
P30 - Live-Timing aufzeichnen und in Echtzeit auswerten
=======================================================

Waehrend einer laufenden Session den Live-Timing-Stream mitschneiden und anschliessend wie eine normale Session laden.

Kategorie:   Live Timing
Niveau:      Profi
Aufwand:     6-8 h
Rollen:      ENG, STRAT

WARUM DAS ZAEHLT
Echtzeitdatenverarbeitung ist selten im Portfolio und technisch anspruchsvoll. Wer das gemacht hat, hat den F1-Datenstack wirklich verstanden.

VORGEHEN
  1. Recorder per CLI starten: python -m fastf1.livetiming save output.txt
  2. Alternativ SignalRClient programmatisch mit asyncio starten
  3. Aufzeichnung in LiveTimingData laden
  4. Session mit livedata=... laden und normal analysieren
  5. Rolling-Auswertung: Pace der letzten 5 Runden je Fahrer

GENUTZTE FASTF1-BAUSTEINE
  - fastf1.livetiming.client.SignalRClient
  - fastf1.livetiming.data.LiveTimingData
  - Session.load(livedata=...)

AUSBAUSTUFE
Schreibe die Live-Daten in eine Zeitreihen-DB (InfluxDB oder TimescaleDB) und baue ein Grafana-Board, das waehrend des Rennens aktualisiert.
"""

# --- Aufzeichnung (waehrend der Session laufen lassen) ---
# CLI:  python -m fastf1.livetiming save saved_data.txt
#
# Oder programmatisch:
import asyncio
from fastf1.livetiming.client import SignalRClient


def record(path="saved_data.txt", timeout_min=120):
    client = SignalRClient(filename=path, timeout=timeout_min * 60,
                           debug=False)
    client.start()


# --- Auswertung nach der Session ---
import pandas as pd
import fastf1
from fastf1.livetiming.data import LiveTimingData

fastf1.Cache.enable_cache("~/f1_cache")


def analyse(path, year, gp, ident):
    live = LiveTimingData(path)
    live.load()

    ses = fastf1.get_session(year, gp, ident)
    ses.load(livedata=live, telemetry=False, weather=False)

    laps = ses.laps.copy()
    laps["Sec"] = laps["LapTime"].dt.total_seconds()

    # Rolling Pace der letzten 5 Runden je Fahrer
    laps = laps.sort_values(["Driver", "LapNumber"])
    laps["Rolling5"] = (laps.groupby("Driver")["Sec"]
                        .transform(lambda s: s.rolling(5, min_periods=3).median()))

    latest = (laps.dropna(subset=["Rolling5"])
              .sort_values("LapNumber").groupby("Driver").tail(1)
              .sort_values("Rolling5"))

    print("Aktuelle Pace (Median letzte 5 Runden):")
    print(latest[["Driver", "LapNumber", "Compound",
                  "TyreLife", "Rolling5"]].round(3).to_string(index=False))
    return laps


if __name__ == "__main__":
    # record()                      # waehrend der Session
    analyse("saved_data.txt", 2024, "Monza", "R")
