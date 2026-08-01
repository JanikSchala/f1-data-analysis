"""
P27 - Telemetrie-API mit FastAPI
================================

Ein REST-Service, der aufbereitete F1-Daten ausliefert: Endpunkte fuer Sessions, Runden, Telemetrie und Vergleiche.

Kategorie:   Data Engineering
Niveau:      Profi
Aufwand:     6-8 h
Rollen:      ENG
Zusaetzliche Pakete: fastapi, uvicorn, pydantic

WARUM DAS ZAEHLT
Zeigt, dass du nicht nur Notebooks schreibst, sondern Datenprodukte baust. Mit Pydantic-Modellen, Caching und OpenAPI-Doku ist es sofort vorzeigbar.

VORGEHEN
  1. Pydantic-Response-Modelle definieren
  2. Session-Loader mit lru_cache, damit nicht jede Anfrage neu laedt
  3. Endpunkte: /sessions, /laps, /telemetry, /compare
  4. Downsampling der Telemetrie fuer die Antwortgroesse
  5. Mit uvicorn starten, Swagger unter /docs pruefen

GENUTZTE FASTF1-BAUSTEINE
  - fastf1 Session/Laps/Telemetry
  - FastAPI
  - pydantic
  - functools.lru_cache

AUSBAUSTUFE
Setz Redis als zweite Cache-Ebene davor und miss die p95-Latenz mit locust unter Last.
"""

# uvicorn f1_api:app --reload
from functools import lru_cache
from typing import List, Optional

import fastf1
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

fastf1.Cache.enable_cache("~/f1_cache")
app = FastAPI(title="F1 Telemetry API", version="1.0")


class LapOut(BaseModel):
    driver: str
    lap: int
    lap_time_s: Optional[float]
    compound: Optional[str]
    tyre_life: Optional[float]
    position: Optional[float]


class TelPoint(BaseModel):
    distance: float
    speed: float
    throttle: float
    brake: bool
    gear: int


@lru_cache(maxsize=16)
def get_loaded(year: int, gp: str, ident: str, with_tel: bool):
    s = fastf1.get_session(year, gp, ident)
    s.load(telemetry=with_tel, weather=False, messages=False)
    return s


@app.get("/sessions/{year}")
def sessions(year: int):
    sched = fastf1.get_event_schedule(year, include_testing=False)
    return sched[["RoundNumber", "EventName", "Country",
                  "EventDate", "EventFormat"]].to_dict("records")


@app.get("/laps/{year}/{gp}/{ident}", response_model=List[LapOut])
def laps(year: int, gp: str, ident: str,
         driver: Optional[str] = None, clean: bool = True):
    try:
        s = get_loaded(year, gp, ident, False)
    except Exception as exc:
        raise HTTPException(404, str(exc))

    df = s.laps
    if driver:
        df = df.pick_drivers(driver.upper())
    if clean:
        df = df.pick_accurate().pick_wo_box()
    if df.empty:
        raise HTTPException(404, "keine Runden gefunden")

    return [LapOut(driver=r["Driver"], lap=int(r["LapNumber"]),
                   lap_time_s=(r["LapTime"].total_seconds()
                               if pd.notna(r["LapTime"]) else None),
                   compound=r["Compound"], tyre_life=r["TyreLife"],
                   position=r["Position"])
            for _, r in df.iterrows()]


@app.get("/telemetry/{year}/{gp}/{ident}/{driver}", response_model=List[TelPoint])
def telemetry(year: int, gp: str, ident: str, driver: str,
              points: int = Query(400, ge=50, le=5000)):
    s = get_loaded(year, gp, ident, True)
    lap = s.laps.pick_drivers(driver.upper()).pick_fastest()
    tel = lap.get_car_data().add_distance()
    step = max(len(tel) // points, 1)
    tel = tel.iloc[::step]
    return [TelPoint(distance=float(r["Distance"]), speed=float(r["Speed"]),
                     throttle=float(r["Throttle"]), brake=bool(r["Brake"]),
                     gear=int(r["nGear"]))
            for _, r in tel.iterrows()]
