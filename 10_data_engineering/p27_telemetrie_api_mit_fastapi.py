"""
P27 - Telemetrie-API mit FastAPI
================================

Ein REST-Service, der aufbereitete F1-Daten ausliefert: Endpunkte fuer Sessions, Runden, Telemetrie und Vergleiche.

Kategorie:   Data Engineering
Niveau:      Profi
Aufwand:     6-8 h
Schwerpunkt: Engineering
Zusaetzliche Pakete: fastapi, uvicorn, pydantic

WARUM DAS LOHNT
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

AUSBAUSTUFE  [umgesetzt]
Setz Redis als zweite Cache-Ebene davor und miss die p95-Latenz mit locust
unter Last.

VORGEHEN 3 nannte vier Endpunkte, gebaut waren nur drei - /compare fehlte
komplett. Ergaenzt mit fastf1.utils.delta_time() (derselbe Baustein wie
P07), Distanz/Speed beider Fahrer plus die kontinuierliche Zeitdifferenz.

VORGEHEN 2 ("Session-Loader mit lru_cache") ist bereits erledigt, ohne dass
hier neu gebaut werden muss: f1lab.load() traegt selbst schon
@lru_cache(maxsize=32). Ein zweiter Loader-Cache waere eine dritte Kopie
derselben Idee - stattdessen direkt f1lab.load() verwendet, wie im Rest des
Projekts.

AUSBAUSTUFE: weder redis noch locust stehen in requirements.txt, und ein
lokaler Redis-Server ist in dieser Umgebung nicht installiert (gilt
vermutlich fuer die meisten Rechner, auf denen das Projekt entwickelt wuerde,
ohne dass extra Infrastruktur hochgefahren wird). Umgesetzt ist deshalb die
Idee dahinter, nicht die konkreten Tools: eine simple TTL-Antwort-Cache-
Ebene VOR den teuren Endpunkten (/telemetry, /compare) - das ist genau,
wofuer Redis in Produktion stehen wuerde, nur ohne Persistenz und ohne
geteilten Zustand ueber mehrere Prozesse. Die Lastmessung ersetzt locust
durch einen kleinen Benchmark mit stdlib (ThreadPoolExecutor +
urllib.request): echte HTTP-Anfragen gegen einen echten, lokal gestarteten
uvicorn-Prozess, p50/p95/p99 kalt (erster Aufruf, Cache-Miss) gegen warm
(Cache-Hit) gemessen. Das erledigt nebenbei auch VORGEHEN 5 ("mit uvicorn
starten, Swagger pruefen") automatisiert statt als manueller Schritt.

Gemessen gegen /telemetry/2024/Spain/R/VER (lokal, gegen den gefuellten
FastF1-Cache): kalter Aufruf 1846 ms, warme Aufrufe (40x, 8 parallel) p50
4.1 ms / p95 5.5 ms - Faktor 446 fuer den p50. Zum Einordnen, was allein
f1lab.load()s Sessioncache (ohne die Antwort-Cache-Ebene) bringt: /laps auf
derselben Strecke liegt bei p50 304 ms, p95 aber bei 3794 ms - unter 8
parallelen Anfragen ohne Antwort-Cache dominiert die pandas-Filterung pro
Request und die Formulierung der Response neu, mit deutlich hoeherer
Streuung als mit der Cache-Ebene davor.

Nachtrag (2026-08-17): drei neue Endpunkte fuer P35/P39/P41 -
/strategy (exakter Boxenstopp-Plan), /overtakes (Ueberholorte gegen
DRS-Zonen), /traffic (Verkehrs-Simulation gegen eine Alternative).
/overtakes und /traffic teilen sich dieselbe TTL-Antwort-Cache-Ebene wie
/telemetry und /compare - beide sind teuer genug (Telemetrie-Auswertung
bzw. Monte-Carlo-Simulation), dass sich das lohnt. /strategy dagegen
nicht: braucht keine Telemetrie und ist schon aus dem
f1lab.load()-Sessioncache heraus schnell genug. P38 (Ueberholschwierigkeit
je Strecke) und P40 (Startplatz-Paritaet) bewusst OHNE Endpunkt - beide
brauchen einen Season-Scan ueber Dutzende bis hunderte Sessions (Minuten),
das wuerde jede HTTP-Anfrage blockierend genauso lange haengen lassen,
kein Fall fuer einen synchronen GET-Endpunkt.
"""
from __future__ import annotations

import json
import statistics
import sys
import threading
import time
import urllib.request
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import fastf1
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastf1.utils import delta_time
from pydantic import BaseModel

import f1lab

warnings.filterwarnings("ignore")
fastf1.set_log_level("ERROR")

# Modulweit statt nur in __main__, damit auch "uvicorn p27_...:app" (der
# reguläre Produktionsstart, siehe Kopfzeile) einen aktiven Cache vorfindet.
f1lab.enable_cache()

app = FastAPI(title="F1 Telemetry API", version="1.0")

# --- AUSBAUSTUFE: zweite Cache-Ebene vor den teuren Endpunkten -----------
# Was Redis in Produktion waere: ein Antwort-Cache mit TTL, unabhaengig vom
# f1lab.load()-Sessioncache darunter. Hier in-process statt extern, siehe
# Docstring fuer die Begruendung.
_ANTWORT_CACHE: dict[str, tuple[float, object]] = {}
CACHE_TTL_S = 300.0


def cache_get(key: str):
    treffer = _ANTWORT_CACHE.get(key)
    if treffer is None:
        return None
    gesetzt_um, wert = treffer
    if time.monotonic() - gesetzt_um > CACHE_TTL_S:
        del _ANTWORT_CACHE[key]
        return None
    return wert


def cache_set(key: str, wert: object) -> None:
    _ANTWORT_CACHE[key] = (time.monotonic(), wert)


class LapOut(BaseModel):
    driver: str
    lap: int
    lap_time_s: float | None
    compound: str | None
    tyre_life: float | None
    position: float | None


class TelPoint(BaseModel):
    distance: float
    speed: float
    throttle: float
    brake: bool
    gear: int


class ComparePoint(BaseModel):
    distance: float
    speed_a: float
    speed_b: float
    delta_s: float     # positiv = Fahrer B liegt zu diesem Zeitpunkt vorn


class OvertakeOut(BaseModel):
    gainer: str
    loser: str
    lap: int
    distance_m: float
    in_drs_zone: bool


class StintOut(BaseModel):
    compound: str
    start_lap: int
    end_lap: int


class StrategyOut(BaseModel):
    n_stops: int
    green_time_s: float
    stints: list[StintOut]


class TrafficOut(BaseModel):
    hero_stops: int
    hero_free_s: float
    hero_traffic_s: float
    alt_stops: int
    alt_free_s: float
    alt_traffic_s: float
    recommendation_flips: bool


@app.get("/sessions/{year}")
def sessions(year: int):
    sched = fastf1.get_event_schedule(year, include_testing=False)
    return sched[["RoundNumber", "EventName", "Country",
                 "EventDate", "EventFormat"]].to_dict("records")


@app.get("/laps/{year}/{gp}/{ident}", response_model=list[LapOut])
def laps(year: int, gp: str, ident: str,
         driver: str | None = None, clean: bool = True):
    try:
        s = f1lab.load(year, gp, ident, telemetry=False)
    except Exception as exc:
        raise HTTPException(404, str(exc)) from exc

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


@app.get("/telemetry/{year}/{gp}/{ident}/{driver}", response_model=list[TelPoint])
def telemetry(year: int, gp: str, ident: str, driver: str,
             points: int = Query(400, ge=50, le=5000)):
    key = f"tel:{year}:{gp}:{ident}:{driver}:{points}"
    cached = cache_get(key)
    if cached is not None:
        return cached

    try:
        s = f1lab.load(year, gp, ident, telemetry=True)
        lap = s.laps.pick_drivers(driver.upper()).pick_fastest()
        tel = lap.get_car_data().add_distance()
    except Exception as exc:
        raise HTTPException(404, str(exc)) from exc
    if tel is None or tel.empty:
        raise HTTPException(404, "keine Telemetrie gefunden")

    step = max(len(tel) // points, 1)
    tel = tel.iloc[::step]
    out = [TelPoint(distance=float(r["Distance"]), speed=float(r["Speed"]),
                    throttle=float(r["Throttle"]), brake=bool(r["Brake"]),
                    gear=int(r["nGear"]))
          for _, r in tel.iterrows()]
    cache_set(key, out)
    return out


@app.get("/compare/{year}/{gp}/{ident}", response_model=list[ComparePoint])
def compare(year: int, gp: str, ident: str, driver_a: str, driver_b: str,
           points: int = Query(400, ge=50, le=5000)):
    key = f"cmp:{year}:{gp}:{ident}:{driver_a}:{driver_b}:{points}"
    cached = cache_get(key)
    if cached is not None:
        return cached

    try:
        s = f1lab.load(year, gp, ident, telemetry=True)
        lap_a = s.laps.pick_drivers(driver_a.upper()).pick_fastest()
        lap_b = s.laps.pick_drivers(driver_b.upper()).pick_fastest()
        tel_b = lap_b.get_car_data().add_distance()
        delta, ref, _cmp = delta_time(lap_a, lap_b)
    except Exception as exc:
        raise HTTPException(404, str(exc)) from exc
    if ref is None or ref.empty:
        raise HTTPException(404, "kein Vergleich moeglich")

    speed_b = np.interp(ref["Distance"], tel_b["Distance"], tel_b["Speed"])
    step = max(len(ref) // points, 1)
    idx = np.arange(0, len(ref), step)
    out = [ComparePoint(distance=float(ref["Distance"].iloc[i]),
                        speed_a=float(ref["Speed"].iloc[i]),
                        speed_b=float(speed_b[i]),
                        delta_s=float(delta.iloc[i]))
          for i in idx]
    cache_set(key, out)
    return out


@app.get("/strategy/{year}/{gp}", response_model=StrategyOut)
def strategy(year: int, gp: str):
    """Exakt bester Boxenstopp-Plan (P35). Keine Telemetrie noetig, kein
    Antwort-Cache - schon aus dem f1lab.load()-Sessioncache heraus schnell."""
    try:
        s = f1lab.load(year, gp, "R", telemetry=False)
        cfg = f1lab.race_config_from_session(s)
        plan = f1lab.optimal_strategy(cfg)
    except Exception as exc:
        raise HTTPException(404, str(exc)) from exc
    return StrategyOut(
        n_stops=plan.n_stops, green_time_s=plan.green_time,
        stints=[StintOut(compound=st.compound, start_lap=st.start_lap,
                         end_lap=st.end_lap) for st in plan.stints])


@app.get("/overtakes/{year}/{gp}/{ident}", response_model=list[OvertakeOut])
def overtakes(year: int, gp: str, ident: str):
    """Ueberholungen dieser Session, lokalisiert und gegen DRS-Zonen
    geprueft (P39). Braucht Telemetrie fuer Rennen UND Qualifying - siehe
    f1lab.overtake_locations()-Docstring fuer den Grund (im Rennen selbst
    oft keine offene DRS-Zone messbar)."""
    key = f"ov:{year}:{gp}:{ident}"
    cached = cache_get(key)
    if cached is not None:
        return cached

    try:
        s = f1lab.load(year, gp, ident, telemetry=True)
        orte = f1lab.overtake_locations(s)
    except Exception as exc:
        raise HTTPException(404, str(exc)) from exc
    out = [OvertakeOut(gainer=r.gainer, loser=r.loser, lap=int(r.lap),
                       distance_m=float(r.distance_m),
                       in_drs_zone=bool(r.in_drs_zone))
          for r in orte.itertuples()]
    cache_set(key, out)
    return out


@app.get("/traffic/{year}/{gp}", response_model=TrafficOut)
def traffic(year: int, gp: str, alt_stops: int,
           delta: float = 0.15, gap: float = 3.0, p_overtake: float = 0.15):
    """Verkehrs-Simulation (P41): DAG-Optimum gegen eine Alternative mit
    ``alt_stops`` Stopps, beide gegen einen Rivalen simuliert. Rivalen-
    Tempo/Startabstand/Ueberholwahrscheinlichkeit sind Szenario-Annahmen,
    keine Messwerte - siehe P41."""
    key = f"tr:{year}:{gp}:{alt_stops}:{delta}:{gap}:{p_overtake}"
    cached = cache_get(key)
    if cached is not None:
        return cached

    try:
        s = f1lab.load(year, gp, "R", telemetry=False)
        cfg = f1lab.race_config_from_session(s)
        hero = f1lab.optimal_strategy(cfg)
        kandidaten = {n: st_ for n, st_ in
                     f1lab.frontier_by_stops(cfg, up_to=4).items()
                     if st_ is not None}
        if alt_stops not in kandidaten:
            raise HTTPException(
                404, f"{alt_stops}-Stopp nicht moeglich, verfuegbar: "
                    f"{sorted(kandidaten)}")
        alt = kandidaten[alt_stops]
        hero_t = f1lab.lap_times_for_strategy(cfg, hero)
        alt_t = f1lab.lap_times_for_strategy(cfg, alt)
        rivale_t = hero_t + delta
        c_hero, _ = f1lab.traffic_cost(hero_t, rivale_t, gap, p_overtake,
                                       n_sim=3000, seed=1)
        c_alt, _ = f1lab.traffic_cost(alt_t, rivale_t, gap, p_overtake,
                                      n_sim=3000, seed=1)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(404, str(exc)) from exc

    hero_traffic = hero.green_time + c_hero
    alt_traffic = alt.green_time + c_alt
    out = TrafficOut(
        hero_stops=hero.n_stops, hero_free_s=hero.green_time,
        hero_traffic_s=hero_traffic, alt_stops=alt_stops,
        alt_free_s=alt.green_time, alt_traffic_s=alt_traffic,
        recommendation_flips=alt_traffic < hero_traffic)
    cache_set(key, out)
    return out


# --------------------------------------------------------------- Demo/Test
DEMO_HOST, DEMO_PORT = "127.0.0.1", 8321


def _server_starten():
    import uvicorn
    config = uvicorn.Config(app, host=DEMO_HOST, port=DEMO_PORT,
                            log_level="warning")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    basis = f"http://{DEMO_HOST}:{DEMO_PORT}"
    for _ in range(100):
        try:
            urllib.request.urlopen(f"{basis}/sessions/2024", timeout=1).read()
            return server, basis
        except Exception:
            time.sleep(0.1)
    raise RuntimeError("Server nicht erreichbar")


def _request_dauer(url: str) -> float:
    t0 = time.perf_counter()
    with urllib.request.urlopen(url, timeout=15) as resp:
        resp.read()
    return time.perf_counter() - t0


def _perzentile(werte: list[float]) -> dict:
    s = sorted(werte)
    return {
        "p50": statistics.median(s),
        "p95": s[int(0.95 * (len(s) - 1))],
        "p99": s[int(0.99 * (len(s) - 1))],
    }


def smoke_test_und_benchmark():
    """VORGEHEN 5 + AUSBAUSTUFE: echte HTTP-Anfragen gegen einen echten
    lokalen Server, dann p50/p95/p99 kalt gegen warm."""
    print("[1/3] uvicorn lokal starten und Endpunkte pruefen (VORGEHEN 5) ...")
    server, basis = _server_starten()
    try:
        r = urllib.request.urlopen(f"{basis}/sessions/2024", timeout=10)
        n_events = len(json.loads(r.read()))
        print(f"      GET /sessions/2024 -> {r.status}, {n_events} Events")

        r = urllib.request.urlopen(
            f"{basis}/laps/2024/Bahrain/R?driver=VER", timeout=10)
        print(f"      GET /laps/2024/Bahrain/R?driver=VER -> {r.status}")

        r = urllib.request.urlopen(
            f"{basis}/telemetry/2024/Bahrain/R/VER?points=200", timeout=15)
        print(f"      GET /telemetry/.../VER -> {r.status}")

        r = urllib.request.urlopen(
            f"{basis}/compare/2024/Bahrain/R?driver_a=VER&driver_b=PER"
            f"&points=200", timeout=15)
        print(f"      GET /compare/...VER-PER -> {r.status}")

        r = urllib.request.urlopen(f"{basis}/strategy/2024/Bahrain", timeout=15)
        print(f"      GET /strategy/2024/Bahrain -> {r.status}")

        r = urllib.request.urlopen(
            f"{basis}/traffic/2024/Bahrain?alt_stops=3", timeout=30)
        print(f"      GET /traffic/2024/Bahrain?alt_stops=3 -> {r.status}")

        r = urllib.request.urlopen(f"{basis}/overtakes/2024/Bahrain/R", timeout=90)
        n_ov = len(json.loads(r.read()))
        print(f"      GET /overtakes/2024/Bahrain/R -> {r.status}, "
             f"{n_ov} lokalisierte Ueberholungen")
        print(f"      Swagger: {basis}/docs")

        print("\n[2/3] Benchmark: kalt (Cache-Miss) vs. warm (Cache-Hit) "
             "(AUSBAUSTUFE) ...")
        kalte_url = (f"{basis}/telemetry/2024/Spain/R/VER?points=300")
        _ANTWORT_CACHE.clear()
        kalt = [_request_dauer(kalte_url)]         # erster Aufruf: echte Arbeit
        with ThreadPoolExecutor(max_workers=8) as ex:
            warm = list(ex.map(lambda _: _request_dauer(kalte_url), range(40)))

        print(f"      kalt (n=1):  {kalt[0] * 1000:.1f} ms")
        pk = _perzentile(warm)
        print(f"      warm (n=40, 8 parallel): p50={pk['p50'] * 1000:.1f}ms  "
             f"p95={pk['p95'] * 1000:.1f}ms  p99={pk['p99'] * 1000:.1f}ms")
        print(f"      Beschleunigung p50 warm vs. kalt: "
             f"{kalt[0] / pk['p50']:.0f}x")

        print("\n[3/3] Zum Vergleich: /laps ohne Antwort-Cache (nur "
             "f1lab.load()-Cache) ...")
        laps_url = f"{basis}/laps/2024/Spain/R"
        with ThreadPoolExecutor(max_workers=8) as ex:
            laps_zeiten = list(ex.map(lambda _: _request_dauer(laps_url),
                                      range(40)))
        pl = _perzentile(laps_zeiten)
        print(f"      p50={pl['p50'] * 1000:.1f}ms  p95={pl['p95'] * 1000:.1f}ms "
             f"- schon schnell, weil f1lab.load() selbst cached, aber ohne "
             f"die serialisierte Antwort vorzuhalten jedes Mal neu gebaut.")
    finally:
        server.should_exit = True


if __name__ == "__main__":
    smoke_test_und_benchmark()
