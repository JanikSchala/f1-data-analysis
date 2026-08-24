"""FastF1-anbindung: sessions laden, runden filtern, stints extrahieren.

trennt die datenbeschaffung von der rechnung in :mod:`f1lab.core`. alles hier
braucht netzzugriff beim ersten aufruf, danach den cache.
"""
from __future__ import annotations

import contextlib
import os
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import fastf1
import numpy as np
import pandas as pd

from .core import (
    FUEL_KG_PER_LAP,
    FUEL_S_PER_KG,
    Interval,
    RaceConfig,
    TyreModel,
    active_distance_zones,
    bootstrap_median,
    braking_zones,
    distance_in_any_zone,
    drs_state,
    elevation_profile,
    estimate_pit_loss,
    fit_degradation,
    fuel_correct,
    match_by_distance,
    path_length,
    status_intervals,
    track_curvature,
)

CACHE_DIR = Path.home() / "f1_cache"

# TrackStatus-codes aus dem live-timing-feed
TRACK_STATUS = {
    "1": "gruen", "2": "gelb", "4": "safety car",
    "5": "rot", "6": "vsc", "7": "vsc endet",
}

# FastF1 legt je session einen ordner mit ausgeschriebenem namen an. fuer die
# bestandsaufnahme muss der weg zurueck zur kennung gehen, die get_session()
# erwartet. sprint-quali heisst je nach jahr anders, landet aber auf "SQ".
SESSION_DIR_IDENT = {
    "Practice_1": "FP1", "Practice_2": "FP2", "Practice_3": "FP3",
    "Qualifying": "Q", "Sprint": "S", "Sprint_Qualifying": "SQ",
    "Sprint_Shootout": "SQ", "Race": "R",
}

# ein session-ordner existiert schon, sobald FastF1 ihn einmal angefasst hat.
# erst diese datei belegt, dass timing-daten wirklich drin liegen; telemetrie
# ist ein eigener, viel groesserer download und fehlt oft.
TIMING_MARKER = "_extended_timing_data.ff1pkl"
TELEMETRY_MARKER = "car_data.ff1pkl"

# "2024-09-01_Italian_Grand_Prix" -> datum + name
_EVENT_DIR = re.compile(r"^(\d{4}-\d{2}-\d{2})_(.+)$")

_active_cache: Path | None = None


def find_cache(path: Path | str | None = None) -> Path | None:
    """ersten vorhandenen cache-ordner suchen, ohne einen anzulegen.

    reihenfolge: explizites argument, umgebungsvariable ``F1_CACHE``, der
    standardpfad ``~/f1_cache``, und zuletzt ein ``f1_cache`` neben dem
    repository. der letzte fall deckt die ablage ab, in der repo und cache
    geschwister in einem projektordner sind.

    Returns:
        pfad oder None, wenn nirgends ein cache liegt. ein fehlender cache
        ist kein fehler: er ist der zustand vor dem ersten warmup.
    """
    env = os.environ.get("F1_CACHE")
    kandidaten = [
        Path(path) if path else None,
        Path(env).expanduser() if env else None,
        CACHE_DIR,
        Path(__file__).resolve().parents[1].parent / "f1_cache",
    ]
    for p in kandidaten:
        if p is not None and p.is_dir():
            return p
    return None


def enable_cache(path: Path | str | None = None,
                 offline: bool = False) -> Path:
    """cache aktivieren. ohne cache dauert jede session-ladung minuten.

    Args:
        path: zielordner. ohne angabe der standardpfad ``~/f1_cache``.
        offline: schaltet jeden netzzugriff ab. sessions, die nicht im cache
            liegen, scheitern dann sofort, statt minutenlang zu laden und ins
            rate-limit zu laufen. das ist fuer eine oberflaeche das
            gewuenschte verhalten.
    """
    global _active_cache
    p = Path(path) if path else CACHE_DIR
    p.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(str(p))
    fastf1.Cache.offline_mode(offline)
    _active_cache = p
    return p


def cache_ready() -> bool:
    """ob enable_cache() in diesem prozess schon lief.

    fuer aufrufer, die eine session laden koennen muessen, ohne eine von
    aussen bereits gesetzte cache-konfiguration stillschweigend zu
    ueberschreiben. siehe f1analyze/data.py: load_session() rief bislang
    bei JEDEM aufruf enable_cache() ohne argumente auf und hat damit einen
    von tests (tests/conftest.py) absichtlich gesetzten offline-fixture-
    cache-pfad wieder auf den standardpfad zurueckgesetzt. ein echter,
    lange unbemerkter bug: die CLI-tests liefen dadurch nie wirklich
    offline, sondern wurden nur zufaellig genug im lokal warmgelaufenen
    cache fuendig (siehe CLAUDE.md, f1analyze-nachtrag)."""
    return _active_cache is not None


def cached_sessions(cache_dir: Path | str | None = None) -> pd.DataFrame:
    """bestandsaufnahme des caches, allein aus der ordnerstruktur.

    liest den cache so, wie FastF1 ihn anlegt
    (``<cache>/<Jahr>/<Datum>_<Event>/<Datum>_<Session>/``), und braucht dafuer
    weder netz noch einen session-download. damit laesst sich eine auswahl
    anbieten, die nur zeigt, was auch wirklich auswertbar ist.

    Returns:
        ein datensatz je session mit saison, event, datum, kennung und zwei
        flags: ``timing`` fuer rundendaten, ``telemetry`` fuer den
        positions- und fahrzeugkanal. leerer rahmen, wenn kein cache da ist.
    """
    cols = ["season", "event", "event_date", "ident", "timing", "telemetry"]
    root = find_cache(cache_dir)
    if root is None:
        return pd.DataFrame(columns=cols)

    rows = []
    for year_dir in sorted(root.iterdir()):
        if not (year_dir.is_dir() and year_dir.name.isdigit()):
            continue
        for event_dir in sorted(year_dir.iterdir()):
            m = _EVENT_DIR.match(event_dir.name) if event_dir.is_dir() else None
            if not m:
                continue
            datum, event = m.group(1), m.group(2).replace("_", " ")
            for ses_dir in sorted(event_dir.iterdir()):
                sm = _EVENT_DIR.match(ses_dir.name) if ses_dir.is_dir() else None
                if not sm:
                    continue
                ident = SESSION_DIR_IDENT.get(sm.group(2))
                if ident is None:
                    continue
                rows.append({
                    "season": int(year_dir.name),
                    "event": event,
                    "event_date": datum,
                    "ident": ident,
                    "timing": (ses_dir / TIMING_MARKER).exists(),
                    "telemetry": (ses_dir / TELEMETRY_MARKER).exists(),
                })

    if not rows:
        return pd.DataFrame(columns=cols)
    df = pd.DataFrame(rows, columns=cols)
    df["event_date"] = pd.to_datetime(df["event_date"])
    return df.sort_values(["season", "event_date", "ident"], ignore_index=True)


@lru_cache(maxsize=32)
def load(year: int, gp, identifier: str = "R",
         telemetry: bool = False, weather: bool = False,
         messages: bool = False):
    """session laden und cachen.

    Args:
        year: saison, telemetrie gibt es erst ab 2018.
        gp: name ("Monza"), land ("Italy") oder rundennummer (14).
        identifier: FP1 FP2 FP3 Q S SQ R.
        telemetry: nur einschalten, wenn wirklich gebraucht. der download
            ist um ein vielfaches groesser.

    ein zuvor gesetzter cache bleibt erhalten: wer :func:`enable_cache` mit
    eigenem pfad oder ``offline=True`` aufgerufen hat, soll das hier nicht
    stillschweigend zurueckgesetzt bekommen.
    """
    if _active_cache is None:
        enable_cache()
    ses = fastf1.get_session(year, gp, identifier)
    ses.load(telemetry=telemetry, weather=weather, messages=messages)
    return ses


def not_deleted_mask(deleted) -> pd.Series:
    """true fuer runden, die NICHT von der rennleitung gestrichen wurden.

    die spalte ``Deleted`` ist ein nullable boolean. neben true und false
    steht dort None, wenn zu einer runde nichts gemeldet wurde. das ist der
    normalfall. FastF1s ``pick_not_deleted()`` invertiert die spalte
    direkt mit ``~``, und genau daran scheitert pandas bei object-dtype:

        TypeError: bad operand type for unary ~

    deswegen hier explizit: fehlende angabe heisst "nicht gestrichen".
    """
    s = pd.Series(deleted)
    if s.empty:
        return s.astype(bool)
    # .where statt .fillna: fillna auf object-dtype loest in pandas 2.2
    # eine downcasting-warnung aus, die in pandas 3 zum verhaltenswechsel
    # wird. .where ist in beiden versionen eindeutig.
    return ~s.where(s.notna(), False).astype(bool)


def clean_laps(session, threshold: float = 1.07) -> pd.DataFrame:
    """runden, auf denen sich pace-aussagen aufbauen lassen.

    entfernt in dieser reihenfolge:
      - boxenrunden (in- und out-lap sind keine renn-runden)
      - unplausible runden (FastF1 markiert sie ueber IsAccurate)
      - alles ausser gruener flagge (safety car verschiebt den median
        um mehrere zehntel)
      - von der rennleitung gestrichene runden
      - ausreisser ueber threshold mal bestzeit
    """
    laps = (session.laps
            .pick_wo_box()
            .pick_accurate()
            .pick_track_status("1"))
    if "Deleted" in laps.columns:
        laps = laps[not_deleted_mask(laps["Deleted"]).to_numpy()]
    return laps.pick_quicklaps(threshold=threshold)


# --------------------------------------------------------------- Pace
@dataclass(frozen=True)
class PaceEntry:
    driver: str
    team: str
    laps: int
    median: Interval

    @property
    def median_s(self) -> float:
        return self.median.value


def race_pace(session, threshold: float = 1.07, min_laps: int = 8,
              n_resamples: int = 1000,
              kg_per_lap: float = FUEL_KG_PER_LAP) -> list[PaceEntry]:
    """bereinigte, treibstoffkorrigierte race pace je fahrer, sortiert vom
    schnellsten.

    jeder eintrag traegt ein bootstrap-intervall. ueberlappen sich zwei
    intervalle, ist der unterschied mit diesen daten nicht belegbar. das
    ist bei benachbarten fahrern regelmaessig der fall.

    ohne :func:`fuel_correct` haengt der median zusaetzlich davon ab, *wann*
    im rennen die sauberen runden eines fahrers liegen. zwei fahrer mit
    identischem tempo koennten sonst unterschiedlich abschneiden, nur weil
    eine safety-car-phase dem einen mehr fruehe (schwere), dem anderen mehr
    spaete (leichte) runden uebrig laesst. ``kg_per_lap=0`` schaltet die
    korrektur ab (fuer den vergleich in P04).
    """
    laps = clean_laps(session, threshold).copy()
    laps["sec"] = fuel_correct(
        laps["LapTime"].dt.total_seconds(), laps["LapNumber"],
        session.total_laps, kg_per_lap=kg_per_lap)

    entries = []
    for drv, g in laps.groupby("Driver"):
        v = g["sec"].dropna().to_numpy()
        if v.size < min_laps:
            continue
        entries.append(PaceEntry(
            driver=str(drv),
            team=str(g["Team"].iloc[0]),
            laps=int(v.size),
            median=bootstrap_median(v, n_resamples=n_resamples),
        ))
    return sorted(entries, key=lambda e: e.median_s)


def pace_table(session, **kwargs) -> pd.DataFrame:
    """race_pace() als DataFrame mit delta zum schnellsten."""
    entries = race_pace(session, **kwargs)
    if not entries:
        return pd.DataFrame()
    best = entries[0].median_s
    return pd.DataFrame([{
        "driver": e.driver, "team": e.team, "laps": e.laps,
        "median_s": round(e.median_s, 3),
        "delta_s": round(e.median_s - best, 3),
        "ci_lo": round(e.median.lo - best, 3),
        "ci_hi": round(e.median.hi - best, 3),
        "ci_width": round(e.median.width, 3),
    } for e in entries])


# --------------------------------------------------------------- Stints
def stints(session) -> pd.DataFrame:
    """ein datensatz je stint: fahrer, compound, start, ende, laenge."""
    df = (session.laps
          .groupby(["Driver", "Stint", "Compound"], dropna=False)["LapNumber"]
          .agg(start="min", end="max", laps="count")
          .reset_index())
    df["stint"] = df["Stint"].astype("Int64")
    return df.drop(columns="Stint").sort_values(["Driver", "start"])


def degradation(session, threshold: float = 1.10,
                min_laps: int = 6) -> pd.DataFrame:
    """degradation je stint, mit herausgerechnetem treibstoffeffekt.

    ohne die korrektur wird die degradation systematisch unterschaetzt: das
    auto wird leichter, waehrend der reifen abbaut, und beide effekte heben
    sich teilweise auf.
    """
    laps = clean_laps(session, threshold).copy()
    laps["sec"] = laps["LapTime"].dt.total_seconds()
    laps["corrected"] = fuel_correct(
        laps["sec"], laps["LapNumber"], session.total_laps)

    rows = []
    for (drv, stint), g in laps.groupby(["Driver", "Stint"]):
        g = g.sort_values("TyreLife")
        if len(g) < min_laps:
            continue
        try:
            fit = fit_degradation(g["TyreLife"], g["corrected"])
        except ValueError:
            continue
        rows.append({
            "driver": str(drv),
            "team": str(g["Team"].iloc[0]),
            "stint": int(stint),
            "compound": str(g["Compound"].iloc[0]),
            # FreshTyre (siehe P13-erweiterung, bislang nirgends gelesen):
            # ein wiederverwendeter satz hat vorverschleiss, den TyreLife
            # allein nicht abbildet. konstant innerhalb eines stints,
            # deshalb der wert der ersten runde.
            "fresh": bool(g["FreshTyre"].iloc[0]),
            "laps": fit.n,
            "deg_s_per_lap": round(fit.slope, 4),
            "base_s": round(fit.intercept, 3),
            "r2": round(fit.r2, 3),
            "reliable": fit.is_reliable,
        })
    return pd.DataFrame(rows).sort_values("deg_s_per_lap", ignore_index=True)


def degradation_by_compound(session, **kwargs) -> pd.DataFrame:
    """mittlere degradation je reifenmischung, nur belastbare fits."""
    deg = degradation(session, **kwargs)
    if deg.empty:
        return deg
    ok = deg[deg["reliable"]]
    return (ok.groupby("compound")["deg_s_per_lap"]
            .agg(mean="mean", std="std", stints="count")
            .round(4).sort_values("mean"))


def race_config_from_session(session, fit_min_laps: int = 6,
                             optimizer_min_stint: int = 4,
                             require_two_compounds: bool = True) -> RaceConfig:
    """RaceConfig fuer den strategie-optimierer (P35) aus echter degradation
    (P13) und echtem pitloss dieser session, statt parameter von hand zu
    setzen.

    je mischung: median von steigung und achsenabschnitt ueber alle
    belastbaren stint-fits (``DegradationFit.is_reliable``), ``max_age`` aus
    der laengsten tatsaechlich gefahrenen stint-laenge dieser mischung. eine
    vorsichtige untergrenze, kein gemessenes limit: niemand testet in einem
    echten rennen, wie lange ein reifen wirklich haelt.

    ``base_time`` ergibt sich aus ``fit.intercept + fit.slope``: der fit
    extrapoliert auf reifenalter 0 (``intercept``), TyreLife ist bei FastF1
    aber einsbasiert. die erste echte runde auf dem satz hat TyreLife 1,
    also ``intercept + 1 * slope``.

    Raises:
        ValueError: keine belastbaren fits, oder (mit
            ``require_two_compounds``) nur eine mischung belastbar.
    """
    deg = degradation(session, min_laps=fit_min_laps)
    ok = deg[deg["reliable"]] if not deg.empty else deg
    if ok.empty:
        raise ValueError("keine belastbaren Degradations-Fits fuer diese Session")
    if require_two_compounds and ok["compound"].nunique() < 2:
        raise ValueError(
            "nur eine Mischung mit belastbaren Fits - "
            "require_two_compounds=False setzen oder andere Session waehlen")

    tyres = []
    for compound, g in ok.groupby("compound"):
        slope = float(g["deg_s_per_lap"].median())
        intercept = float(g["base_s"].median())
        tyres.append(TyreModel(
            compound=str(compound), base_time=intercept + slope,
            deg_linear=slope, max_age=int(g["laps"].max())))
    tyres.sort(key=lambda t: t.base_time)

    return RaceConfig(
        n_laps=int(session.total_laps), pit_loss=pit_loss(session),
        tyres=tuple(tyres), min_stint=optimizer_min_stint,
        require_two_compounds=require_two_compounds,
        fuel_effect=FUEL_KG_PER_LAP * FUEL_S_PER_KG)


# --------------------------------------------------------------- Boxenstopps
def pit_loss(session) -> float:
    """zeitverlust eines boxenstopps auf dieser strecke, in sekunden.

    vergleicht in- und out-laps mit der normalen rundenzeit desselben
    fahrers. enthaelt damit anfahrt, standzeit und ausfahrt.

    runden unter rotem flag ausgeschlossen: ``PitInTime``/``PitOutTime``
    stehen dann oft auf runden, in denen autos waehrend der
    session-unterbrechung geparkt bzw. wieder losgeschickt werden. keine
    echten boxenstopps, aber mit rundenzeiten von zig minuten (ein einzelner
    solcher fall kann den median stark verzerren). ``TrackStatus`` traegt
    bei FastF1 alle waehrend der runde durchlaufenen zustaende als
    zeichenkette (z.b. "1254"). "5" irgendwo darin heisst rotes flag zu
    irgendeinem zeitpunkt in dieser runde.
    """
    laps = session.laps.copy()
    laps["sec"] = laps["LapTime"].dt.total_seconds()

    baseline = (clean_laps(session)
                .groupby("Driver")["LapTime"].median().dt.total_seconds())

    nicht_rot = ~laps["TrackStatus"].astype(str).str.contains("5", na=False)
    in_laps = laps[laps["PitInTime"].notna() & nicht_rot]
    out_laps = laps[laps["PitOutTime"].notna() & nicht_rot]

    return estimate_pit_loss(
        (in_laps["sec"] - in_laps["Driver"].map(baseline)).dropna(),
        (out_laps["sec"] - out_laps["Driver"].map(baseline)).dropna(),
    )


# --------------------------------------------------------------- Track Status
def track_status_phases(session) -> pd.DataFrame:
    """safety-car-, VSC- und gelbphasen als intervalle mit rundennummern.

    ``session.track_status`` ist ein log von zustandswechseln, keine
    zeitreihe: "SCDeployed" steht dort typischerweise genau einmal, auch wenn
    das safety car neun runden lang draussen bleibt. ein intervall muss
    deshalb bis zum naechsten wechsel reichen, nicht nur bis zum letzten
    auftreten desselben codes (siehe :func:`f1lab.core.status_intervals`.
    die urspruengliche fassung gruppierte nach gleichem code und ergab so
    fast durchweg intervalle der laenge 0, siehe P18).
    """
    ts = session.track_status.copy()
    ts["label"] = ts["Status"].map(TRACK_STATUS).fillna(ts["Status"])
    ts["Sekunden"] = ts["Time"].dt.total_seconds()

    labels, start_s, end_s = status_intervals(
        ts["label"].to_numpy(), ts["Sekunden"].to_numpy())
    session_ende = session.laps["Time"].max()
    session_ende_s = (session_ende.total_seconds()
                      if pd.notna(session_ende) else np.nan)
    end_s = np.where(np.isnan(end_s), session_ende_s, end_s)
    # die letzte runden-zeit kann vor dem start der letzten status-phase
    # liegen (z.B. eine spaete meldung nach der zieldurchfahrt, gesehen bei
    # einer session mit rotem start). ohne diese klammer wuerde daraus eine
    # negative dauer.
    end_s = np.maximum(end_s, start_s)

    phases = pd.DataFrame({
        "label": labels,
        "start": pd.to_timedelta(start_s, unit="s"),
        "end": pd.to_timedelta(end_s, unit="s"),
    })
    phases["duration_s"] = (phases["end"] - phases["start"]).dt.total_seconds()

    leader = (session.laps[session.laps["Position"] == 1]
              [["LapNumber", "LapStartTime"]].dropna())

    def lap_at(t):
        m = leader[leader["LapStartTime"] <= t]
        return int(m["LapNumber"].max()) if len(m) else 0

    phases["lap_start"] = phases["start"].apply(lap_at)
    phases["lap_end"] = phases["end"].apply(lap_at)
    return phases


# --------------------------------------------------------------- dimensionen
# FastF1 liefert positionen in zehntelmetern (siehe core.py-doku zu X/Y/Z).
POS_UNITS_PER_M = 10.0

EVENT_DIM_COLS = ["season", "round", "event_name", "official_name", "country",
                  "location", "event_date", "event_format", "is_sprint",
                  "n_sessions"]


def event_dimension(years) -> pd.DataFrame:
    """rennkalender mehrerer saisons als dimensionstabelle.

    eine zeile je rennwochenende, jahre untereinander. saisons, die die API
    nicht kennt, werden uebersprungen statt den aufbau abzubrechen. bei der
    laufenden saison steht der kalender teils erst spaet.

    das sprint-format heisst je nach jahr anders ("sprint",
    "sprint_shootout", "sprint_qualifying"), deshalb wird auf das teilwort
    geprueft und nicht auf gleichheit.
    """
    frames = []
    for year in years:
        try:
            sched = fastf1.get_event_schedule(int(year), include_testing=False)
        except Exception:
            continue
        if sched is None or not len(sched):
            continue
        sched = sched.copy()
        sched["Season"] = int(year)
        frames.append(sched)

    if not frames:
        return pd.DataFrame(columns=EVENT_DIM_COLS)

    cal = pd.concat(frames, ignore_index=True)
    fmt = cal["EventFormat"].astype("string").fillna("")

    session_cols = [c for c in cal.columns
                    if c.startswith("Session") and c[7:].isdigit()]
    n_sessions = (cal[session_cols].notna()
                  & (cal[session_cols].astype("string") != "")).sum(axis=1)

    out = pd.DataFrame({
        "season": cal["Season"].astype(int),
        "round": cal["RoundNumber"].astype(int),
        "event_name": cal["EventName"].astype("string"),
        "official_name": cal.get("OfficialEventName", pd.Series(dtype="string")),
        "country": cal["Country"].astype("string"),
        "location": cal["Location"].astype("string"),
        "event_date": pd.to_datetime(cal["EventDate"]),
        "event_format": fmt,
        "is_sprint": fmt.str.contains("sprint", case=False, na=False),
        "n_sessions": n_sessions.astype(int),
    })
    return out.sort_values(["season", "round"], ignore_index=True)


def circuit_geometry(session) -> dict:
    """kurvenzahl, streckenlaenge und hoehenprofil einer geladenen session.

    braucht telemetrie: die kurvenliste kommt zwar aus der MultiViewer-API,
    laenge und hoehe aber aus dem positionskanal der schnellsten runde.
    deshalb ist das hier bewusst getrennt von :func:`event_dimension`, die
    ohne jeden session-download auskommt.

    gemessen wird die *gefahrene linie*, nicht die offizielle streckenlaenge.
    die ideallinie schneidet kurven und faellt dadurch typisch ein bis zwei
    prozent kuerzer aus als die angabe im reglement.
    """
    out = {"corners": pd.NA, "length_m": pd.NA,
           "elev_gain_m": pd.NA, "elev_span_m": pd.NA}

    # kurvenliste und positionsdaten kommen aus verschiedenen quellen und
    # fallen unabhaengig voneinander aus. deshalb einzeln abgesichert.
    with contextlib.suppress(Exception):
        out["corners"] = int(len(session.get_circuit_info().corners))

    try:
        pos = session.laps.pick_fastest().get_pos_data()
    except Exception:
        return out
    if pos is None or pos.empty:
        return out

    out["length_m"] = round(
        path_length(pos["X"], pos["Y"]) / POS_UNITS_PER_M, 1)
    elev = elevation_profile(pos["Z"].to_numpy() / POS_UNITS_PER_M)
    out["elev_gain_m"] = elev.gain
    out["elev_span_m"] = elev.span
    return out


def circuit_dimension(events, identifier: str = "Q") -> pd.DataFrame:
    """circuit-dimension aus einer liste von (jahr, GP)-paaren.

    streckengeometrie ist pro layout konstant, es genuegt also *eine* session
    je strecke, nicht das ganze archiv. sessions, deren telemetrie nicht im
    cache liegt, werden mit leeren kennzahlen zurueckgegeben statt den aufbau
    abzubrechen. die zeile bleibt erhalten, damit sichtbar ist, was fehlt.
    """
    rows = []
    for year, gp in events:
        row = {"season": int(year), "gp": str(gp)}
        try:
            ses = load(int(year), gp, identifier, telemetry=True)
            row["circuit"] = str(ses.event["Location"])
            row.update(circuit_geometry(ses))
        except Exception as exc:
            row["circuit"] = str(gp)
            row["error"] = f"{type(exc).__name__}"
        rows.append(row)

    df = pd.DataFrame(rows)
    if "length_m" in df.columns:
        df = df.sort_values("length_m", ascending=False, ignore_index=True)
    return df


def lap_speed_profile(session) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """distanz, streckenkruemmung und echte geschwindigkeit der schnellsten
    runde. eingabe fuer die rundenzeit-simulation (siehe P37,
    :func:`f1lab.core.simulate_lap`/:func:`calibrate_lap_model`).

    Returns:
        (distanz [m], kruemmung [1/m], geschwindigkeit [m/s])
    """
    lap = session.laps.pick_fastest()
    tel = lap.get_telemetry().add_distance()
    dist = tel["Distance"].to_numpy(dtype=float)
    kappa = track_curvature(tel["X"], tel["Y"], dist)
    speed_ms = tel["Speed"].to_numpy(dtype=float) / 3.6
    return dist, kappa, speed_ms


def corner_labels(session) -> pd.DataFrame:
    """kurven der session, auf die distanz einer referenzrunde projiziert.

    die XY-position jeder kurve (aus get_circuit_info(), unabhaengig von
    jedem fahrer) wird auf den naechstgelegenen punkt der schnellsten runde
    der session projiziert. das gibt jeder kurve eine distanz entlang der
    strecke, mit der sich telemetrie faehrerbergreifend vergleichen laesst.
    """
    ci = session.get_circuit_info()
    ref = session.laps.pick_fastest().get_telemetry().add_distance()
    ref_xy = ref[["X", "Y"]].to_numpy(dtype=float)
    ref_dist = ref["Distance"].to_numpy()

    distanzen = []
    for c in ci.corners.itertuples():
        d = np.hypot(ref_xy[:, 0] - c.X, ref_xy[:, 1] - c.Y)
        distanzen.append(float(ref_dist[np.argmin(d)]))

    out = ci.corners.copy()
    out["Distance"] = distanzen
    out["label"] = [f"T{int(n)}{letter}" for n, letter in
                    zip(out["Number"], out["Letter"], strict=True)]
    return out.sort_values("Distance", ignore_index=True)


def marshal_sector_labels(session) -> pd.DataFrame:
    """marshal-sektorgrenzen, auf dieselbe referenzrunde projiziert wie
    :func:`corner_labels`. dieselbe idee, andere punktliste (siehe P11).
    """
    ci = session.get_circuit_info()
    ref = session.laps.pick_fastest().get_telemetry().add_distance()
    ref_xy = ref[["X", "Y"]].to_numpy(dtype=float)
    ref_dist = ref["Distance"].to_numpy()

    zeilen = []
    for m in ci.marshal_sectors.itertuples():
        d = np.hypot(ref_xy[:, 0] - m.X, ref_xy[:, 1] - m.Y)
        zeilen.append({"number": int(m.Number),
                       "distance": float(ref_dist[np.argmin(d)])})
    return pd.DataFrame(zeilen).sort_values("distance", ignore_index=True)


def marshal_light_labels(session) -> pd.DataFrame:
    """positionen der gelbflaggen-lichttafeln, auf dieselbe referenzrunde
    projiziert wie :func:`corner_labels`/:func:`marshal_sector_labels`.
    dieselbe idee, dritte punktliste. ``CircuitInfo.marshal_lights`` liefert
    kein ``Distance`` (anders als ``corners``), deshalb dieselbe
    naechste-nachbar-projektion wie bei den anderen beiden."""
    ci = session.get_circuit_info()
    ref = session.laps.pick_fastest().get_telemetry().add_distance()
    ref_xy = ref[["X", "Y"]].to_numpy(dtype=float)
    ref_dist = ref["Distance"].to_numpy()

    zeilen = []
    for m in ci.marshal_lights.itertuples():
        d = np.hypot(ref_xy[:, 0] - m.X, ref_xy[:, 1] - m.Y)
        zeilen.append({"number": int(m.Number),
                       "distance": float(ref_dist[np.argmin(d)])})
    return pd.DataFrame(zeilen).sort_values("distance", ignore_index=True)


def corner_speeds(session, window_m: float = 60.0) -> pd.DataFrame:
    """minimalgeschwindigkeit je kurve und fahrer (siehe P11/P12).

    kurven kommen aus :func:`corner_labels` (auf eine referenzrunde
    projiziert). je fahrer wird die minimalgeschwindigkeit in einem fenster
    von +/- window_m um die kurvendistanz gesucht, nicht per exakter
    naechster-nachbar-zuordnung: benachbarte kurven koennten sich sonst
    gegenseitig telemetriepunkte wegnehmen.

    Returns:
        DataFrame fahrer x kurve (kuerzel wie "T7"), NaN wo kein
        telemetriepunkt im fenster liegt.
    """
    corners = corner_labels(session)
    rows = {}
    for drv in session.drivers:
        try:
            lap = session.laps.pick_drivers(drv).pick_fastest()
            tel = lap.get_telemetry().add_distance()
        except Exception:
            continue
        if tel is None or tel.empty or pd.isna(lap["LapTime"]):
            continue
        info = session.get_driver(drv)
        d = tel["Distance"].to_numpy()
        v = tel["Speed"].to_numpy()
        speeds = {}
        for c in corners.itertuples():
            m = (d > c.Distance - window_m) & (d < c.Distance + window_m)
            if m.any():
                speeds[c.label] = float(v[m].min())
        rows[info["Abbreviation"]] = speeds
    return pd.DataFrame(rows).T


# --------------------------------------------------------------- bremszonen
def driver_braking_zones(session, driver: str, min_length_m: float = 20.0
                         ) -> pd.DataFrame:
    """bremszonen der schnellsten runde eines fahrers (siehe P08).

    braucht telemetrie. leerer rahmen, wenn der fahrer keine gewertete
    schnellste runde hat (z.b. nach einem ausfall vor der ersten runde).
    """
    lap = session.laps.pick_drivers(driver).pick_fastest()
    if lap is None or pd.isna(lap["LapTime"]):
        return pd.DataFrame()
    car = lap.get_car_data().add_distance()
    return pd.DataFrame(braking_zones(
        car["Brake"], car["Distance"], car["Speed"],
        car["Time"].dt.total_seconds(), min_length_m=min_length_m))


def compare_braking_zones(zones_a: pd.DataFrame, zones_b: pd.DataFrame,
                          tolerance_m: float = 150.0) -> pd.DataFrame:
    """bremszonen zweier fahrer paaren und den abstand ihrer bremspunkte
    zeigen (siehe P07/P08). nutzt :func:`f1lab.core.match_by_distance`.
    """
    if zones_a.empty or zones_b.empty:
        return pd.DataFrame()
    paare = match_by_distance(zones_a["start_m"], zones_b["start_m"],
                              tolerance_m)
    zeilen = [{"start_m_a": zones_a["start_m"].iloc[i],
              "start_m_b": zones_b["start_m"].iloc[j],
              "delta_m": zones_b["start_m"].iloc[j] - zones_a["start_m"].iloc[i]}
             for i, j in paare]
    return pd.DataFrame(zeilen).sort_values("start_m_a", ignore_index=True)


# --------------------------------------------------------------- DRS
def drs_zones(session, driver: str, min_length_m: float = 100.0
             ) -> pd.DataFrame:
    """DRS-aktivzonen der schnellsten runde eines fahrers (siehe P10).

    filtert alles unter ``min_length_m`` als rauschen. ohne den filter
    meldet dieselbe flankenlogik am start/ziel-bereich mehrere kurze
    wackel-zonen, rest-aktivierung vom ende der vorrunde.
    """
    lap = session.laps.pick_drivers(driver).pick_fastest()
    if lap is None or pd.isna(lap["LapTime"]):
        return pd.DataFrame()
    car = lap.get_car_data().add_distance()
    offen = drs_state(car["DRS"].to_numpy()) == 2
    return pd.DataFrame(active_distance_zones(offen, car["Distance"].to_numpy(),
                                              min_length_m=min_length_m))


def drs_usage(session) -> pd.DataFrame:
    """DRS-zeitanteil und topspeed-gewinn je fahrer (siehe P10 VORGEHEN 1/3/4)."""
    rows = []
    for drv in session.drivers:
        try:
            lap = session.laps.pick_drivers(drv).pick_fastest()
            tel = lap.get_car_data().add_distance()
        except Exception:
            continue
        if tel is None or tel.empty or pd.isna(lap["LapTime"]):
            continue

        offen = drs_state(tel["DRS"].to_numpy()) == 2
        dt = tel["Time"].diff().dt.total_seconds().fillna(0).to_numpy()
        total_t = dt.sum()
        if total_t <= 0:
            continue

        info = session.get_driver(drv)
        vmax_offen = tel.loc[offen, "Speed"].max()
        vmax_zu = tel.loc[~offen, "Speed"].max()
        rows.append({
            "driver": info["Abbreviation"], "team": info["TeamName"],
            "drs_s": round(float(dt[offen].sum()), 2),
            "drs_pct": round(100 * dt[offen].sum() / total_t, 1),
            "vmax_offen": vmax_offen, "vmax_zu": vmax_zu,
            "gewinn_kmh": round(vmax_offen - vmax_zu, 1) if pd.notna(vmax_offen)
            and pd.notna(vmax_zu) else float("nan"),
        })
    return pd.DataFrame(rows).sort_values("drs_pct", ascending=False,
                                          ignore_index=True)


# --------------------------------------------------------------- position
def position_progression(session) -> pd.DataFrame:
    """position je runde und fahrer, pivotiert (siehe P20 VORGEHEN 1)."""
    return session.laps.pivot_table(index="LapNumber", columns="Driver",
                                    values="Position", aggfunc="first")


def overtake_events(session) -> pd.DataFrame:
    """einzelne ueberholvorgaenge (fahrer A ueberholt fahrer B in runde N),
    ohne boxenstopp-effekt und nur auf gruener flagge (siehe P20 VORGEHEN
    3/4). :func:`overtakes_matrix` aggregiert genau diese liste zu einer
    matrix, :func:`overtake_locations` (P39) sucht je ereignis die stelle
    auf der strecke.

    zwei faelle zaehlen nicht als echtes duell: ein boxenstopp verschiebt
    die position ohne ueberholen auf der strecke, und ein
    safety-car-restart wirbelt das feld durcheinander, ohne dass position
    durch tempo gewonnen wurde. die positionstabelle selbst bleibt ueber
    die volle, ungefilterte rundenliste. sonst fehlen rundennummern in der
    reihe, sobald eine runde komplett herausfaellt, und lap - 1 zeigt ins
    leere.

    Returns:
        DataFrame mit spalten ``gainer``, ``loser``, ``lap``.
    """
    laps = session.laps
    pos = laps.pivot_table(index="LapNumber", columns="Driver",
                           values="Position", aggfunc="first")
    if pos.empty:
        return pd.DataFrame(columns=["gainer", "loser", "lap"])

    box_laps = set(zip(laps.loc[laps["PitInTime"].notna(), "Driver"],
                       laps.loc[laps["PitInTime"].notna(), "LapNumber"]))
    gruene_laps = set(zip(laps.pick_track_status("1")["Driver"],
                          laps.pick_track_status("1")["LapNumber"]))
    drivers = list(pos.columns)
    events = []

    for lap in pos.index[1:]:
        prev, cur = pos.loc[lap - 1], pos.loc[lap]
        for a in drivers:
            for b in drivers:
                if a == b or pd.isna(prev[a]) or pd.isna(cur[a]) \
                        or pd.isna(prev[b]) or pd.isna(cur[b]):
                    continue
                if prev[a] > prev[b] and cur[a] < cur[b]:
                    if (b, lap) in box_laps or (a, lap - 1) in box_laps:
                        continue
                    if (a, lap) not in gruene_laps or (b, lap) not in gruene_laps:
                        continue
                    events.append({"gainer": a, "loser": b, "lap": int(lap)})
    return pd.DataFrame(events, columns=["gainer", "loser", "lap"])


def overtakes_matrix(session) -> pd.DataFrame:
    """wer ueberholt wen wie oft (siehe P20 VORGEHEN 3/4). zeile ueberholt
    spalte. aggregiert :func:`overtake_events`."""
    laps = session.laps
    drivers = list(laps.pivot_table(index="LapNumber", columns="Driver",
                                    values="Position", aggfunc="first").columns)
    mat = pd.DataFrame(0, index=drivers, columns=drivers)
    for e in overtake_events(session).itertuples():
        mat.loc[e.gainer, e.loser] += 1
    return mat


def overtake_locations(session, drs_session=None, drs_referenz: str | None = None,
                       naehe_m: float = 30.0) -> pd.DataFrame:
    """ort jedes ueberholvorgangs auf der strecke, gegen die DRS-zonen
    geprueft (siehe P39).

    fuer jedes ereignis aus :func:`overtake_events`: in der telemetrie des
    ueberholers (``gainer``) fuer genau diese runde die letzte stelle
    suchen, an der ``DriverAhead`` (FastF1s eigener, GPS-basierter
    "auto direkt davor"-kanal) noch dem ueberholten (``loser``) entspricht.
    das ist die stelle kurz vor dem positionswechsel. nur uebernommen, wenn
    dort auch ``DistanceToDriverAhead`` unter ``naehe_m`` liegt (sonst war
    ``loser`` zwar irgendwann auf der runde davor, aber nicht mehr in dem
    moment, der als wechsel gezaehlt wird, z.b. bei mehreren
    positionswechseln in derselben runde).

    die DRS-zonen kommen bewusst NICHT aus der renn-session selbst: DRS
    braucht in einem rennen einen rueckstand unter 1s auf das auto davor,
    eine schnellste rennrunde entsteht aber typisch in freier fahrt ohne
    vordermann. dann bleibt DRS auf der ganzen runde zu, und
    :func:`drs_zones` faende keine einzige zone, obwohl die zonen
    physisch existieren (siehe P39 fuer den Monza-fall, an dem das
    auffiel). im qualifying ist DRS ohne abstandsregel verfuegbar, deshalb
    default: die qualifying-session desselben events (ueber die
    rundennummer, nicht den streckennamen, robuster gegen
    schreibweisen). ``drs_session`` laesst sich trotzdem explizit setzen,
    falls keine qualifying-telemetrie im cache liegt.

    Returns:
        DataFrame mit ``gainer``, ``loser``, ``lap``, ``distance_m``,
        ``in_drs_zone``. ereignisse ohne verwertbare telemetriestelle
        fehlen in der rueckgabe (nicht als zeile mit NaN). die differenz
        zu :func:`overtake_events` ist die abdeckung dieser methode.
    """
    events = overtake_events(session)
    if events.empty:
        return pd.DataFrame(columns=["gainer", "loser", "lap", "distance_m",
                                     "in_drs_zone"])

    nummer_zu_code = (session.laps[["Driver", "DriverNumber"]]
                      .drop_duplicates().set_index("DriverNumber")["Driver"]
                      .to_dict())
    code_zu_nummer = {v: k for k, v in nummer_zu_code.items()}

    if drs_session is None:
        drs_session = load(int(session.event.year),
                           int(session.event["RoundNumber"]), "Q",
                           telemetry=True)
    if drs_referenz is None:
        drs_referenz = str(drs_session.laps.pick_fastest()["Driver"])
    zonen = drs_zones(drs_session, drs_referenz)
    zone_starts = zonen["start_m"].to_numpy() if not zonen.empty else np.array([])
    zone_ends = zonen["end_m"].to_numpy() if not zonen.empty else np.array([])

    rows = []
    for e in events.itertuples():
        loser_nr = code_zu_nummer.get(e.loser)
        if loser_nr is None:
            continue
        try:
            lap = session.laps.pick_drivers(e.gainer).pick_laps(e.lap)
            tel = lap.get_telemetry().add_distance().add_driver_ahead()
        except Exception:
            continue
        if tel.empty:
            continue

        treffer = tel[(tel["DriverAhead"] == loser_nr)
                     & (tel["DistanceToDriverAhead"] < naehe_m)]
        if treffer.empty:
            continue
        letzte = treffer.iloc[-1]
        rows.append({"gainer": e.gainer, "loser": e.loser, "lap": e.lap,
                    "distance_m": round(float(letzte["Distance"]), 1)})

    orte = pd.DataFrame(rows, columns=["gainer", "loser", "lap", "distance_m"])
    if orte.empty:
        orte["in_drs_zone"] = pd.Series(dtype=bool)
        return orte
    orte["in_drs_zone"] = distance_in_any_zone(orte["distance_m"], zone_starts,
                                               zone_ends)
    return orte


def undercut_duels(session, fenster: int = 3, nachlauf: int = 2) -> pd.DataFrame:
    """echte, paarweise undercut-versuche und ob sie gelangen (siehe P42).

    anders als eine flaechendeckende vorher-nachher-positionszaehlung (die
    JEDEN boxenstopp automatisch als verlust gegen das ganze feld zaehlt,
    weil nicht-stoppende autos in der zwischenzeit einfach weiterfahren,
    siehe CLAUDE.md, verworfener erster versuch) vergleicht das hier gezielt
    gegen einen konkreten rivalen: fuer jeden boxenstopp (fahrer A, runde L)
    der fahrer, der zu rundenbeginn genau eine position vor A lag (der
    eigentliche gegner des stopps). nur gezaehlt, wenn dieser rivale nicht
    selbst in derselben runde stoppt (sonst kein undercut-versuch) und
    innerhalb von ``fenster`` runden danach selbst an die box faehrt (sonst
    ist die verfolgung kein undercut, sondern nur zufaellig zeitversetzte
    stopps). erfolg heisst: ``nachlauf`` runden nach dem spaeteren der
    beiden stopps liegt A vor dem rivalen.

    Args:
        fenster: wie viele runden nach As stopp der rivale noch selbst
            stoppen darf, damit es als undercut-versuch zaehlt.
        nachlauf: wie viele gruene runden nach dem letzten der beiden
            stopps abgewartet wird, bevor die position verglichen wird.

    Returns:
        DataFrame mit ``driver``, ``lap``, ``rival``, ``rival_lap``,
        ``erfolg``.
    """
    laps = session.laps
    pos = laps.pivot_table(index="LapNumber", columns="Driver",
                           values="Position", aggfunc="first")
    if pos.empty:
        return pd.DataFrame(columns=["driver", "lap", "rival", "rival_lap",
                                     "erfolg"])
    box_laps_liste = list(zip(laps.loc[laps["PitInTime"].notna(), "Driver"],
                              laps.loc[laps["PitInTime"].notna(), "LapNumber"]))
    box_laps = set(box_laps_liste)

    versuche = []
    for drv, lap in box_laps_liste:
        if lap not in pos.index or (lap - 1) not in pos.index:
            continue
        p_vor = pos.loc[lap - 1, drv]
        if pd.isna(p_vor):
            continue
        kandidaten = pos.loc[lap - 1]
        rivale = kandidaten[kandidaten == p_vor - 1]
        if rivale.empty:
            continue
        rival_drv = str(rivale.index[0])
        if (rival_drv, lap) in box_laps:
            continue
        rivale_stopps = sorted(stopp for (d, stopp) in box_laps
                               if d == rival_drv and lap < stopp <= lap + fenster)
        if not rivale_stopps:
            continue
        rival_lap = rivale_stopps[0]
        nach = max(lap, rival_lap) + nachlauf
        if nach not in pos.index:
            continue
        p_a_nach, p_r_nach = pos.loc[nach, drv], pos.loc[nach, rival_drv]
        if pd.isna(p_a_nach) or pd.isna(p_r_nach):
            continue
        versuche.append({"driver": drv, "lap": int(lap), "rival": rival_drv,
                        "rival_lap": int(rival_lap),
                        "erfolg": bool(p_a_nach < p_r_nach)})
    return pd.DataFrame(versuche, columns=["driver", "lap", "rival",
                                           "rival_lap", "erfolg"])


def qualifying_track_evolution(session) -> pd.DataFrame:
    """paarweise fahrer-deltas zwischen Q1/Q2/Q3 als mass fuer streckenentwicklung
    (siehe P43).

    vergleicht je fahrer die schnellste gewertete rundenzeit eines segments
    gegen die des naechsten, nur fuer fahrer, die in BEIDEN segmenten eine
    zeit gesetzt haben. dieser paarweise vergleich haelt auto- und
    fahrerqualitaet konstant: ein reiner vergleich der segment-durchschnitte
    waere verzerrt, weil in Q2/Q3 nur die schnelleren autos uebrig bleiben.
    "im schnitt schneller" wuerde dann teilweise nur "die langsameren autos
    sind raus" messen, nicht streckenentwicklung.

    Args:
        session: geladene qualifying-session (braucht session-status-daten
            fuer ``Laps.split_qualifying_sessions()``).

    Returns:
        DataFrame mit ``driver``, ``segment`` ("Q1->Q2"/"Q2->Q3``),
        ``delta_s`` (positiv heisst: das spaetere segment war schneller).
    """
    laps = session.laps
    try:
        q1, q2, q3 = laps.split_qualifying_sessions()
    except Exception:
        return pd.DataFrame(columns=["driver", "segment", "delta_s"])

    def bestzeit_je_fahrer(q) -> pd.Series:
        if q is None or q.empty:
            return pd.Series(dtype="timedelta64[ns]")
        gueltig = q.dropna(subset=["LapTime"])
        return gueltig.groupby("Driver")["LapTime"].min()

    b1 = bestzeit_je_fahrer(q1)
    b2 = bestzeit_je_fahrer(q2)
    b3 = bestzeit_je_fahrer(q3)

    zeilen = []
    for segment, a, b in (("Q1->Q2", b1, b2), ("Q2->Q3", b2, b3)):
        gemeinsam = a.index.intersection(b.index)
        delta = (a[gemeinsam] - b[gemeinsam]).dt.total_seconds()
        for fahrer, wert in delta.items():
            zeilen.append({"driver": fahrer, "segment": segment,
                           "delta_s": float(wert)})
    return pd.DataFrame(zeilen, columns=["driver", "segment", "delta_s"])


# --------------------------------------------------------------- start
def _zeit_bei_speed(t: np.ndarray, v: np.ndarray, ziel: float, t0: float
                    ) -> float | None:
    """erste zeit (relativ zu t0), zu der v mindestens ziel erreicht."""
    mask = v >= ziel
    return round(float(t[mask.argmax()] - t0), 2) if mask.any() else None


def start_performance(session, fenster_s: float = 8.0) -> pd.DataFrame:
    """startkennzahlen je fahrer: zeit bis 100/200 km/h, distanz nach 5s,
    positionsgewinn grid -> ende runde 1 (siehe P31).

    boxenstarts (PitOutTime auf runde 1 gesetzt) werden ausgeschlossen. ein
    start aus der box hat eine komplett andere ausgangsgeschwindigkeit
    (boxengassen-limit statt ampel-start) und ist nicht vergleichbar.

    liest ``session.results`` fuer die startaufstellung. das kann fuer
    manche saisons echten Ergast/jolpica-netzzugriff ausloesen, obwohl es
    wie eine reine lokaldaten-spalte aussieht (siehe P40, dort in einem
    saison-scan entdeckt). fuer eine einzelne session unkritisch, bei vielen
    sessions hintereinander siehe dortige drosselung.
    """
    rows = []
    for drv in session.drivers:
        info = session.get_driver(drv)
        try:
            lap1 = session.laps.pick_drivers(drv).pick_laps(1).iloc[0]
        except (IndexError, KeyError):
            continue
        if pd.notna(lap1["PitOutTime"]):
            continue

        tel = lap1.get_car_data().add_distance()
        if tel is None or tel.empty:
            continue
        start = tel["SessionTime"].iloc[0]
        fenster = tel.slice_by_time(start, start + pd.Timedelta(seconds=fenster_s))
        if fenster.empty:
            continue

        t = fenster["Time"].dt.total_seconds().to_numpy()
        v = fenster["Speed"].to_numpy()
        d = fenster["Distance"].to_numpy()
        t0 = t[0]
        nach_5s = d[(t - t0) <= 5]
        grid = session.results.loc[session.results["DriverNumber"] == drv,
                                   "GridPosition"].squeeze()
        ende = lap1["Position"]
        rows.append({
            "driver": info["Abbreviation"], "grid": grid, "ende_r1": ende,
            "gewinn": (grid - ende) if pd.notna(ende) and pd.notna(grid)
            else None,
            "t_100": _zeit_bei_speed(t, v, 100, t0),
            "t_200": _zeit_bei_speed(t, v, 200, t0),
            "m_nach_5s": round(float(nach_5s.max()), 1) if nach_5s.size else None,
        })
    return pd.DataFrame(rows).sort_values("m_nach_5s", ascending=False,
                                          ignore_index=True)


def grid_lap1_positions(session) -> pd.DataFrame:
    """startplatz gegen position am ende von runde 1, ohne boxenstarts
    (siehe P40, die telemetriefreie variante von P31s startkennzahlen,
    fuer saison-scans ueber viele rennen ohne telemetrie-download).

    Returns:
        DataFrame mit ``driver_number``, ``grid``, ``lap1``, ``gewinn``
        (grid - lap1, positiv = positionen gewonnen).
    """
    grid = session.results[["DriverNumber", "GridPosition"]].dropna()
    lap1 = session.laps.pick_laps(1)
    lap1 = lap1.loc[lap1["PitOutTime"].isna(), ["DriverNumber", "Position"]].dropna()
    m = grid.merge(lap1, on="DriverNumber").rename(
        columns={"DriverNumber": "driver_number", "GridPosition": "grid",
                "Position": "lap1"})
    m["gewinn"] = m["grid"] - m["lap1"]
    return m


# --------------------------------------------------------------- verfolgung
def close_following(session, driver: str, nah_schwelle_m: float = 50.0
                    ) -> pd.DataFrame:
    """abstand zum vordermann je gruener runde, treibstoffkorrigiert
    (siehe P32 VORGEHEN 1-2).

    add_driver_ahead() laeuft einmal auf die gesamte renntelemetrie des
    fahrers, nicht einmal je runde. um ein vielfaches schneller bei
    identischem ergebnis (siehe P05/P20/P32).
    """
    laps = (session.laps.pick_drivers(driver).pick_wo_box().pick_accurate()
           .pick_track_status("1").sort_values("LapStartTime"))
    if laps.empty:
        return pd.DataFrame()
    try:
        tel = laps.get_telemetry().add_driver_ahead().sort_values("SessionTime")
    except Exception:
        return pd.DataFrame()
    if tel.empty:
        return pd.DataFrame()

    grenzen = (laps[["LapNumber", "LapStartTime"]]
              .rename(columns={"LapStartTime": "SessionTime"})
              .sort_values("SessionTime"))
    zug = pd.merge_asof(tel, grenzen, on="SessionTime", direction="backward")
    zug["gap"] = zug["DistanceToDriverAhead"].replace(0, float("nan"))

    total_laps = int(session.total_laps)
    rows = []
    for lapnum, g in zug.groupby("LapNumber"):
        treffer = laps.loc[laps["LapNumber"] == lapnum]
        if treffer.empty or pd.isna(treffer.iloc[0]["LapTime"]):
            continue
        lap = treffer.iloc[0]
        sec = lap["LapTime"].total_seconds()
        rows.append({
            "lap": int(lapnum),
            "sec_fuel": float(fuel_correct([sec], [lapnum], total_laps)[0]),
            "gap_median_m": g["gap"].median(), "gap_min_m": g["gap"].min(),
            "anteil_nah": 100 * (g["gap"] < nah_schwelle_m).mean(),
            "compound": lap["Compound"], "tyre_life": lap["TyreLife"],
        })
    return pd.DataFrame(rows)


def dirty_air_effect(df: pd.DataFrame) -> tuple[float, float, float, pd.DataFrame]:
    """rundenzeit (bereits treibstoffkorrigiert) gegen nahanteil regressieren,
    nach herausrechnen der reifendegradation (siehe P32).

    nutzt :func:`fit_degradation` fuer die degradations-bereinigung, dieselbe
    funktion wie in P13/dashboard, nur hier auf nahanteil statt compound
    angewendet.

    Args:
        df: ergebnis von :func:`close_following`.

    Returns:
        (slope, intercept, r2, df mit zusaetzlicher spalte ``sec_corr``).
        slope/intercept/r2 sind NaN, wenn zu wenige runden vorliegen.
    """
    d = df.dropna(subset=["gap_median_m", "tyre_life"]).copy()
    d = d[d["gap_median_m"] < 500]
    if len(d) < 5 or d["tyre_life"].nunique() < 2:
        return float("nan"), float("nan"), float("nan"), d

    fit = fit_degradation(d["tyre_life"], d["sec_fuel"])
    d["sec_corr"] = d["sec_fuel"] - fit.slope * d["tyre_life"]

    slope, inter = np.polyfit(d["anteil_nah"], d["sec_corr"], 1)
    pred = slope * d["anteil_nah"] + inter
    ss_res = float(((d["sec_corr"] - pred) ** 2).sum())
    ss_tot = float(((d["sec_corr"] - d["sec_corr"].mean()) ** 2).sum())
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return float(slope), float(inter), r2, d


# --------------------------------------------------------------- mini-sektoren
def mini_sectors(session, drivers: list[str], n: int = 25) -> dict:
    """zerlegt die runde in n gleich lange distanz-abschnitte und ermittelt
    je abschnitt, welcher der uebergebenen fahrer dort am wenigsten zeit
    gebraucht hat (siehe P06).

    nur wenige fahrer uebergeben (MAX_SERIEN aus f1lab.design). bei allen
    20 waere weder die farbdarstellung noch die frage "wer dominiert wo"
    sinnvoll, siehe P06-docstring.

    Returns:
        dict mit ``telemetrie`` (distanz/zeit/X/Y je fahrer),
        ``edges`` (grenzen der abschnitte) und ``gewinner`` (fahrer je
        abschnitt, laenge n).
    """
    telemetrie = {}
    for drv in drivers:
        lap = session.laps[session.laps["Driver"] == drv].pick_fastest()
        if lap is None or pd.isna(lap["LapTime"]):
            continue
        tel = lap.get_telemetry()
        telemetrie[drv] = pd.DataFrame({
            "Distance": tel["Distance"].to_numpy(dtype=float),
            "sec": tel["Time"].dt.total_seconds().to_numpy(),
            "X": tel["X"].to_numpy(dtype=float),
            "Y": tel["Y"].to_numpy(dtype=float),
        })
    if len(telemetrie) < 2:
        return {"telemetrie": telemetrie, "edges": None, "gewinner": None}

    strecke = min(t["Distance"].max() for t in telemetrie.values())
    edges = np.linspace(0, strecke, n + 1)
    dauer = pd.DataFrame({
        drv: np.diff(np.interp(edges, t["Distance"], t["sec"]))
        for drv, t in telemetrie.items()
    })
    return {"telemetrie": telemetrie, "edges": edges,
           "gewinner": dauer.idxmin(axis=1).to_numpy(), "dauer": dauer}


# --------------------------------------------------------------- teamkollegen
def teammate_duels(session) -> list[dict]:
    """team-duelle einer session: schneller teamkollege gegen langsamer,
    aus quali-bestzeit (gueltig, nicht gestrichen) oder race pace
    (siehe P05).

    race-sessions nutzen :func:`pace_table` (bereinigt, treibstoffkorrigiert),
    alle anderen die schnellste gueltige runde je fahrer.
    """
    if session.name in ("Race", "Sprint"):
        pace = pace_table(session)
        if pace.empty:
            return []
        return _duelle(pace, "team", "driver", "median_s")

    laps = session.laps.pick_wo_box().pick_accurate()
    laps = laps[not_deleted_mask(laps["Deleted"]).to_numpy()]
    beste = (laps.groupby(["Team", "Driver"])["LapTime"].min()
            .dt.total_seconds().reset_index())
    return _duelle(beste, "Team", "Driver", "LapTime")


def _duelle(tab: pd.DataFrame, team_col: str, driver_col: str,
           wert_col: str) -> list[dict]:
    out = []
    for team, grp in tab.groupby(team_col):
        if len(grp) != 2:
            continue
        grp = grp.sort_values(wert_col)
        schnell, langsam = grp.iloc[0], grp.iloc[1]
        out.append({
            "team": team, "a": schnell[driver_col], "b": langsam[driver_col],
            "score_a": 1.0,
            "delta_pct": float((langsam[wert_col] / schnell[wert_col] - 1) * 100),
        })
    return out


# ------------------------------------------------------------------- wetter
def weather_join(session) -> pd.DataFrame:
    """gruene, gewertete, boxenlose runden mit dem naechstgelegenen
    wettermesspunkt verknuepft, plus treibstoffkorrigierte rundenzeit
    (siehe P17)."""
    laps = (session.laps.pick_wo_box().pick_accurate()
           .pick_track_status("1")).copy()
    wl = laps.get_weather_data().reset_index(drop=True)
    laps = laps.reset_index(drop=True)
    merged = pd.concat([laps, wl.loc[:, ~wl.columns.isin(laps.columns)]], axis=1)
    merged["sec"] = merged["LapTime"].dt.total_seconds()
    merged["corr"] = fuel_correct(
        merged["sec"], merged["LapNumber"], session.total_laps)
    return merged


def temperature_effect(merged: pd.DataFrame) -> dict:
    """streckentemperatur-effekt auf trockenen runden: erst die naive
    gepoolte regression, dann kontrolliert um fahrer-niveau und reifenalter
    (siehe P17, der effekt geht in der gepoolten fassung fast immer in der
    streuung durch fahrer/reifenalter unter).

    gibt ein leeres ergebnis (``n=0``) zurueck, wenn zu wenige trockene
    runden mit vollstaendigen werten vorliegen.
    """
    dry = merged[~merged["Rainfall"]].dropna(
        subset=["TrackTemp", "corr", "TyreLife"]).copy()
    if len(dry) < 20:
        return {"n": 0}

    naiv_slope, naiv_inter = np.polyfit(dry["TrackTemp"], dry["corr"], 1)
    naiv_pred = naiv_slope * dry["TrackTemp"] + naiv_inter
    naiv_r2 = 1 - ((dry["corr"] - naiv_pred) ** 2).sum() / \
        ((dry["corr"] - dry["corr"].mean()) ** 2).sum()

    dry["rel"] = dry["corr"] - dry.groupby("Driver")["corr"].transform("median")
    dry = dry[dry["rel"].abs() < 3].copy()
    y = dry["rel"].to_numpy()
    if len(dry) < 20:
        return {"n": 0}

    x_tyre = np.column_stack([dry["TyreLife"], np.ones(len(dry))])
    c_tyre, *_ = np.linalg.lstsq(x_tyre, y, rcond=None)
    r2_tyre = 1 - ((y - x_tyre @ c_tyre) ** 2).sum() / ((y - y.mean()) ** 2).sum()

    x_voll = np.column_stack([dry["TrackTemp"], dry["TyreLife"], np.ones(len(dry))])
    c_voll, *_ = np.linalg.lstsq(x_voll, y, rcond=None)
    pred_voll = x_voll @ c_voll
    r2_voll = 1 - ((y - pred_voll) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    resid = y - pred_voll
    sigma2 = (resid ** 2).sum() / (len(y) - 3)
    se = np.sqrt(np.diag(np.linalg.inv(x_voll.T @ x_voll)) * sigma2)

    # partial-residual-plot: TyreLife-anteil herausgerechnet, damit die
    # TrackTemp-wirkung isoliert sichtbar wird.
    dry["partial"] = y - c_voll[1] * dry["TyreLife"]

    return {
        "naiv_slope": naiv_slope, "naiv_r2": naiv_r2,
        "r2_tyre_only": r2_tyre, "r2_voll": r2_voll,
        "coef_temp": c_voll[0], "intercept": c_voll[2],
        "se_temp": se[0], "n": len(y), "dry": dry,
    }


def weather_phases(session) -> pd.DataFrame:
    """wetter-phasen ueber das Rainfall-flag segmentiert (siehe P17)."""
    w = session.weather_data
    gruppe = (w["Rainfall"] != w["Rainfall"].shift()).cumsum()
    return (w.groupby(gruppe)
            .agg(nass=("Rainfall", "first"), start=("Time", "min"),
                end=("Time", "max"))
            .reset_index(drop=True))


def wet_dry_classifier(session) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """logistische regression auf feld-aggregaten je runde (rundenzeit-
    streuung, mittlere speed-trap-geschwindigkeit), leave-one-out-
    kreuzvalidiert gegen die tatsaechlich mehrheitlich gefahrene mischung
    (siehe P17 AUSBAUSTUFE). keine compound-spalte als feature, nur was
    auch ohne boxenfunk beobachtbar waere.

    braucht mindestens eine runde je klasse (nass/trocken), sonst wirft
    ``LeaveOneOut`` einen fehler. das prueft der aufrufer.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import LeaveOneOut, cross_val_predict

    laps = session.laps.pick_accurate().copy()
    laps["sec"] = laps["LapTime"].dt.total_seconds()
    je_runde = laps.groupby("LapNumber").agg(
        std_sec=("sec", "std"), mean_speedFL=("SpeedFL", "mean"),
        n=("sec", "count")).dropna()
    je_runde = je_runde[je_runde["n"] >= 10]

    mehrheit = laps.groupby("LapNumber")["Compound"].agg(
        lambda s: s.value_counts().idxmax())
    je_runde["compound"] = mehrheit.reindex(je_runde.index)
    je_runde["nass"] = je_runde["compound"].isin(
        ["INTERMEDIATE", "WET"]).astype(int)

    X = je_runde[["std_sec", "mean_speedFL"]].to_numpy()
    y = je_runde["nass"].to_numpy()
    pred = cross_val_predict(LogisticRegression(), X, y, cv=LeaveOneOut())
    return je_runde, y, pred


# ------------------------------------------------------------- race control
def field_spread(session) -> pd.Series:
    """sekunden zwischen erstem und letztem fahrer je runde (siehe P18)."""
    laps = session.laps
    return (laps.dropna(subset=["Position"])
            .groupby("LapNumber")["Time"]
            .agg(lambda s: (s.max() - s.min()).total_seconds()))


def sc_compaction(neutral: pd.DataFrame, spread: pd.Series) -> pd.DataFrame:
    """baseline (letzte 3 gruene runden vor der phase) gegen die staerkste
    kompaktierung waehrend safety-car-/VSC-phasen (siehe P18, der mittelwert
    waere vom ausloese-zwischenfall verzerrt, deshalb das minimum statt dem
    durchschnitt waehrend der phase)."""
    zeilen = []
    for p in neutral.itertuples():
        vorher = spread.reindex(range(p.lap_start - 3, p.lap_start)).dropna()
        waehrend = spread.reindex(range(p.lap_start, p.lap_end + 1)).dropna()
        if vorher.empty or waehrend.empty:
            continue
        zeilen.append({
            "start": p.lap_start, "ende": p.lap_end,
            "baseline_s": vorher.median(), "minimum_s": waehrend.min(),
            "kompaktierung_pct": 100 * (1 - waehrend.min() / vorher.median()),
        })
    return pd.DataFrame(zeilen)


def sc_deployment_sectors(session) -> pd.DataFrame:
    """in welchem timing-sektor stand jeder fahrer im moment einer safety-
    car-deployment-meldung (siehe P18-erweiterung)? nutzt
    ``Sector1/2/3SessionTime`` (session-relativ, bislang ungenutzt) direkt
    gegen ``track_status['Time']`` (ebenfalls session-relativ), kein
    ``t0_date``/telemetrie noetig, anders als ein abgleich gegen die
    race-control-meldungszeit (die ist absolut datiert).

    fuer jeden fahrer wird die zum deployment-zeitpunkt laufende runde ueber
    das juengste ``LapStartTime`` vor dem zeitpunkt bestimmt, dann der
    zeitpunkt gegen die drei sektor-enden dieser runde eingeordnet.
    """
    ts = session.track_status
    deploy = ts.loc[ts["Status"] == "4", "Time"]
    if deploy.empty:
        return pd.DataFrame(columns=["time", "driver", "sector"])

    laps = session.laps.dropna(subset=["LapStartTime"]).sort_values("LapStartTime")
    zeilen = []
    for t in deploy:
        for drv, g in laps.groupby("Driver"):
            vor = g[g["LapStartTime"] <= t]
            if vor.empty:
                continue
            lap = vor.iloc[-1]
            if pd.isna(lap["Sector1SessionTime"]) or pd.isna(lap["Sector2SessionTime"]):
                sektor = None
            elif t <= lap["Sector1SessionTime"]:
                sektor = 1
            elif t <= lap["Sector2SessionTime"]:
                sektor = 2
            else:
                sektor = 3
            zeilen.append({"time": t, "driver": str(drv), "sector": sektor})
    return pd.DataFrame(zeilen)


# VORGEHEN 2 (P19): reale FIA-meldungen nennen strafmass und fahrer in
# umgekehrter reihenfolge zur naheliegenden annahme ("10 SECOND ... FOR CAR
# 14 (ALO)", nicht "CAR 14 (ALO) ... 10 SECOND").
PENALTY = re.compile(
    r"(\d+ SECOND (?:TIME|STOP/GO) PENALTY|DRIVE.?THROUGH PENALTY|REPRIMAND)"
    r" FOR CAR (\d+) \(([A-Z]{3})\)(?: - (.*))?", re.I)
TRACKLIM = re.compile(r"CAR (\d+) \(([A-Z]{3})\).*TRACK LIMITS AT TURN (\d+)", re.I)
# fuer die gegenpruefung zusaetzlich die im text genannte betroffene runde,
# NICHT dieselbe runde, in der die meldung gepostet wurde (die loeschung
# wird oft erst 1-2 runden spaeter verbucht). nicht jede meldung nennt sie
# explizit: "(NEXT LAP)"-faelle bleiben aussen vor.
TRACKLIM_RUNDE = re.compile(
    r"CAR (\d+) \(([A-Z]{3})\).*TRACK LIMITS AT TURN (\d+) LAP (\d+)", re.I)


def parse_penalties(rcm: pd.DataFrame) -> pd.DataFrame:
    """strafmeldungen der rennleitung parsen (siehe P19)."""
    zeilen = []
    for m in rcm.itertuples():
        treffer = PENALTY.search(str(m.Message))
        if treffer:
            zeilen.append({"lap": m.Lap, "strafmass": treffer.group(1).upper(),
                           "nr": treffer.group(2), "driver": treffer.group(3),
                           "grund": treffer.group(4)})
    return pd.DataFrame(zeilen)


def blue_flags(session, rcm: pd.DataFrame) -> pd.DataFrame:
    """blaue flaggen je fahrer, direkt aus den strukturierten spalten
    ``Category``/``Flag``/``RacingNumber`` statt aus freitext geparst.
    robuster als eine regex auf ``Message``, weil FastF1 fahrzeugnummer und
    flaggenfarbe hier schon getrennt mitliefert (siehe P19-erweiterung).
    eine blaue flagge ist kein vergehen, sondern die aufforderung, einen
    schnelleren (meist ueberrundenden) fahrer durchzulassen. viele blaue
    flaggen fuer denselben fahrer heissen deshalb "wurde oft ueberrundet",
    nicht "hat oft gestoert".
    """
    blau = rcm[(rcm["Category"] == "Flag") & (rcm["Flag"] == "BLUE")].copy()
    if blau.empty:
        return pd.DataFrame(columns=["time", "lap", "driver", "nr"])
    blau["driver"] = blau["RacingNumber"].map(
        lambda nr: session.get_driver(str(nr))["Abbreviation"])
    return (blau.rename(columns={"Time": "time", "Lap": "lap",
                                 "RacingNumber": "nr"})
            [["time", "lap", "driver", "nr"]].reset_index(drop=True))


def deleted_reason_crosscheck(session, rcm: pd.DataFrame) -> pd.DataFrame:
    """umgekehrte richtung zu :func:`track_limit_crosscheck` (siehe P19-
    erweiterung): laps mit ``Deleted=True`` und einem track-limits-grund in
    ``DeletedReason`` (bislang nirgends gelesen, nur in docstrings erwaehnt),
    die zu KEINER per regex geparsten race-control-meldung passen. deckt
    damit faelle auf, in denen der meldungs-regex etwas verpasst. die
    andere richtung als ``track_limit_crosscheck``, die FastF1s
    Deleted-spalte als unvollstaendig entlarvt.

    ``DeletedReason`` traegt dasselbe textformat wie die meldung selbst
    ("TRACK LIMITS AT TURN <n> LAP <m>") und wird hier direkt geparst,
    unabhaengig vom meldungstext.
    """
    laps = session.laps
    grund = laps["DeletedReason"].astype(str)
    treffer = grund.str.extract(r"TRACK LIMITS AT TURN (\d+) LAP (\d+)")
    maske = laps["Deleted"].astype(bool).to_numpy() & treffer[0].notna().to_numpy()
    if not maske.any():
        return pd.DataFrame(columns=["driver", "turn", "runde"])

    aus_laps = {(str(drv), int(turn), int(runde))
               for drv, turn, runde in zip(
                   laps.loc[maske, "Driver"], treffer.loc[maske, 0],
                   treffer.loc[maske, 1], strict=True)}
    aus_text = set()
    for m in rcm.itertuples():
        t = TRACKLIM_RUNDE.search(str(m.Message))
        if t:
            aus_text.add((t.group(2), int(t.group(3)), int(t.group(4))))

    fehlend = sorted(aus_laps - aus_text)
    return pd.DataFrame(fehlend, columns=["driver", "turn", "runde"])


def parse_track_limits(rcm: pd.DataFrame) -> pd.DataFrame:
    """track-limit-meldungen je fahrer und kurve parsen (siehe P19)."""
    zeilen = []
    for m in rcm.itertuples():
        treffer = TRACKLIM.search(str(m.Message))
        if treffer:
            zeilen.append({"lap": m.Lap, "nr": treffer.group(1),
                           "driver": treffer.group(2),
                           "turn": int(treffer.group(3))})
    return pd.DataFrame(zeilen)


def track_limit_crosscheck(
        session, rcm: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    """track-limit-meldungen gegen Laps.Deleted gegenpruefen (siehe P19).

    nutzt die im text genannte betroffene runde, nicht die runde, in der
    die meldung gepostet wurde. gibt (fehlend, deleted, n_mit_runde) zurueck:
    ``fehlend`` sind meldungen, die im text stehen, aber zu keiner
    Deleted=True-runde passen. FastF1s Deleted-spalte ist fuer
    track-limit-auswertungen leicht unvollstaendig (siehe docstring P19).
    """
    treffer = []
    for m in rcm.itertuples():
        t = TRACKLIM_RUNDE.search(str(m.Message))
        if t:
            treffer.append({"driver": t.group(2), "turn": int(t.group(3)),
                           "runde": int(t.group(4))})
    mit_runde = pd.DataFrame(treffer)

    deleted = session.laps[session.laps["Deleted"]]
    im_text_nicht_in_laps = []
    for r in mit_runde.itertuples():
        passt = ((deleted["Driver"] == r.driver)
                & (deleted["LapNumber"] == r.runde)).any()
        if not passt:
            im_text_nicht_in_laps.append({"driver": r.driver, "runde": r.runde})
    return pd.DataFrame(im_text_nicht_in_laps), deleted, len(mit_runde)
