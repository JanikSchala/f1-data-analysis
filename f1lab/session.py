"""FastF1-Anbindung: Sessions laden, Runden filtern, Stints extrahieren.

Trennt die Datenbeschaffung von der Rechnung in :mod:`f1lab.core`. Alles hier
braucht Netzzugriff beim ersten Aufruf, danach den Cache.
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

# TrackStatus-Codes aus dem Live-Timing-Feed
TRACK_STATUS = {
    "1": "gruen", "2": "gelb", "4": "safety car",
    "5": "rot", "6": "vsc", "7": "vsc endet",
}

# FastF1 legt je Session einen Ordner mit ausgeschriebenem Namen an. Fuer die
# Bestandsaufnahme muss der Weg zurueck zur Kennung gehen, die get_session()
# erwartet. Sprint-Quali heisst je nach Jahr anders, landet aber auf "SQ".
SESSION_DIR_IDENT = {
    "Practice_1": "FP1", "Practice_2": "FP2", "Practice_3": "FP3",
    "Qualifying": "Q", "Sprint": "S", "Sprint_Qualifying": "SQ",
    "Sprint_Shootout": "SQ", "Race": "R",
}

# Ein Session-Ordner existiert schon, sobald FastF1 ihn einmal angefasst hat.
# Erst diese Datei belegt, dass Timing-Daten wirklich drin liegen; Telemetrie
# ist ein eigener, viel groesserer Download und fehlt oft.
TIMING_MARKER = "_extended_timing_data.ff1pkl"
TELEMETRY_MARKER = "car_data.ff1pkl"

# "2024-09-01_Italian_Grand_Prix" -> Datum + Name
_EVENT_DIR = re.compile(r"^(\d{4}-\d{2}-\d{2})_(.+)$")

_active_cache: Path | None = None


def find_cache(path: Path | str | None = None) -> Path | None:
    """Ersten vorhandenen Cache-Ordner suchen, ohne einen anzulegen.

    Reihenfolge: explizites Argument, Umgebungsvariable ``F1_CACHE``, der
    Standardpfad ``~/f1_cache``, und zuletzt ein ``f1_cache`` neben dem
    Repository. Der letzte Fall deckt die Ablage ab, in der Repo und Cache
    Geschwister in einem Projektordner sind.

    Returns:
        Pfad oder None, wenn nirgends ein Cache liegt. Ein fehlender Cache
        ist kein Fehler - er ist der Zustand vor dem ersten Warmup.
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
    """Cache aktivieren. Ohne Cache dauert jede Session-Ladung Minuten.

    Args:
        path: Zielordner. Ohne Angabe der Standardpfad ``~/f1_cache``.
        offline: Schaltet jeden Netzzugriff ab. Sessions, die nicht im Cache
            liegen, scheitern dann sofort, statt minutenlang zu laden und ins
            Rate-Limit zu laufen - das ist fuer eine Oberflaeche das
            gewuenschte Verhalten.
    """
    global _active_cache
    p = Path(path) if path else CACHE_DIR
    p.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(str(p))
    fastf1.Cache.offline_mode(offline)
    _active_cache = p
    return p


def cached_sessions(cache_dir: Path | str | None = None) -> pd.DataFrame:
    """Bestandsaufnahme des Caches, allein aus der Ordnerstruktur.

    Liest den Cache so, wie FastF1 ihn anlegt
    (``<cache>/<Jahr>/<Datum>_<Event>/<Datum>_<Session>/``), und braucht dafuer
    weder Netz noch einen Session-Download. Damit laesst sich eine Auswahl
    anbieten, die nur zeigt, was auch wirklich auswertbar ist.

    Returns:
        Ein Datensatz je Session mit Saison, Event, Datum, Kennung und zwei
        Flags: ``timing`` fuer Rundendaten, ``telemetry`` fuer den
        Positions- und Fahrzeugkanal. Leerer Rahmen, wenn kein Cache da ist.
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
    """Session laden und cachen.

    Args:
        year: Saison, Telemetrie gibt es erst ab 2018.
        gp: Name ("Monza"), Land ("Italy") oder Rundennummer (14).
        identifier: FP1 FP2 FP3 Q S SQ R.
        telemetry: Nur einschalten, wenn wirklich gebraucht - der Download
            ist um ein Vielfaches groesser.

    Ein zuvor gesetzter Cache bleibt erhalten: wer :func:`enable_cache` mit
    eigenem Pfad oder ``offline=True`` aufgerufen hat, soll das hier nicht
    stillschweigend zurueckgesetzt bekommen.
    """
    if _active_cache is None:
        enable_cache()
    ses = fastf1.get_session(year, gp, identifier)
    ses.load(telemetry=telemetry, weather=weather, messages=messages)
    return ses


def not_deleted_mask(deleted) -> pd.Series:
    """True fuer Runden, die NICHT von der Rennleitung gestrichen wurden.

    Die Spalte ``Deleted`` ist ein nullable Boolean. Neben True und False
    steht dort None, wenn zu einer Runde nichts gemeldet wurde - was der
    Normalfall ist. FastF1s ``pick_not_deleted()`` invertiert die Spalte
    direkt mit ``~``, und genau daran scheitert pandas bei object-dtype:

        TypeError: bad operand type for unary ~

    Deswegen hier explizit: fehlende Angabe heisst "nicht gestrichen".
    """
    s = pd.Series(deleted)
    if s.empty:
        return s.astype(bool)
    # .where statt .fillna: fillna auf object-dtype loest in pandas 2.2
    # eine Downcasting-Warnung aus, die in pandas 3 zum Verhaltenswechsel
    # wird. .where ist in beiden Versionen eindeutig.
    return ~s.where(s.notna(), False).astype(bool)


def clean_laps(session, threshold: float = 1.07) -> pd.DataFrame:
    """Runden, auf denen sich Pace-Aussagen aufbauen lassen.

    Entfernt in dieser Reihenfolge:
      - Boxenrunden (In- und Out-Lap sind keine Renn-Runden)
      - unplausible Runden (FastF1 markiert sie ueber IsAccurate)
      - alles ausser gruener Flagge (Safety Car verschiebt den Median
        um mehrere Zehntel)
      - von der Rennleitung gestrichene Runden
      - Ausreisser ueber threshold mal Bestzeit
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
    """Bereinigte, treibstoffkorrigierte Race Pace je Fahrer, sortiert vom
    Schnellsten.

    Jeder Eintrag traegt ein Bootstrap-Intervall. Ueberlappen sich zwei
    Intervalle, ist der Unterschied mit diesen Daten nicht belegbar - das
    ist bei benachbarten Fahrern regelmaessig der Fall.

    Ohne :func:`fuel_correct` haengt der Median zusaetzlich davon ab, *wann*
    im Rennen die sauberen Runden eines Fahrers liegen - zwei Fahrer mit
    identischem Tempo koennten sonst unterschiedlich abschneiden, nur weil
    eine Safety-Car-Phase dem einen mehr fruehe (schwere), dem anderen mehr
    spaete (leichte) Runden uebrig laesst. ``kg_per_lap=0`` schaltet die
    Korrektur ab (fuer den Vergleich in P04).
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
    """race_pace() als DataFrame mit Delta zum Schnellsten."""
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
    """Ein Datensatz je Stint: Fahrer, Compound, Start, Ende, Laenge."""
    df = (session.laps
          .groupby(["Driver", "Stint", "Compound"], dropna=False)["LapNumber"]
          .agg(start="min", end="max", laps="count")
          .reset_index())
    df["stint"] = df["Stint"].astype("Int64")
    return df.drop(columns="Stint").sort_values(["Driver", "start"])


def degradation(session, threshold: float = 1.10,
                min_laps: int = 6) -> pd.DataFrame:
    """Degradation je Stint, mit herausgerechnetem Treibstoffeffekt.

    Ohne die Korrektur wird die Degradation systematisch unterschaetzt: das
    Auto wird leichter, waehrend der Reifen abbaut, und beide Effekte heben
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
            # FreshTyre (siehe P13-Erweiterung, bislang nirgends gelesen):
            # ein wiederverwendeter Satz hat Vorverschleiss, den TyreLife
            # allein nicht abbildet - konstant innerhalb eines Stints,
            # deshalb der Wert der ersten Runde.
            "fresh": bool(g["FreshTyre"].iloc[0]),
            "laps": fit.n,
            "deg_s_per_lap": round(fit.slope, 4),
            "base_s": round(fit.intercept, 3),
            "r2": round(fit.r2, 3),
            "reliable": fit.is_reliable,
        })
    return pd.DataFrame(rows).sort_values("deg_s_per_lap", ignore_index=True)


def degradation_by_compound(session, **kwargs) -> pd.DataFrame:
    """Mittlere Degradation je Reifenmischung, nur belastbare Fits."""
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
    """RaceConfig fuer den Strategie-Optimierer (P35) aus echter Degradation
    (P13) und echtem Pitloss dieser Session, statt Parameter von Hand zu
    setzen.

    Je Mischung: Median von Steigung und Achsenabschnitt ueber alle
    belastbaren Stint-Fits (``DegradationFit.is_reliable``), ``max_age`` aus
    der laengsten tatsaechlich gefahrenen Stint-Laenge dieser Mischung - eine
    vorsichtige Untergrenze, kein gemessenes Limit: niemand testet in einem
    echten Rennen, wie lange ein Reifen wirklich haelt.

    ``base_time`` ergibt sich aus ``fit.intercept + fit.slope``: der Fit
    extrapoliert auf Reifenalter 0 (``intercept``), TyreLife ist bei FastF1
    aber einsbasiert - die erste echte Runde auf dem Satz hat TyreLife 1,
    also ``intercept + 1 * slope``.

    Raises:
        ValueError: keine belastbaren Fits, oder (mit
            ``require_two_compounds``) nur eine Mischung belastbar.
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
    """Zeitverlust eines Boxenstopps auf dieser Strecke, in Sekunden.

    Vergleicht In- und Out-Laps mit der normalen Rundenzeit desselben
    Fahrers. Enthaelt damit Anfahrt, Standzeit und Ausfahrt.

    Runden unter rotem Flag ausgeschlossen: ``PitInTime``/``PitOutTime``
    stehen dann oft auf Runden, in denen Autos waehrend der
    Session-Unterbrechung geparkt bzw. wieder losgeschickt werden - keine
    echten Boxenstopps, aber mit Rundenzeiten von zig Minuten (Monaco 2024 R,
    roter Start nach der Startunfall-Massenkarambolage: ein einzelner
    solcher Fall zog den Median auf ueber 2400 Sekunden). ``TrackStatus``
    traegt bei FastF1 alle waehrend der Runde durchlaufenen Zustaende als
    Zeichenkette (z.B. "1254") - "5" irgendwo darin heisst rotes Flag zu
    irgendeinem Zeitpunkt in dieser Runde.
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
    """Safety-Car-, VSC- und Gelbphasen als Intervalle mit Rundennummern.

    ``session.track_status`` ist ein Log von Zustandswechseln, keine
    Zeitreihe: "SCDeployed" steht dort typischerweise genau einmal, auch wenn
    das Safety Car neun Runden lang draussen bleibt. Ein Intervall muss
    deshalb bis zum naechsten Wechsel reichen, nicht nur bis zum letzten
    Auftreten desselben Codes (siehe :func:`f1lab.core.status_intervals` -
    die urspruengliche Fassung gruppierte nach gleichem Code und ergab so
    fast durchweg Intervalle der Laenge 0, siehe P18).
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


# --------------------------------------------------------------- Dimensionen
# FastF1 liefert Positionen in Zehntelmetern (siehe core.py-Doku zu X/Y/Z).
POS_UNITS_PER_M = 10.0

EVENT_DIM_COLS = ["season", "round", "event_name", "official_name", "country",
                  "location", "event_date", "event_format", "is_sprint",
                  "n_sessions"]


def event_dimension(years) -> pd.DataFrame:
    """Rennkalender mehrerer Saisons als Dimensionstabelle.

    Eine Zeile je Rennwochenende, Jahre untereinander. Saisons, die die API
    nicht kennt, werden uebersprungen statt den Aufbau abzubrechen - bei der
    laufenden Saison steht der Kalender teils erst spaet.

    Das Sprint-Format heisst je nach Jahr anders ("sprint",
    "sprint_shootout", "sprint_qualifying"), deshalb wird auf das Teilwort
    geprueft und nicht auf Gleichheit.
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
    """Kurvenzahl, Streckenlaenge und Hoehenprofil einer geladenen Session.

    Braucht Telemetrie: die Kurvenliste kommt zwar aus der MultiViewer-API,
    Laenge und Hoehe aber aus dem Positionskanal der schnellsten Runde.
    Deshalb ist das hier bewusst getrennt von :func:`event_dimension`, die
    ohne jeden Session-Download auskommt.

    Gemessen wird die *gefahrene Linie*, nicht die offizielle Streckenlaenge -
    die Ideallinie schneidet Kurven und faellt dadurch typisch ein bis zwei
    Prozent kuerzer aus als die Angabe im Reglement.
    """
    out = {"corners": pd.NA, "length_m": pd.NA,
           "elev_gain_m": pd.NA, "elev_span_m": pd.NA}

    # Kurvenliste und Positionsdaten kommen aus verschiedenen Quellen und
    # fallen unabhaengig voneinander aus - deshalb einzeln abgesichert.
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
    """Circuit-Dimension aus einer Liste von (Jahr, GP)-Paaren.

    Streckengeometrie ist pro Layout konstant, es genuegt also *eine* Session
    je Strecke - nicht das ganze Archiv. Sessions, deren Telemetrie nicht im
    Cache liegt, werden mit leeren Kennzahlen zurueckgegeben statt den Aufbau
    abzubrechen; die Zeile bleibt erhalten, damit sichtbar ist, was fehlt.
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
    """Distanz, Streckenkruemmung und echte Geschwindigkeit der schnellsten
    Runde - Eingabe fuer die Rundenzeit-Simulation (siehe P37,
    :func:`f1lab.core.simulate_lap`/:func:`calibrate_lap_model`).

    Returns:
        (Distanz [m], Kruemmung [1/m], Geschwindigkeit [m/s])
    """
    lap = session.laps.pick_fastest()
    tel = lap.get_telemetry().add_distance()
    dist = tel["Distance"].to_numpy(dtype=float)
    kappa = track_curvature(tel["X"], tel["Y"], dist)
    speed_ms = tel["Speed"].to_numpy(dtype=float) / 3.6
    return dist, kappa, speed_ms


def corner_labels(session) -> pd.DataFrame:
    """Kurven der Session, auf die Distanz einer Referenzrunde projiziert.

    Die XY-Position jeder Kurve (aus get_circuit_info(), unabhaengig von
    jedem Fahrer) wird auf den naechstgelegenen Punkt der schnellsten Runde
    der Session projiziert - das gibt jeder Kurve eine Distanz entlang der
    Strecke, mit der sich Telemetrie faehrerbergreifend vergleichen laesst.
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
    """Marshal-Sektorgrenzen, auf dieselbe Referenzrunde projiziert wie
    :func:`corner_labels` - dieselbe Idee, andere Punktliste (siehe P11).
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
    """Positionen der Gelbflaggen-Lichttafeln, auf dieselbe Referenzrunde
    projiziert wie :func:`corner_labels`/:func:`marshal_sector_labels` -
    dieselbe Idee, dritte Punktliste. ``CircuitInfo.marshal_lights`` liefert
    kein ``Distance`` (anders als ``corners``), deshalb dieselbe
    naechste-Nachbar-Projektion wie bei den anderen beiden."""
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
    """Minimalgeschwindigkeit je Kurve und Fahrer (siehe P11/P12).

    Kurven kommen aus :func:`corner_labels` (auf eine Referenzrunde
    projiziert); je Fahrer wird die Minimalgeschwindigkeit in einem Fenster
    von +/- window_m um die Kurvendistanz gesucht - nicht per exakter
    naechster-Nachbar-Zuordnung, weil benachbarte Kurven sich sonst
    gegenseitig Telemetriepunkte wegnehmen koennten.

    Returns:
        DataFrame Fahrer x Kurve (Kuerzel wie "T7"), NaN wo kein
        Telemetriepunkt im Fenster liegt.
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


# --------------------------------------------------------------- Bremszonen
def driver_braking_zones(session, driver: str, min_length_m: float = 20.0
                         ) -> pd.DataFrame:
    """Bremszonen der schnellsten Runde eines Fahrers (siehe P08).

    Braucht Telemetrie. Leerer Rahmen, wenn der Fahrer keine gewertete
    schnellste Runde hat (z.B. nach einem Ausfall vor der ersten Runde).
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
    """Bremszonen zweier Fahrer paaren und den Abstand ihrer Bremspunkte
    zeigen (siehe P07/P08). Nutzt :func:`f1lab.core.match_by_distance`.
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
    """DRS-Aktivzonen der schnellsten Runde eines Fahrers (siehe P10).

    Filtert alles unter ``min_length_m`` als Rauschen - ohne den Filter
    meldet dieselbe Flankenlogik am Start/Ziel-Bereich mehrere kurze
    Wackel-Zonen, Rest-Aktivierung vom Ende der Vorrunde.
    """
    lap = session.laps.pick_drivers(driver).pick_fastest()
    if lap is None or pd.isna(lap["LapTime"]):
        return pd.DataFrame()
    car = lap.get_car_data().add_distance()
    offen = drs_state(car["DRS"].to_numpy()) == 2
    return pd.DataFrame(active_distance_zones(offen, car["Distance"].to_numpy(),
                                              min_length_m=min_length_m))


def drs_usage(session) -> pd.DataFrame:
    """DRS-Zeitanteil und Topspeed-Gewinn je Fahrer (siehe P10 VORGEHEN 1/3/4)."""
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


# --------------------------------------------------------------- Position
def position_progression(session) -> pd.DataFrame:
    """Position je Runde und Fahrer, pivotiert (siehe P20 VORGEHEN 1)."""
    return session.laps.pivot_table(index="LapNumber", columns="Driver",
                                    values="Position", aggfunc="first")


def overtake_events(session) -> pd.DataFrame:
    """Einzelne Ueberholvorgaenge (Fahrer A ueberholt Fahrer B in Runde N),
    ohne Boxenstopp-Effekt und nur auf gruener Flagge (siehe P20 VORGEHEN
    3/4). :func:`overtakes_matrix` aggregiert genau diese Liste zu einer
    Matrix, :func:`overtake_locations` (P39) sucht je Ereignis die Stelle
    auf der Strecke.

    Zwei Faelle zaehlen nicht als echtes Duell: ein Boxenstopp verschiebt
    die Position ohne Ueberholen auf der Strecke, und ein
    Safety-Car-Restart wirbelt das Feld durcheinander, ohne dass Position
    durch Tempo gewonnen wurde. Die Positionstabelle selbst bleibt ueber
    die volle, ungefilterte Rundenliste - sonst fehlen Rundennummern in der
    Reihe, sobald eine Runde komplett herausfaellt, und lap - 1 zeigt ins
    Leere.

    Returns:
        DataFrame mit Spalten ``gainer``, ``loser``, ``lap``.
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
    """Wer ueberholt wen wie oft (siehe P20 VORGEHEN 3/4). Zeile ueberholt
    Spalte. Aggregiert :func:`overtake_events`."""
    laps = session.laps
    drivers = list(laps.pivot_table(index="LapNumber", columns="Driver",
                                    values="Position", aggfunc="first").columns)
    mat = pd.DataFrame(0, index=drivers, columns=drivers)
    for e in overtake_events(session).itertuples():
        mat.loc[e.gainer, e.loser] += 1
    return mat


def overtake_locations(session, drs_session=None, drs_referenz: str | None = None,
                       naehe_m: float = 30.0) -> pd.DataFrame:
    """Ort jedes Ueberholvorgangs auf der Strecke, gegen die DRS-Zonen
    geprueft (siehe P39).

    Fuer jedes Ereignis aus :func:`overtake_events`: in der Telemetrie des
    Ueberholers (``gainer``) fuer genau diese Runde die letzte Stelle
    suchen, an der ``DriverAhead`` (FastF1s eigener, GPS-basierter
    "Auto direkt davor"-Kanal) noch dem Ueberholten (``loser``) entspricht -
    das ist die Stelle kurz vor dem Positionswechsel. Nur uebernommen, wenn
    dort auch ``DistanceToDriverAhead`` unter ``naehe_m`` liegt (sonst war
    ``loser`` zwar irgendwann auf der Runde davor, aber nicht mehr in dem
    Moment, der als Wechsel gezaehlt wird - z.B. bei mehreren
    Positionswechseln in derselben Runde).

    Die DRS-Zonen kommen bewusst NICHT aus der Renn-Session selbst: DRS
    braucht in einem Rennen einen Rueckstand unter 1s auf das Auto davor,
    eine schnellste Rennrunde entsteht aber typisch in freier Fahrt ohne
    Vordermann - dann bleibt DRS auf der ganzen Runde zu, und
    :func:`drs_zones` faende keine einzige Zone, obwohl die Zonen
    physisch existieren (siehe P39 fuer den Monza-Fall, an dem das
    auffiel). Im Qualifying ist DRS ohne Abstandsregel verfuegbar, deshalb
    Default: die Qualifying-Session desselben Events (ueber die
    Rundennummer, nicht den Streckennamen - robuster gegen
    Schreibweisen). ``drs_session`` laesst sich trotzdem explizit setzen,
    falls keine Qualifying-Telemetrie im Cache liegt.

    Returns:
        DataFrame mit ``gainer``, ``loser``, ``lap``, ``distance_m``,
        ``in_drs_zone``. Ereignisse ohne verwertbare Telemetriestelle
        fehlen in der Rueckgabe (nicht als Zeile mit NaN) - die Differenz
        zu :func:`overtake_events` ist die Abdeckung dieser Methode.
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
    """Echte, paarweise Undercut-Versuche und ob sie gelangen (siehe P42).

    Anders als eine flaechendeckende Vorher-Nachher-Positionszaehlung (die
    JEDEN Boxenstopp automatisch als Verlust gegen das ganze Feld zaehlt,
    weil nicht-stoppende Autos in der Zwischenzeit einfach weiterfahren -
    siehe CLAUDE.md, verworfener erster Versuch) vergleicht das hier gezielt
    gegen einen konkreten Rivalen: fuer jeden Boxenstopp (Fahrer A, Runde L)
    der Fahrer, der zu Rundenbeginn genau eine Position vor A lag (der
    eigentliche Gegner des Stopps) - nur gezaehlt, wenn dieser Rivale nicht
    selbst in derselben Runde stoppt (sonst kein Undercut-Versuch) und
    innerhalb von ``fenster`` Runden danach selbst an die Box faehrt (sonst
    ist die Verfolgung kein Undercut, sondern nur zufaellig zeitversetzte
    Stopps). Erfolg heisst: ``nachlauf`` Runden nach dem spaeteren der
    beiden Stopps liegt A vor dem Rivalen.

    Args:
        fenster: wie viele Runden nach As Stopp der Rivale noch selbst
            stoppen darf, damit es als Undercut-Versuch zaehlt.
        nachlauf: wie viele gruene Runden nach dem letzten der beiden
            Stopps abgewartet wird, bevor die Position verglichen wird.

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


# --------------------------------------------------------------- Start
def _zeit_bei_speed(t: np.ndarray, v: np.ndarray, ziel: float, t0: float
                    ) -> float | None:
    """Erste Zeit (relativ zu t0), zu der v mindestens ziel erreicht."""
    mask = v >= ziel
    return round(float(t[mask.argmax()] - t0), 2) if mask.any() else None


def start_performance(session, fenster_s: float = 8.0) -> pd.DataFrame:
    """Startkennzahlen je Fahrer: Zeit bis 100/200 km/h, Distanz nach 5s,
    Positionsgewinn Grid -> Ende Runde 1 (siehe P31).

    Boxenstarts (PitOutTime auf Runde 1 gesetzt) werden ausgeschlossen - ein
    Start aus der Box hat eine komplett andere Ausgangsgeschwindigkeit
    (Boxengassen-Limit statt Ampel-Start) und ist nicht vergleichbar.

    Liest ``session.results`` fuer die Startaufstellung - das kann fuer
    manche Saisons echten Ergast/jolpica-Netzzugriff ausloesen, obwohl es
    wie eine reine Lokaldaten-Spalte aussieht (siehe P40, dort in einem
    Saison-Scan entdeckt). Fuer eine einzelne Session unkritisch, bei vielen
    Sessions hintereinander siehe dortige Drosselung.
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
    """Startplatz gegen Position am Ende von Runde 1, ohne Boxenstarts
    (siehe P40 - die telemetriefreie Variante von P31s Startkennzahlen,
    fuer Saison-Scans ueber viele Rennen ohne Telemetrie-Download).

    Returns:
        DataFrame mit ``driver_number``, ``grid``, ``lap1``, ``gewinn``
        (grid - lap1, positiv = Positionen gewonnen).
    """
    grid = session.results[["DriverNumber", "GridPosition"]].dropna()
    lap1 = session.laps.pick_laps(1)
    lap1 = lap1.loc[lap1["PitOutTime"].isna(), ["DriverNumber", "Position"]].dropna()
    m = grid.merge(lap1, on="DriverNumber").rename(
        columns={"DriverNumber": "driver_number", "GridPosition": "grid",
                "Position": "lap1"})
    m["gewinn"] = m["grid"] - m["lap1"]
    return m


# --------------------------------------------------------------- Verfolgung
def close_following(session, driver: str, nah_schwelle_m: float = 50.0
                    ) -> pd.DataFrame:
    """Abstand zum Vordermann je gruener Runde, treibstoffkorrigiert
    (siehe P32 VORGEHEN 1-2).

    add_driver_ahead() laeuft einmal auf die gesamte Renntelemetrie des
    Fahrers, nicht einmal je Runde - um ein Vielfaches schneller bei
    identischem Ergebnis (siehe P05/P20/P32).
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
    """Rundenzeit (bereits treibstoffkorrigiert) gegen Nahanteil regressieren,
    nach Herausrechnen der Reifendegradation (siehe P32).

    Nutzt :func:`fit_degradation` fuer die Degradations-Bereinigung - dieselbe
    Funktion wie in P13/Dashboard, nur hier auf Nahanteil statt Compound
    angewendet.

    Args:
        df: Ergebnis von :func:`close_following`.

    Returns:
        (slope, intercept, r2, df mit zusaetzlicher Spalte ``sec_corr``).
        slope/intercept/r2 sind NaN, wenn zu wenige Runden vorliegen.
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


# --------------------------------------------------------------- Mini-Sektoren
def mini_sectors(session, drivers: list[str], n: int = 25) -> dict:
    """Zerlegt die Runde in n gleich lange Distanz-Abschnitte und ermittelt
    je Abschnitt, welcher der uebergebenen Fahrer dort am wenigsten Zeit
    gebraucht hat (siehe P06).

    Nur wenige Fahrer uebergeben (MAX_SERIEN aus f1lab.design) - bei allen
    20 waere weder die Farbdarstellung noch die Frage "wer dominiert wo"
    sinnvoll, siehe P06-Docstring.

    Returns:
        dict mit ``telemetrie`` (Distanz/Zeit/X/Y je Fahrer),
        ``edges`` (Grenzen der Abschnitte) und ``gewinner`` (Fahrer je
        Abschnitt, laenge n).
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


# --------------------------------------------------------------- Teamkollegen
def teammate_duels(session) -> list[dict]:
    """Team-Duelle einer Session: schneller Teamkollege gegen langsamer,
    aus Quali-Bestzeit (gueltig, nicht gestrichen) oder Race Pace
    (siehe P05).

    Race-Sessions nutzen :func:`pace_table` (bereinigt, treibstoffkorrigiert),
    alle anderen die schnellste gueltige Runde je Fahrer.
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


# ------------------------------------------------------------------- Wetter
def weather_join(session) -> pd.DataFrame:
    """Gruene, gewertete, boxenlose Runden mit dem naechstgelegenen
    Wettermesspunkt verknuepft, plus treibstoffkorrigierte Rundenzeit
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
    """Streckentemperatur-Effekt auf trockenen Runden: erst die naive
    gepoolte Regression, dann kontrolliert um Fahrer-Niveau und Reifenalter
    (siehe P17 - der Effekt geht in der gepoolten Fassung fast immer in der
    Streuung durch Fahrer/Reifenalter unter).

    Gibt ein leeres Ergebnis (``n=0``) zurueck, wenn zu wenige trockene
    Runden mit vollstaendigen Werten vorliegen.
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

    # Partial-Residual-Plot: TyreLife-Anteil herausgerechnet, damit die
    # TrackTemp-Wirkung isoliert sichtbar wird.
    dry["partial"] = y - c_voll[1] * dry["TyreLife"]

    return {
        "naiv_slope": naiv_slope, "naiv_r2": naiv_r2,
        "r2_tyre_only": r2_tyre, "r2_voll": r2_voll,
        "coef_temp": c_voll[0], "intercept": c_voll[2],
        "se_temp": se[0], "n": len(y), "dry": dry,
    }


def weather_phases(session) -> pd.DataFrame:
    """Wetter-Phasen ueber das Rainfall-Flag segmentiert (siehe P17)."""
    w = session.weather_data
    gruppe = (w["Rainfall"] != w["Rainfall"].shift()).cumsum()
    return (w.groupby(gruppe)
            .agg(nass=("Rainfall", "first"), start=("Time", "min"),
                end=("Time", "max"))
            .reset_index(drop=True))


def wet_dry_classifier(session) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Logistische Regression auf Feld-Aggregaten je Runde (Rundenzeit-
    Streuung, mittlere Speed-Trap-Geschwindigkeit), Leave-one-out-
    kreuzvalidiert gegen die tatsaechlich mehrheitlich gefahrene Mischung
    (siehe P17 AUSBAUSTUFE). Keine Compound-Spalte als Feature - nur was
    auch ohne Boxenfunk beobachtbar waere.

    Braucht mindestens eine Runde je Klasse (nass/trocken), sonst wirft
    ``LeaveOneOut`` einen Fehler - das prueft der Aufrufer.
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


# ------------------------------------------------------------- Race Control
def field_spread(session) -> pd.Series:
    """Sekunden zwischen erstem und letztem Fahrer je Runde (siehe P18)."""
    laps = session.laps
    return (laps.dropna(subset=["Position"])
            .groupby("LapNumber")["Time"]
            .agg(lambda s: (s.max() - s.min()).total_seconds()))


def sc_compaction(neutral: pd.DataFrame, spread: pd.Series) -> pd.DataFrame:
    """Baseline (letzte 3 gruene Runden vor der Phase) gegen die staerkste
    Kompaktierung waehrend Safety-Car-/VSC-Phasen (siehe P18 - der Mittelwert
    waere vom Ausloese-Zwischenfall verzerrt, deshalb das Minimum statt dem
    Durchschnitt waehrend der Phase)."""
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
    """In welchem Timing-Sektor stand jeder Fahrer im Moment einer Safety-
    Car-Deployment-Meldung (siehe P18-Erweiterung)? Nutzt
    ``Sector1/2/3SessionTime`` (session-relativ, bislang ungenutzt) direkt
    gegen ``track_status['Time']`` (ebenfalls session-relativ) - kein
    ``t0_date``/Telemetrie noetig, anders als ein Abgleich gegen die
    Race-Control-Meldungszeit (die ist absolut datiert).

    Fuer jeden Fahrer wird die zum Deployment-Zeitpunkt laufende Runde ueber
    das juengste ``LapStartTime`` vor dem Zeitpunkt bestimmt, dann der
    Zeitpunkt gegen die drei Sektor-Enden dieser Runde eingeordnet.
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


# VORGEHEN 2 (P19): reale FIA-Meldungen nennen Strafmass und Fahrer in
# umgekehrter Reihenfolge zur naheliegenden Annahme ("10 SECOND ... FOR CAR
# 14 (ALO)", nicht "CAR 14 (ALO) ... 10 SECOND") - 49/49 Treffer Saison 2024.
PENALTY = re.compile(
    r"(\d+ SECOND (?:TIME|STOP/GO) PENALTY|DRIVE.?THROUGH PENALTY|REPRIMAND)"
    r" FOR CAR (\d+) \(([A-Z]{3})\)(?: - (.*))?", re.I)
TRACKLIM = re.compile(r"CAR (\d+) \(([A-Z]{3})\).*TRACK LIMITS AT TURN (\d+)", re.I)
# Fuer die Gegenpruefung zusaetzlich die im Text genannte betroffene Runde -
# NICHT dieselbe Runde, in der die Meldung gepostet wurde (die Loeschung
# wird oft erst 1-2 Runden spaeter verbucht). Nicht jede Meldung nennt sie
# explizit: "(NEXT LAP)"-Faelle bleiben aussen vor.
TRACKLIM_RUNDE = re.compile(
    r"CAR (\d+) \(([A-Z]{3})\).*TRACK LIMITS AT TURN (\d+) LAP (\d+)", re.I)


def parse_penalties(rcm: pd.DataFrame) -> pd.DataFrame:
    """Strafmeldungen der Rennleitung parsen (siehe P19)."""
    zeilen = []
    for m in rcm.itertuples():
        treffer = PENALTY.search(str(m.Message))
        if treffer:
            zeilen.append({"lap": m.Lap, "strafmass": treffer.group(1).upper(),
                           "nr": treffer.group(2), "driver": treffer.group(3),
                           "grund": treffer.group(4)})
    return pd.DataFrame(zeilen)


def blue_flags(session, rcm: pd.DataFrame) -> pd.DataFrame:
    """Blaue Flaggen je Fahrer, direkt aus den strukturierten Spalten
    ``Category``/``Flag``/``RacingNumber`` statt aus Freitext geparst -
    robuster als eine Regex auf ``Message``, weil FastF1 Fahrzeugnummer und
    Flaggenfarbe hier schon getrennt mitliefert (siehe P19-Erweiterung).
    Eine blaue Flagge ist kein Vergehen, sondern die Aufforderung, einen
    schnelleren (meist ueberrundenden) Fahrer durchzulassen - viele blaue
    Flaggen fuer denselben Fahrer heissen deshalb "wurde oft ueberrundet",
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
    """Umgekehrte Richtung zu :func:`track_limit_crosscheck` (siehe P19-
    Erweiterung): Laps mit ``Deleted=True`` und einem Track-Limits-Grund in
    ``DeletedReason`` (bislang nirgends gelesen, nur in Docstrings erwaehnt),
    die zu KEINER per Regex geparsten Race-Control-Meldung passen. Deckt
    damit Faelle auf, in denen der Meldungs-Regex etwas verpasst - die
    andere Richtung als ``track_limit_crosscheck``, die FastF1s
    Deleted-Spalte als unvollstaendig entlarvt.

    ``DeletedReason`` traegt dasselbe Textformat wie die Meldung selbst
    ("TRACK LIMITS AT TURN <n> LAP <m>") und wird hier direkt geparst,
    unabhaengig vom Meldungstext.
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
    """Track-Limit-Meldungen je Fahrer und Kurve parsen (siehe P19)."""
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
    """Track-Limit-Meldungen gegen Laps.Deleted gegenpruefen (siehe P19).

    Nutzt die im Text genannte betroffene Runde, nicht die Runde, in der
    die Meldung gepostet wurde. Gibt (fehlend, deleted, n_mit_runde) zurueck:
    ``fehlend`` sind Meldungen, die im Text stehen, aber zu keiner
    Deleted=True-Runde passen - FastF1s Deleted-Spalte ist fuer
    Track-Limit-Auswertungen leicht unvollstaendig (siehe Docstring P19).
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
