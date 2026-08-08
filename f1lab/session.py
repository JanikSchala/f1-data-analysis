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
import pandas as pd

from .core import (
    Interval,
    bootstrap_median,
    elevation_profile,
    estimate_pit_loss,
    fit_degradation,
    fuel_correct,
    path_length,
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
              n_resamples: int = 1000) -> list[PaceEntry]:
    """Bereinigte Race Pace je Fahrer, sortiert vom Schnellsten.

    Jeder Eintrag traegt ein Bootstrap-Intervall. Ueberlappen sich zwei
    Intervalle, ist der Unterschied mit diesen Daten nicht belegbar - das
    ist bei benachbarten Fahrern regelmaessig der Fall.
    """
    laps = clean_laps(session, threshold).copy()
    laps["sec"] = laps["LapTime"].dt.total_seconds()

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


# --------------------------------------------------------------- Boxenstopps
def pit_loss(session) -> float:
    """Zeitverlust eines Boxenstopps auf dieser Strecke, in Sekunden.

    Vergleicht In- und Out-Laps mit der normalen Rundenzeit desselben
    Fahrers. Enthaelt damit Anfahrt, Standzeit und Ausfahrt.
    """
    laps = session.laps.copy()
    laps["sec"] = laps["LapTime"].dt.total_seconds()

    baseline = (clean_laps(session)
                .groupby("Driver")["LapTime"].median().dt.total_seconds())

    in_laps = laps[laps["PitInTime"].notna()]
    out_laps = laps[laps["PitOutTime"].notna()]

    return estimate_pit_loss(
        (in_laps["sec"] - in_laps["Driver"].map(baseline)).dropna(),
        (out_laps["sec"] - out_laps["Driver"].map(baseline)).dropna(),
    )


# --------------------------------------------------------------- Track Status
def track_status_phases(session) -> pd.DataFrame:
    """Safety-Car-, VSC- und Gelbphasen als Intervalle mit Rundennummern."""
    ts = session.track_status.copy()
    ts["label"] = ts["Status"].map(TRACK_STATUS).fillna(ts["Status"])
    ts["group"] = (ts["Status"] != ts["Status"].shift()).cumsum()

    phases = (ts.groupby(["group", "label"])
              .agg(start=("Time", "min"), end=("Time", "max"))
              .reset_index().drop(columns="group"))
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
