"""FastF1-Anbindung: Sessions laden, Runden filtern, Stints extrahieren.

Trennt die Datenbeschaffung von der Rechnung in :mod:`f1lab.core`. Alles hier
braucht Netzzugriff beim ersten Aufruf, danach den Cache.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

import fastf1

from .core import (
    DegradationFit,
    Interval,
    bootstrap_median,
    estimate_pit_loss,
    fit_degradation,
    fuel_correct,
)

CACHE_DIR = Path.home() / "f1_cache"

# TrackStatus-Codes aus dem Live-Timing-Feed
TRACK_STATUS = {
    "1": "gruen", "2": "gelb", "4": "safety car",
    "5": "rot", "6": "vsc", "7": "vsc endet",
}


def enable_cache(path: Path | str | None = None) -> Path:
    """Cache aktivieren. Ohne Cache dauert jede Session-Ladung Minuten."""
    p = Path(path) if path else CACHE_DIR
    p.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(str(p))
    return p


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
    """
    enable_cache()
    ses = fastf1.get_session(year, gp, identifier)
    ses.load(telemetry=telemetry, weather=weather, messages=messages)
    return ses


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
    return (session.laps
            .pick_wo_box()
            .pick_accurate()
            .pick_track_status("1")
            .pick_not_deleted()
            .pick_quicklaps(threshold=threshold))


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
