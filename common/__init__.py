"""Gemeinsame Helfer fuer alle Projekte.

    from common import setup, clean_laps, load

Damit steht der Cache in jedem Skript identisch und du wiederholst dich nicht.
"""
from __future__ import annotations

from pathlib import Path

import fastf1

CACHE_DIR = Path.home() / "f1_cache"


def setup(mpl: bool = False) -> Path:
    """Cache aktivieren, optional das FastF1-Matplotlib-Theme setzen."""
    CACHE_DIR.mkdir(exist_ok=True)
    fastf1.Cache.enable_cache(str(CACHE_DIR))
    if mpl:
        import fastf1.plotting as f1plt
        f1plt.setup_mpl(mpl_timedelta_support=False, color_scheme="fastf1")
    return CACHE_DIR


def load(year: int, gp, ident: str = "R", **kwargs):
    """Session laden. kwargs werden an Session.load() durchgereicht."""
    setup()
    ses = fastf1.get_session(year, gp, ident)
    ses.load(**kwargs)
    return ses


def clean_laps(session, threshold: float = 1.07):
    """Runden, auf denen man Pace-Aussagen aufbauen darf.

    Entfernt Boxenrunden, unplausible Runden, alle Nicht-Gruenphasen,
    geloeschte Runden und alles langsamer als threshold * Bestzeit.
    """
    return (session.laps
            .pick_wo_box()
            .pick_accurate()
            .pick_track_status("1")
            .pick_not_deleted()
            .pick_quicklaps(threshold=threshold))


def fuel_correct(laps, total_laps: int,
                 kg_per_lap: float = 1.8, s_per_kg: float = 0.03):
    """Rundenzeiten auf konstante Tankfuellung normieren (neue Spalte FuelCorr)."""
    laps = laps.copy()
    sec = laps["LapTime"].dt.total_seconds()
    laps["FuelCorr"] = sec - (total_laps - laps["LapNumber"]) * kg_per_lap * s_per_kg
    return laps
