"""f1lab - wiederverwendbare Bausteine fuer Formel-1-Datenanalyse.

Die Skripte in diesem Repository zeigen einzelne Analysen. Dieses Paket
buendelt die Teile, die mehrfach gebraucht werden, und trennt dabei sauber:

    f1lab.core      reine Rechnung auf numpy-Arrays, ohne Netz testbar
    f1lab.session   FastF1-Anbindung: laden, filtern, aggregieren

Beispiel::

    import f1lab

    ses = f1lab.load(2024, "Spain", "R")
    print(f1lab.pace_table(ses).head())
    print(f1lab.degradation_by_compound(ses))
    print(f"Pitloss: {f1lab.pit_loss(ses):.2f} s")
"""
from .core import (
    DegradationFit,
    Elevation,
    Interval,
    bootstrap_median,
    braking_zones,
    elevation_profile,
    elo_expected,
    elo_update,
    estimate_pit_loss,
    find_cliff,
    fit_degradation,
    fuel_correct,
    mad_outlier_mask,
    optimal_undercut_window,
    path_length,
    undercut_gain,
)
from .session import (
    PaceEntry,
    cached_sessions,
    circuit_dimension,
    circuit_geometry,
    clean_laps,
    degradation,
    degradation_by_compound,
    enable_cache,
    event_dimension,
    find_cache,
    load,
    pace_table,
    pit_loss,
    race_pace,
    stints,
    track_status_phases,
)

__version__ = "0.1.0"

__all__ = [
    # core
    "Interval", "DegradationFit", "Elevation", "bootstrap_median",
    "mad_outlier_mask", "fuel_correct", "fit_degradation", "find_cliff",
    "estimate_pit_loss", "undercut_gain", "optimal_undercut_window",
    "braking_zones", "path_length", "elevation_profile",
    "elo_expected", "elo_update",
    # session
    "enable_cache", "load", "clean_laps", "PaceEntry", "race_pace",
    "pace_table", "stints", "degradation", "degradation_by_compound",
    "pit_loss", "track_status_phases", "event_dimension", "circuit_geometry",
    "circuit_dimension", "find_cache", "cached_sessions",
]
