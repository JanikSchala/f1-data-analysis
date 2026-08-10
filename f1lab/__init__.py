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
    GRUEN,
    SC,
    DegradationFit,
    Elevation,
    InfeasibleRace,
    Interval,
    RaceConfig,
    SafetyCarProcess,
    Stint,
    Strategy,
    TyreModel,
    active_distance_zones,
    bootstrap_median,
    braking_zones,
    drs_state,
    elevation_profile,
    elo_expected,
    elo_update,
    estimate_pit_loss,
    expected_cost_of_plan,
    find_cliff,
    fit_degradation,
    frontier_by_stops,
    fuel_correct,
    hindsight_value,
    mad_outlier_mask,
    match_by_distance,
    optimal_strategy,
    optimal_undercut_window,
    path_length,
    pit_loss_crossovers,
    roll_out,
    solve_policy,
    status_intervals,
    stint_arcs,
    undercut_gain,
)
from .session import (
    PaceEntry,
    cached_sessions,
    circuit_dimension,
    circuit_geometry,
    clean_laps,
    close_following,
    compare_braking_zones,
    corner_labels,
    corner_speeds,
    degradation,
    degradation_by_compound,
    dirty_air_effect,
    driver_braking_zones,
    drs_usage,
    drs_zones,
    enable_cache,
    event_dimension,
    field_spread,
    find_cache,
    load,
    marshal_sector_labels,
    mini_sectors,
    not_deleted_mask,
    overtakes_matrix,
    pace_table,
    parse_penalties,
    parse_track_limits,
    pit_loss,
    position_progression,
    race_config_from_session,
    race_pace,
    sc_compaction,
    start_performance,
    stints,
    teammate_duels,
    temperature_effect,
    track_limit_crosscheck,
    track_status_phases,
    weather_join,
    weather_phases,
    wet_dry_classifier,
)

__version__ = "0.1.0"

__all__ = [
    # core
    "Interval", "DegradationFit", "Elevation", "bootstrap_median",
    "mad_outlier_mask", "fuel_correct", "fit_degradation", "find_cliff",
    "estimate_pit_loss", "undercut_gain", "optimal_undercut_window",
    "braking_zones", "path_length", "elevation_profile",
    "elo_expected", "elo_update", "match_by_distance",
    "active_distance_zones", "drs_state", "status_intervals",
    # session
    "enable_cache", "load", "clean_laps", "PaceEntry", "race_pace",
    "pace_table", "stints", "degradation", "degradation_by_compound",
    "pit_loss", "track_status_phases", "event_dimension", "circuit_geometry",
    "circuit_dimension", "find_cache", "cached_sessions",
    "driver_braking_zones", "compare_braking_zones", "drs_zones", "drs_usage",
    "position_progression", "overtakes_matrix", "start_performance",
    "close_following", "dirty_air_effect", "mini_sectors", "teammate_duels",
    "corner_labels", "corner_speeds", "not_deleted_mask",
    "marshal_sector_labels", "weather_join", "temperature_effect",
    "weather_phases", "wet_dry_classifier", "field_spread", "sc_compaction",
    "parse_penalties", "parse_track_limits", "track_limit_crosscheck",
    # Rennstrategie (P35)
    "TyreModel", "RaceConfig", "Stint", "Strategy", "InfeasibleRace",
    "GRUEN", "SC", "stint_arcs", "optimal_strategy", "frontier_by_stops",
    "pit_loss_crossovers", "SafetyCarProcess", "solve_policy", "roll_out",
    "expected_cost_of_plan", "hindsight_value", "race_config_from_session",
]
