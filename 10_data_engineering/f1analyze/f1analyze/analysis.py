"""Analyse-Funktionen mit Typannotationen (VORGEHEN 2).

Bewusst duenne Wrapper um f1lab, nicht neu geschriebene Rechnung: f1lab ist
in diesem Repository schon die getestete, von der App und den P-Skripten
gemeinsam genutzte Quelle fuer Race Pace, Degradation und Stints. Ein
"Wochenend-Analyzer", der diese drei Dinge stattdessen neu implementiert,
waere genau die Art Kopie, die das Projekt an anderer Stelle wiederholt
vermieden hat (siehe CLAUDE.md, DRY-Regel) - hier zaehlt, dass die
Bausteine zu einem Werkzeug zusammenkommen, nicht dass sie ein zweites Mal
geschrieben werden.
"""
from __future__ import annotations

import pandas as pd

import f1lab
from f1lab import RaceConfig, Strategy


def race_pace(session, threshold: float = 1.07) -> pd.DataFrame:
    """Bereinigte, treibstoffkorrigierte Race Pace, schnellster zuerst."""
    return f1lab.pace_table(session, threshold=threshold)


def degradation(session, threshold: float = 1.10, min_laps: int = 6) -> pd.DataFrame:
    """Degradation je Stint, mit Treibstoffkorrektur."""
    return f1lab.degradation(session, threshold=threshold, min_laps=min_laps)


def degradation_by_compound(session, **kwargs) -> pd.DataFrame:
    """Mittlere Degradation je Reifenmischung, nur belastbare Fits."""
    return f1lab.degradation_by_compound(session, **kwargs)


def stint_summary(session) -> pd.DataFrame:
    """Ein Datensatz je Stint: Fahrer, Compound, Start, Ende, Laenge."""
    return f1lab.stints(session)


def pit_loss(session) -> float:
    """Zeitverlust eines Boxenstopps auf dieser Strecke, in Sekunden."""
    return f1lab.pit_loss(session)


def optimal_plan(session) -> tuple[RaceConfig, Strategy]:
    """Exakt bester Boxenstopp-Plan dieser Session (P35), aus echter
    Degradation und echtem Pitloss - keine von Hand gesetzten Zahlen.

    Raises:
        ValueError: keine belastbaren Degradations-Fits (siehe
            f1lab.race_config_from_session).
    """
    cfg = f1lab.race_config_from_session(session)
    return cfg, f1lab.optimal_strategy(cfg)


def lap_simulation(session) -> dict:
    """Punktmassen-Rundenzeitsimulation der schnellsten Runde (P37):
    kalibrierte Fahrzeugparameter plus Abweichung von der echten Zeit."""
    dist, kappa, speed_real = f1lab.lap_speed_profile(session)
    t_real = float(session.laps.pick_fastest()["LapTime"].total_seconds())
    params = f1lab.calibrate_lap_model(dist, kappa, speed_real)
    _, t_sim = f1lab.simulate_lap(dist, kappa, params["mu_g"], params["a_accel"],
                                  params["a_brake"], params["v_top"])
    return {**params, "t_real_s": t_real, "t_sim_s": t_sim,
           "diff_pct": 100 * (t_sim - t_real) / t_real}


def traffic_scenario(session, alt_stops: int, delta: float = 0.15,
                     start_gap: float = 3.0, p_overtake: float = 0.15,
                     n_sim: int = 3000) -> dict:
    """Zwei DAG-optimale Plaene (P35) gegen einen Rivalen simuliert (P41):
    aendert Verkehr die Empfehlung? Szenario-Parameter, keine gemessenen
    Werte - siehe P41-Docstring."""
    cfg = f1lab.race_config_from_session(session)
    hero = f1lab.optimal_strategy(cfg)
    kandidaten = {n: s for n, s in f1lab.frontier_by_stops(cfg, up_to=4).items()
                 if s is not None}
    if alt_stops not in kandidaten:
        raise ValueError(f"{alt_stops}-Stopp nicht moeglich in dieser Session "
                         f"(verfuegbar: {sorted(kandidaten)})")
    alt = kandidaten[alt_stops]
    hero_t = f1lab.lap_times_for_strategy(cfg, hero)
    alt_t = f1lab.lap_times_for_strategy(cfg, alt)
    rivale_t = hero_t + delta
    c_hero, _ = f1lab.traffic_cost(hero_t, rivale_t, start_gap, p_overtake,
                                   n_sim=n_sim, seed=1)
    c_alt, _ = f1lab.traffic_cost(alt_t, rivale_t, start_gap, p_overtake,
                                  n_sim=n_sim, seed=1)
    return {"hero_stops": hero.n_stops, "hero_frei": hero.green_time,
           "hero_verkehr": hero.green_time + c_hero,
           "alt_stops": alt_stops, "alt_frei": alt.green_time,
           "alt_verkehr": alt.green_time + c_alt}


def overtake_summary(session, quali_session) -> dict:
    """Ueberholungen dieser Session und ihr Anteil in der DRS-Zone (P39).
    ``quali_session`` liefert die DRS-Zonen (im Rennen selbst oft keine
    offene Zone messbar, siehe P39-Docstring)."""
    events = f1lab.overtake_events(session)
    orte = f1lab.overtake_locations(session, drs_session=quali_session)
    anteil = float(orte["in_drs_zone"].mean()) if not orte.empty else float("nan")
    return {"ueberholungen": len(events), "lokalisiert": len(orte),
           "drs_anteil": anteil}
