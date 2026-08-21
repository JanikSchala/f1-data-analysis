from __future__ import annotations

from f1analyze import analysis

import f1lab


def test_race_pace_sorted_fastest_first(race_session):
    df = analysis.race_pace(race_session)
    assert not df.empty
    assert df["delta_s"].iloc[0] == 0
    assert (df["delta_s"].diff().dropna() >= 0).all()


def test_stint_summary_has_expected_columns(race_session):
    df = analysis.stint_summary(race_session)
    assert not df.empty
    for col in ("Driver", "Compound", "start", "end", "laps"):
        assert col in df.columns


def test_degradation_by_compound_only_reliable_fits(race_session):
    df = analysis.degradation_by_compound(race_session)
    assert not df.empty
    assert (df["mean"].abs() < 2.0).all()  # sekunden je runde, plausibilitaet


def test_pit_loss_is_a_positive_number_of_seconds(race_session):
    verlust = analysis.pit_loss(race_session)
    assert 10.0 < verlust < 40.0


def test_optimal_plan_has_at_least_one_stop(race_session):
    cfg, plan = analysis.optimal_plan(race_session)
    assert cfg.n_laps == race_session.total_laps
    assert plan.n_stops >= 1
    assert plan.green_time > 0


def test_lap_simulation_is_close_to_real_laptime(quali_session_tel):
    erg = analysis.lap_simulation(quali_session_tel)
    assert abs(erg["diff_pct"]) < 5.0    # grobe plausibilitaet
    assert erg["v_top"] > 50.0           # m/s, plausible hoechstgeschwindigkeit


def test_overtake_summary_counts_are_consistent(race_session_tel, quali_session_tel):
    erg = analysis.overtake_summary(race_session_tel, quali_session_tel)
    assert erg["lokalisiert"] <= erg["ueberholungen"]
    if erg["lokalisiert"]:
        assert 0.0 <= erg["drs_anteil"] <= 1.0


def test_traffic_scenario_never_helps_hero(race_session):
    """verkehr kann die Alternative nur unattraktiver machen. nie die
    Rangfolge zufaellig zugunsten des Optimums verzerren: der
    Verkehrs-Zuschlag ist per Konstruktion nie negativ."""
    cfg, plan = analysis.optimal_plan(race_session)
    frontier = f1lab.frontier_by_stops(cfg, up_to=4)
    alt_n = next(n for n, s in frontier.items()
                if s is not None and n != plan.n_stops)
    erg = analysis.traffic_scenario(race_session, alt_n, n_sim=500)
    naiv = erg["alt_frei"] - erg["hero_frei"]
    mit_verkehr = erg["alt_verkehr"] - erg["hero_verkehr"]
    assert mit_verkehr >= naiv - 1e-6
