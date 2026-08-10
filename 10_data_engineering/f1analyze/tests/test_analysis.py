from __future__ import annotations

from f1analyze import analysis


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
    assert (df["mean"].abs() < 2.0).all()  # Sekunden je Runde, Plausibilitaet


def test_pit_loss_is_a_positive_number_of_seconds(race_session):
    verlust = analysis.pit_loss(race_session)
    assert 10.0 < verlust < 40.0
