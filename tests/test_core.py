"""tests fuer f1lab.core.

laufen ohne netzzugriff. alle eingaben sind synthetisch mit bekannter
wahrheit. die rechnung ist damit pruefbar ohne die F1-API.

    pytest -q
"""
from __future__ import annotations

import random

import numpy as np
import pytest

from f1lab.core import (
    GRUEN,
    SC,
    InfeasibleRace,
    Interval,
    RaceConfig,
    SafetyCarProcess,
    TyreModel,
    active_distance_zones,
    bootstrap_median,
    braking_zones,
    calibrate_lap_model,
    distance_in_any_zone,
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
    gap_evolution,
    hindsight_value,
    lap_times_for_strategy,
    lead_distance_to_zone,
    line_segments,
    mad_outlier_mask,
    match_by_distance,
    optimal_strategy,
    optimal_undercut_window,
    path_length,
    pit_loss_crossovers,
    roll_out,
    sieg_grund,
    simulate_lap,
    simulate_stint,
    solve_policy,
    status_intervals,
    stint_arcs,
    telemetry_source_quality,
    track_curvature,
    traffic_cost,
    undercut_gain,
)


# --------------------------------------------------------------- Interval
class TestInterval:
    def test_width(self):
        assert Interval(1.0, 0.5, 1.5).width == pytest.approx(1.0)

    def test_overlap_true(self):
        a, b = Interval(1.0, 0.5, 1.5), Interval(1.2, 1.0, 2.0)
        assert a.overlaps(b) and b.overlaps(a)

    def test_overlap_false(self):
        a, b = Interval(1.0, 0.5, 1.5), Interval(3.0, 2.5, 3.5)
        assert not a.overlaps(b) and not b.overlaps(a)

    def test_overlap_touching_counts_as_overlap(self):
        a, b = Interval(1.0, 0.5, 1.5), Interval(2.0, 1.5, 2.5)
        assert a.overlaps(b)


# --------------------------------------------------------------- Bootstrap
class TestBootstrapMedian:
    def test_recovers_known_median(self):
        rng = np.random.default_rng(0)
        values = rng.normal(90.0, 0.5, size=200)
        res = bootstrap_median(values)
        assert res.value == pytest.approx(90.0, abs=0.15)
        assert res.lo < res.value < res.hi

    def test_more_data_narrows_interval(self):
        rng = np.random.default_rng(1)
        small = bootstrap_median(rng.normal(90, 0.5, 15))
        large = bootstrap_median(rng.normal(90, 0.5, 400))
        assert large.width < small.width

    def test_deterministic_with_seed(self):
        v = [90.1, 90.4, 89.9, 90.7, 90.2, 90.0]
        assert bootstrap_median(v, seed=7) == bootstrap_median(v, seed=7)

    def test_ignores_nan(self):
        clean = bootstrap_median([90.0, 90.5, 91.0, 90.2], seed=3)
        dirty = bootstrap_median([90.0, np.nan, 90.5, 91.0, 90.2], seed=3)
        assert dirty.value == pytest.approx(clean.value)

    def test_too_few_values_raises(self):
        with pytest.raises(ValueError, match="mindestens 2"):
            bootstrap_median([90.0])

    def test_robust_against_single_outlier(self):
        """ein ausreisser darf den median kaum bewegen. deshalb steht hier
        nicht der mittelwert."""
        base = [90.0, 90.1, 90.2, 90.3, 90.4]
        assert (bootstrap_median(base + [140.0]).value
                == pytest.approx(bootstrap_median(base).value, abs=0.2))


# --------------------------------------------------------------- MAD
class TestMadOutlier:
    def test_flags_the_outlier(self):
        v = [90.0, 90.1, 90.2, 90.1, 90.0, 105.0]
        assert mad_outlier_mask(v).tolist() == [False] * 5 + [True]

    def test_clean_data_flags_nothing(self):
        assert not mad_outlier_mask([90.0, 90.1, 90.2, 90.3, 90.4]).any()

    def test_zero_mad_returns_all_false(self):
        assert not mad_outlier_mask([90.0] * 6).any()


# --------------------------------------------------------------- Elo
class TestEloExpected:
    def test_equal_ratings_is_a_coinflip(self):
        assert elo_expected(1500, 1500) == pytest.approx(0.5)

    def test_higher_rating_expects_to_win(self):
        assert elo_expected(1600, 1500) > 0.5

    def test_symmetric(self):
        a = elo_expected(1600, 1400)
        b = elo_expected(1400, 1600)
        assert a + b == pytest.approx(1.0)

    def test_400_point_gap_is_about_91_percent(self):
        """definierender referenzwert des Elo-Systems."""
        assert elo_expected(1900, 1500) == pytest.approx(0.909, abs=1e-3)


class TestEloUpdate:
    def test_win_raises_rating(self):
        new_a, _ = elo_update(1500, 1500, score_a=1, k=24)
        assert new_a > 1500

    def test_loss_lowers_rating(self):
        new_a, _ = elo_update(1500, 1500, score_a=0, k=24)
        assert new_a < 1500

    def test_zero_sum(self):
        """Elo verschiebt nur Punkte. der Pool bleibt konstant."""
        new_a, new_b = elo_update(1520, 1480, score_a=1, k=24)
        assert (new_a - 1520) == pytest.approx(-(new_b - 1480))

    def test_upset_moves_more_than_expected_win(self):
        """ein ueberraschender Sieg bewegt das Rating staerker als ein
        erwarteter. die Ueberraschung steckt in der Differenz zur
        erwarteten Punktzahl und nicht im Sieg selbst."""
        underdog_wins, _ = elo_update(1300, 1700, score_a=1, k=24)
        favorite_wins, _ = elo_update(1700, 1300, score_a=1, k=24)
        assert abs(underdog_wins - 1300) > abs(favorite_wins - 1700)

    def test_draw_of_equals_is_unchanged(self):
        new_a, new_b = elo_update(1500, 1500, score_a=0.5, k=24)
        assert new_a == pytest.approx(1500)
        assert new_b == pytest.approx(1500)

    def test_k_zero_never_moves_rating(self):
        new_a, new_b = elo_update(1500, 1200, score_a=1, k=0)
        assert (new_a, new_b) == (1500, 1200)


# --------------------------------------------------------------- Fuel
class TestFuelCorrect:
    def test_last_lap_is_unchanged(self):
        """am Rennende ist der Tank leer. dort ist die Korrektur null."""
        out = fuel_correct([90.0], [50], total_laps=50)
        assert out[0] == pytest.approx(90.0)

    def test_first_lap_gets_largest_correction(self):
        out = fuel_correct([90.0] * 3, [1, 25, 50], total_laps=50)
        assert out[0] < out[1] < out[2]

    def test_magnitude_matches_formula(self):
        # 49 runden rest * 1.8 kg * 0.03 s = 2.646 s
        out = fuel_correct([90.0], [1], total_laps=50)
        assert out[0] == pytest.approx(90.0 - 2.646, abs=1e-6)

    def test_removes_artificial_speedup(self):
        """reifen bleiben konstant und nur der Spritverbrauch veraendert
        sich. nach der Korrektur muessen die Zeiten flach sein."""
        laps = np.arange(1, 51)
        raw = 90.0 + (50 - laps) * 1.8 * 0.03
        corrected = fuel_correct(raw, laps, total_laps=50)
        assert np.std(corrected) == pytest.approx(0.0, abs=1e-9)


# --------------------------------------------------------------- Degradation
class TestFitDegradation:
    def test_recovers_known_slope(self):
        x = np.arange(1, 21)
        y = 90.0 + 0.08 * x
        fit = fit_degradation(x, y)
        assert fit.slope == pytest.approx(0.08, abs=1e-9)
        assert fit.intercept == pytest.approx(90.0, abs=1e-9)
        assert fit.r2 == pytest.approx(1.0)
        assert fit.n == 20

    def test_noise_lowers_r2_but_keeps_slope(self):
        rng = np.random.default_rng(5)
        x = np.arange(1, 26)
        y = 90.0 + 0.08 * x + rng.normal(0, 0.1, x.size)
        fit = fit_degradation(x, y)
        assert fit.slope == pytest.approx(0.08, abs=0.02)
        assert 0.5 < fit.r2 < 1.0

    def test_reliability_flag(self):
        x = np.arange(1, 16)
        good = fit_degradation(x, 90 + 0.08 * x)
        assert good.is_reliable

        short = fit_degradation([1, 2, 3], [90.0, 90.1, 90.2])
        assert not short.is_reliable          # zu wenige runden

    def test_flat_stint_has_low_r2(self):
        rng = np.random.default_rng(9)
        x = np.arange(1, 21)
        noise_only = fit_degradation(x, 90.0 + rng.normal(0, 0.3, x.size))
        assert not noise_only.is_reliable     # R^2 zu klein

    def test_ignores_nan_pairs(self):
        x = [1, 2, 3, np.nan, 5]
        y = [90.1, 90.2, 90.3, 90.4, 90.5]
        assert fit_degradation(x, y).n == 4

    def test_too_few_points_raises(self):
        with pytest.raises(ValueError, match="mindestens 3"):
            fit_degradation([1, 2], [90.0, 90.1])

    def test_constant_tyre_life_raises(self):
        with pytest.raises(ValueError, match="konstant"):
            fit_degradation([5, 5, 5, 5], [90.0, 90.1, 90.2, 90.3])


class TestFindCliff:
    def test_detects_the_break(self):
        """die Kurve ist bis Runde 12 flach und wird danach steiler. der
        Knick muss gefunden werden und der zweite Abschnitt muss staerker
        steigen."""
        x = np.arange(1, 25)
        y = np.where(x <= 12, 90 + 0.03 * x, 90 + 0.03 * 12 + 0.25 * (x - 12))
        cliff, left, right = find_cliff(x, y)
        assert cliff is not None
        assert 10 <= cliff <= 14
        assert right.slope > left.slope * 3

    def test_linear_stint_has_no_cliff(self):
        x = np.arange(1, 25)
        cliff, single, right = find_cliff(x, 90 + 0.08 * x)
        assert cliff is None and right is None
        assert single.slope == pytest.approx(0.08, abs=1e-9)

    def test_short_stint_returns_single_fit(self):
        x = np.arange(1, 7)
        cliff, single, right = find_cliff(x, 90 + 0.08 * x)
        assert cliff is None and right is None and single.n == 6

    def test_improving_tyre_is_not_a_cliff(self):
        """der zweite Abschnitt kann auch flacher werden. dann ist das
        kein Cliff."""
        x = np.arange(1, 25)
        y = np.where(x <= 12, 90 + 0.25 * x, 90 + 0.25 * 12 + 0.02 * (x - 12))
        cliff, _, right = find_cliff(x, y)
        assert cliff is None and right is None

    def test_noiseless_line_is_deterministic_across_platforms(self):
        """Regressionstest fuer eine exakt rauschfreie Gerade. single_sse
        liegt dann nahe der Gleitkomma-Aufloesung. der Vergleich
        sse > 0.8*single_sse vergleicht dann Rauschen gegen Rauschen und
        kann je nach BLAS/LAPACK der Plattform kippen. die 1e-6-Schwelle in
        find_cliff() faengt das ab unabhaengig vom Vorzeichen des
        Rauschens."""
        x = np.arange(1, 25, dtype=float)
        y = 90.0 + 0.08 * x       # exakt, keine Zufallskomponente
        cliff, _, right = find_cliff(x, y)
        assert cliff is None and right is None

    def test_every_split_candidate_invalid_returns_single_fit(self):
        """8 Punkte in zwei Reifenalter-Clustern (je 4x Alter 1 und 4x Alter
        5): gross genug fuer die 2*min_segment-Vorpruefung und mit echter
        Streuung innerhalb jedes Clusters (kein single_sse~0-Fast-Path), aber
        bei min_segment=4 gibt es nur einen einzigen Split-Kandidaten
        (i=4) - und der teilt exakt an der Cluster-Grenze, x[:4] ist
        konstant (ptp=0). fit_degradation() wirft dort ValueError, die
        Schleife faengt das ab (continue) und best bleibt None."""
        x = np.array([1.0, 1.0, 1.0, 1.0, 5.0, 5.0, 5.0, 5.0])
        y = np.array([90.0, 91.0, 89.0, 92.0, 95.0, 96.0, 94.0, 97.0])
        cliff, single, right = find_cliff(x, y)
        assert cliff is None and right is None
        assert single.n == 8


# --------------------------------------------------------------- Strategie
class TestPitLoss:
    def test_sums_medians(self):
        assert estimate_pit_loss([12.0, 12.4, 12.2], [8.0, 8.2, 8.1]) \
            == pytest.approx(12.2 + 8.1)

    def test_robust_against_botched_stop(self):
        normal = estimate_pit_loss([12.0, 12.2, 12.4], [8.0, 8.1, 8.2])
        with_fail = estimate_pit_loss([12.0, 12.2, 12.4, 45.0],
                                      [8.0, 8.1, 8.2, 40.0])
        assert abs(with_fail - normal) < 1.0

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            estimate_pit_loss([], [8.0])


class TestUndercut:
    def test_pays_off_when_old_tyre_degrades_faster(self):
        assert undercut_gain(deg_old=0.15, deg_new=0.04, n_laps=3) > 0

    def test_does_not_pay_off_on_fresh_rubber(self):
        """Baut der alte Reifen kaum ab, kostet die kalte Out-Lap mehr,
        als der Undercut bringt."""
        assert undercut_gain(deg_old=0.01, deg_new=0.01, n_laps=1,
                             out_lap_penalty=0.8) < 0

    def test_out_lap_penalty_reduces_gain(self):
        cold = undercut_gain(0.12, 0.05, 3, out_lap_penalty=1.2)
        warm = undercut_gain(0.12, 0.05, 3, out_lap_penalty=0.2)
        assert warm > cold
        assert warm - cold == pytest.approx(1.0)

    def test_gain_grows_with_window(self):
        gains = [undercut_gain(0.15, 0.04, n) for n in range(1, 6)]
        assert gains == sorted(gains)

    def test_zero_laps_raises(self):
        with pytest.raises(ValueError, match="mindestens 1"):
            undercut_gain(0.1, 0.05, n_laps=0)

    def test_optimal_window_returns_best(self):
        n, gain = optimal_undercut_window(0.15, 0.04, max_laps=6)
        assert 1 <= n <= 6
        assert all(gain >= undercut_gain(0.15, 0.04, k) for k in range(1, 7))


# --------------------------------------------------------------- Telemetrie
class TestBrakingZones:
    @staticmethod
    def _lap():
        """synthetische Runde mit zwei Bremszonen und dazwischen Vollgas."""
        n = 300
        t = np.linspace(0, 60, n)
        d = np.linspace(0, 5000, n)
        brake = np.zeros(n, dtype=bool)
        brake[50:80] = True          # ~500 m
        brake[180:200] = True        # ~330 m
        speed = np.full(n, 280.0)
        speed[50:80] = np.linspace(280, 120, 30)
        speed[180:200] = np.linspace(300, 90, 20)
        return brake, d, speed, t

    def test_finds_both_zones(self):
        assert len(braking_zones(*self._lap())) == 2

    def test_zone_metrics_are_plausible(self):
        z = braking_zones(*self._lap())[0]
        assert z["v_entry_kmh"] > z["v_min_kmh"]
        assert z["length_m"] > 0
        assert 0 < z["decel_g"] < 8          # F1 bremst mit bis zu ~6 g

    def test_min_length_filters_short_taps(self):
        brake, d, speed, t = self._lap()
        assert len(braking_zones(brake, d, speed, t, min_length_m=400)) == 1

    def test_no_braking_returns_empty(self):
        _, d, speed, t = self._lap()
        assert braking_zones(np.zeros(len(d), bool), d, speed, t) == []

    def test_braking_at_lap_start_is_captured(self):
        """Flankenerkennung darf eine Zone am Array-Anfang nicht verlieren."""
        n = 100
        brake = np.zeros(n, dtype=bool)
        brake[:30] = True
        d = np.linspace(0, 2000, n)
        speed = np.full(n, 200.0)
        speed[:30] = np.linspace(200, 100, 30)
        assert len(braking_zones(brake, d, speed, np.linspace(0, 20, n))) == 1

    def test_mismatched_lengths_raise(self):
        with pytest.raises(ValueError, match="gleich lang"):
            braking_zones([True, False], [0, 1, 2], [200, 190, 180], [0, 1, 2])

    def test_braking_at_lap_end_is_captured(self):
        """spiegelbildlich zu test_braking_at_lap_start_is_captured: eine
        Zone, die genau bis zur letzten Probe des Arrays reicht, darf nicht
        verlorengehen (kein schliessendes -1-Flankenereignis vorhanden)."""
        n = 100
        brake = np.zeros(n, dtype=bool)
        brake[70:] = True
        d = np.linspace(0, 2000, n)
        speed = np.full(n, 200.0)
        speed[70:] = np.linspace(200, 100, 30)
        zones = braking_zones(brake, d, speed, np.linspace(0, 20, n))
        assert len(zones) == 1
        assert zones[0]["length_m"] == pytest.approx(d[-1] - d[70], rel=0.05)


class TestTelemetrySourceQuality:
    def test_empty_returns_zeroed_summary(self):
        assert telemetry_source_quality([]) == {
            "n": 0, "car": 0.0, "pos": 0.0, "interpolation": 0.0}

    def test_mixed_sources_compute_ratios(self):
        source = ["car"] * 6 + ["pos"] * 3 + ["interpolation"] * 1
        erg = telemetry_source_quality(source)
        assert erg["n"] == 10
        assert erg["car"] == pytest.approx(0.6)
        assert erg["pos"] == pytest.approx(0.3)
        assert erg["interpolation"] == pytest.approx(0.1)


class TestMatchByDistance:
    def test_exact_match(self):
        assert match_by_distance([100, 500], [100, 500], tolerance=1) == \
            [(0, 0), (1, 1)]

    def test_within_tolerance(self):
        assert match_by_distance([100], [107], tolerance=10) == [(0, 0)]

    def test_outside_tolerance_stays_unmatched(self):
        assert match_by_distance([100], [200], tolerance=10) == []

    def test_extra_value_in_b_is_ignored(self):
        """eine zusaetzliche Bremsung eines Fahrers ist kein Fehlerfall."""
        paare = match_by_distance([100, 500], [100, 300, 500], tolerance=5)
        assert paare == [(0, 0), (1, 2)]

    def test_no_double_booking(self):
        """zwei nahe a-Werte duerfen sich nicht denselben b-Wert teilen."""
        paare = match_by_distance([100, 102], [101], tolerance=5)
        assert len(paare) == 1

    def test_empty_inputs(self):
        assert match_by_distance([], [1, 2, 3], tolerance=5) == []
        assert match_by_distance([1, 2, 3], [], tolerance=5) == []

    def test_order_follows_a(self):
        paare = match_by_distance([500, 100], [500, 100], tolerance=1)
        assert paare == [(0, 0), (1, 1)]


class TestActiveDistanceZones:
    def test_single_zone(self):
        active = [False, False, True, True, True, False]
        d = [0, 10, 20, 30, 40, 50]
        zonen = active_distance_zones(active, d, min_length_m=5)
        assert len(zonen) == 1
        assert zonen[0] == {"start_m": 20.0, "end_m": 40.0, "length_m": 20.0}

    def test_short_zone_filtered(self):
        active = [False, True, False]
        d = [0, 10, 20]
        assert active_distance_zones(active, d, min_length_m=5) == []

    def test_active_at_edges_captured(self):
        """Flankenerkennung darf Zonen am Anfang/Ende nicht verlieren -
        beide Randlaeufe brauchen genug Punkte fuer eine echte Distanz."""
        active = [True, True, False, False, True, True]
        d = [0, 10, 20, 30, 40, 50]
        zonen = active_distance_zones(active, d, min_length_m=5)
        assert len(zonen) == 2
        assert zonen[0] == {"start_m": 0.0, "end_m": 10.0, "length_m": 10.0}
        assert zonen[1] == {"start_m": 40.0, "end_m": 50.0, "length_m": 10.0}

    def test_never_active_returns_empty(self):
        assert active_distance_zones([False] * 5, range(5)) == []

    def test_mismatched_lengths_raise(self):
        with pytest.raises(ValueError, match="gleich lang"):
            active_distance_zones([True, False], [0, 1, 2])


# ----------------------------------------------------------- distance_in_any_zone
class TestDistanceInAnyZone:
    """zonen-Zugehoerigkeit einzelner Distanzen (P39)."""

    def test_inside_and_outside(self):
        out = distance_in_any_zone([5, 50, 150, 250], [0, 200], [100, 300])
        assert out.tolist() == [True, True, False, True]

    def test_boundary_is_inclusive(self):
        out = distance_in_any_zone([0, 100], [0], [100])
        assert out.tolist() == [True, True]

    def test_no_zones_returns_all_false(self):
        out = distance_in_any_zone([1, 2, 3], [], [])
        assert out.tolist() == [False, False, False]

    def test_overlapping_zones_still_true(self):
        """eine Distanz in mehreren Zonen zaehlt trotzdem nur einmal True."""
        out = distance_in_any_zone([50], [0, 40], [60, 80])
        assert out.tolist() == [True]


# ------------------------------------------------------- lead_distance_to_zone
class TestLeadDistanceToZone:
    """abstand zur naechsten Zone in Fahrtrichtung mit Rundenumbruch (P39)."""

    def test_before_a_zone(self):
        out = lead_distance_to_zone([80], [100, 300], track_length_m=1000)
        assert out.tolist() == [20.0]

    def test_at_zone_start_is_zero(self):
        out = lead_distance_to_zone([100], [100, 300], track_length_m=1000)
        assert out.tolist() == [0.0]

    def test_between_two_zones_picks_the_next_one(self):
        out = lead_distance_to_zone([150], [100, 300], track_length_m=1000)
        assert out.tolist() == [150.0]

    def test_past_last_zone_wraps_to_next_lap(self):
        """nach der letzten Zone zaehlt die Reststrecke bis zum Ziel plus
        die Distanz vom Start bis zur ersten Zone der naechsten Runde."""
        out = lead_distance_to_zone([950], [100, 300], track_length_m=1000)
        assert out.tolist() == [150.0]        # (1000-950) + 100

    def test_no_zones_returns_nan(self):
        out = lead_distance_to_zone([1, 2], [], track_length_m=1000)
        assert np.isnan(out).all()

    def test_vectorized_over_multiple_points(self):
        out = lead_distance_to_zone([80, 150, 950], [100, 300],
                                    track_length_m=1000)
        assert out.tolist() == [20.0, 150.0, 150.0]


class TestDrsState:
    def test_closed_by_default(self):
        assert drs_state([0, 1, 3]).tolist() == [0, 0, 0]

    def test_detected_is_one(self):
        assert drs_state([8, 8]).tolist() == [1, 1]

    def test_open_codes_are_two(self):
        assert drs_state([10, 12, 14]).tolist() == [2, 2, 2]

    def test_mixed_sequence(self):
        assert drs_state([0, 8, 10, 1, 12]).tolist() == [0, 1, 2, 0, 2]


# --------------------------------------------------------------- path_length
class TestPathLength:
    def test_unit_square_closed(self):
        """vier Seiten a 1 -> Umfang 4. das Schlusssegment zaehlt mit."""
        x, y = [0, 1, 1, 0], [0, 0, 1, 1]
        assert path_length(x, y) == pytest.approx(4.0)

    def test_unit_square_open_drops_last_side(self):
        x, y = [0, 1, 1, 0], [0, 0, 1, 1]
        assert path_length(x, y, closed=False) == pytest.approx(3.0)

    def test_straight_line(self):
        assert path_length([0, 3], [0, 4], closed=False) == pytest.approx(5.0)

    def test_closed_line_counts_the_way_back(self):
        assert path_length([0, 3], [0, 4], closed=True) == pytest.approx(10.0)

    def test_circle_approximates_two_pi_r(self):
        """ein feines Polygon naehert den Kreisumfang von unten an."""
        t = np.linspace(0, 2 * np.pi, 2000, endpoint=False)
        got = path_length(np.cos(t) * 100, np.sin(t) * 100)
        assert got == pytest.approx(2 * np.pi * 100, rel=1e-5)

    def test_resolution_does_not_change_result(self):
        """doppelt so viele Stuetzpunkte auf derselben Geraden -> gleiche Laenge."""
        coarse = path_length([0, 10], [0, 0], closed=False)
        fine = path_length(np.linspace(0, 10, 50), np.zeros(50), closed=False)
        assert coarse == pytest.approx(fine)

    def test_nan_points_are_dropped(self):
        x = [0, np.nan, 1, 1, 0]
        y = [0, 0.5, 0, 1, 1]
        assert path_length(x, y) == pytest.approx(4.0)

    def test_too_few_points_is_zero(self):
        assert path_length([1], [1]) == 0.0
        assert path_length([], []) == 0.0

    def test_mismatched_lengths_raise(self):
        with pytest.raises(ValueError, match="gleich lang"):
            path_length([0, 1, 2], [0, 1])


class TestLineSegments:
    """Eingabeformat fuer LineCollection: je Segment Start- und Endpunkt."""

    def test_shape_is_one_segment_less_than_points(self):
        got = line_segments([0, 1, 2, 3], [0, 0, 0, 0])
        assert got.shape == (3, 2, 2)

    def test_segments_chain_point_to_point(self):
        """das Ende eines Segments ist der Anfang des naechsten - sonst
        entstehen Luecken in der eingefaerbten Strecke."""
        got = line_segments([0, 1, 2], [0, 10, 20])
        assert got[0].tolist() == [[0, 0], [1, 10]]
        assert got[1].tolist() == [[1, 10], [2, 20]]

    def test_too_few_points_is_empty_but_keeps_shape(self):
        """LineCollection braucht die (n, 2, 2)-Form auch ohne Segmente."""
        for x, y in (([1], [1]), ([], [])):
            got = line_segments(x, y)
            assert got.shape == (0, 2, 2)

    def test_mismatched_lengths_raise(self):
        with pytest.raises(ValueError, match="gleich lang"):
            line_segments([0, 1, 2], [0, 1])

    def test_one_color_per_segment_not_per_point(self):
        """der eigentliche Zweck: n Punkte tragen n-1 Farben. wer die
        Farbliste aus den Punkten baut, ist um eins zu lang."""
        x = np.linspace(0, 100, 25)
        got = line_segments(x, np.zeros(25))
        assert len(got) == len(x) - 1


# --------------------------------------------------------------- elevation
class TestElevationProfile:
    """hoehenmeter mit Hysterese gegen Messrauschen."""

    def test_monotonic_climb(self):
        got = elevation_profile(np.arange(0, 51, 1.0))
        assert got.gain == pytest.approx(50.0)
        assert got.drop == pytest.approx(0.0)
        assert got.span == pytest.approx(50.0)

    def test_climb_then_descent_is_balanced(self):
        """start und Ziel liegen am selben Punkt -> Anstieg gleich Abstieg."""
        z = np.r_[np.arange(0, 31, 1.0), np.arange(29, -1, -1.0)]
        got = elevation_profile(z)
        assert got.gain == pytest.approx(30.0)
        assert got.drop == pytest.approx(30.0)
        assert got.span == pytest.approx(30.0)

    @staticmethod
    def _messrauschen(rng, n, sigma=0.3, window=25):
        """korreliertes Rauschen wie es Positionsdaten tatsaechlich zeigen.

        weisses Rauschen waere das falsche Modell. aufeinanderfolgende Proben
        eines Positionskanals haengen zusammen und springen nicht
        unabhaengig.
        """
        glatt = np.ones(window) / window
        return np.convolve(rng.normal(0, sigma, n + window), glatt,
                           mode="valid")[:n]

    def test_noise_below_threshold_is_ignored(self):
        """der eigentliche Zweck: Rauschen darf keine Hoehenmeter erzeugen."""
        z = self._messrauschen(np.random.default_rng(3), 5000)
        got = elevation_profile(z, min_step=1.0)
        assert got.gain == 0.0
        assert got.drop == 0.0

    def test_naive_sum_would_massively_overcount(self):
        """zeigt warum die Hysterese noetig ist."""
        z = self._messrauschen(np.random.default_rng(3), 5000)
        naiv = float(np.abs(np.diff(z)).sum())
        assert naiv > 50                         # ohne Schwelle frei erfunden
        assert elevation_profile(z, min_step=1.0).gain == 0.0

    def test_white_noise_still_leaks_above_threshold(self):
        """eine bewusst festgehaltene Grenze des Verfahrens.

        weisses Rauschen mit sigma=0.3 ueberschreitet die 1-m-Schwelle ueber
        5000 Proben hunderte Male. die Hysterese daempft das deutlich. sie
        setzt es aber nicht auf null. wer die Funktion auf ungeglaettete
        Daten anwendet muss min_step anheben.
        """
        rng = np.random.default_rng(0)
        z = rng.normal(0, 0.3, 5000)
        naiv = float(np.abs(np.diff(z)).sum())
        gedaempft = elevation_profile(z, min_step=1.0).gain
        assert gedaempft < naiv / 5              # klar besser als naiv
        assert gedaempft > 0                     # aber eben nicht null
        assert elevation_profile(z, min_step=2.0).gain == 0.0

    def test_real_climb_survives_noise(self):
        """ein echter Anstieg darf durch ueberlagertes Rauschen nicht verschwinden."""
        rng = np.random.default_rng(4)
        z = np.linspace(0, 40, 2000) + self._messrauschen(rng, 2000, sigma=0.5)
        got = elevation_profile(z, min_step=1.0)
        assert got.gain == pytest.approx(40.0, abs=2.0)
        assert got.drop < 2.0

    def test_span_uses_raw_extremes(self):
        assert elevation_profile([0.0, 5.0, -3.0, 2.0]).span == pytest.approx(8.0)

    def test_is_flat_flag(self):
        assert elevation_profile([0.0, 1.0, 0.0]).is_flat
        assert not elevation_profile([0.0, 40.0, 0.0]).is_flat

    def test_larger_threshold_counts_less(self):
        z = np.r_[np.arange(0, 11, 1.0), np.arange(9, -1, -1.0)]
        fein = elevation_profile(z, min_step=0.5).gain
        grob = elevation_profile(z, min_step=5.0).gain
        assert grob <= fein

    def test_too_few_points_is_zero(self):
        got = elevation_profile([42.0])
        assert (got.gain, got.drop, got.span) == (0.0, 0.0, 0.0)

    def test_nan_is_dropped(self):
        got = elevation_profile([0.0, np.nan, 10.0])
        assert got.gain == pytest.approx(10.0)


# --------------------------------------------------------------- StatusIntervals
class TestStatusIntervals:
    """sparses Aenderungs-Log (wie FastF1s track_status) in Intervalle."""

    def test_end_is_next_start_not_own_last_occurrence(self):
        """der eigentliche Zweck: 'SCDeployed' erscheint einmal und gilt bis
        zum naechsten Statuswechsel und nicht nur an seinem eigenen
        Zeitpunkt. sonst waere das Intervall Laenge 0."""
        status = ["1", "4", "1"]
        zeit = [0.0, 100.0, 500.0]
        s, start, ende = status_intervals(status, zeit)
        assert list(s) == ["1", "4", "1"]
        assert list(start) == [0.0, 100.0, 500.0]
        assert ende[0] == pytest.approx(100.0)
        assert ende[1] == pytest.approx(500.0)
        assert np.isnan(ende[2])                  # letztes intervall offen

    def test_immediate_repeats_are_merged(self):
        status = ["1", "1", "4", "1"]
        zeit = [0.0, 10.0, 100.0, 500.0]
        s, start, ende = status_intervals(status, zeit)
        assert list(s) == ["1", "4", "1"]
        assert list(start) == [0.0, 100.0, 500.0]

    def test_naive_groupby_would_give_zero_length(self):
        """dokumentiert den vermiedenen Fehler. Gruppieren nach
        *aufeinanderfolgend* gleichem Status (cumsum ueber Wechsel) und
        dessen eigenem Min/Max ergibt hier ueberall Laenge 0 weil kein
        Status zweimal hintereinander steht. das war die urspruengliche
        Fassung von f1lab.session.track_status_phases()."""
        status = np.array(["1", "4", "1", "6", "1"])
        zeit = np.array([0.0, 100.0, 500.0, 600.0, 900.0])
        gruppe = np.cumsum(np.r_[True, status[1:] != status[:-1]])
        naiv_dauer = 0.0
        for g in np.unique(gruppe):
            m = gruppe == g
            naiv_dauer += float(zeit[m].max() - zeit[m].min())
        assert naiv_dauer == 0.0

        _, start, ende = status_intervals(status, zeit)
        echte_dauer = np.nansum(ende - start)
        assert echte_dauer == pytest.approx(900.0)

    def test_empty_input(self):
        s, start, ende = status_intervals([], [])
        assert len(s) == 0 and len(start) == 0 and len(ende) == 0

    def test_single_entry_is_open_ended(self):
        s, start, ende = status_intervals(["1"], [0.0])
        assert list(s) == ["1"]
        assert np.isnan(ende[0])


# --------------------------------------------------------------- Rennstrategie
class TestTyreModel:
    def test_lap_time_at_age_one_is_base_time(self):
        t = TyreModel("SOFT", 90.0, 0.05)
        assert t.lap_time(1) == pytest.approx(90.0)

    def test_lap_time_includes_linear_and_quad_terms(self):
        t = TyreModel("SOFT", 90.0, 0.05, 0.01)
        # a = age - 1 = 4
        assert t.lap_time(5) == pytest.approx(90.0 + 0.05 * 4 + 0.01 * 16)

    def test_age_below_one_raises(self):
        t = TyreModel("SOFT", 90.0, 0.05)
        with pytest.raises(ValueError, match="einsbasiert"):
            t.lap_time(0)

    def test_stint_time_sums_each_lap(self):
        t = TyreModel("SOFT", 90.0, 1.0)
        # runden 1..3: 90, 91, 92
        assert t.stint_time(3) == pytest.approx(90 + 91 + 92)

    def test_stint_time_below_one_raises(self):
        with pytest.raises(ValueError, match="mindestens 1"):
            TyreModel("SOFT", 90.0, 0.05).stint_time(0)


class TestRaceConfig:
    def _tyres(self):
        return (TyreModel("SOFT", 90.0, 0.05), TyreModel("HARD", 91.0, 0.02))

    def test_duplicate_compounds_raise(self):
        with pytest.raises(ValueError, match="doppelte Mischungen"):
            RaceConfig(n_laps=10, pit_loss=20.0,
                      tyres=(TyreModel("SOFT", 90.0, 0.05),
                             TyreModel("SOFT", 91.0, 0.02)))

    def test_no_tyres_raises(self):
        with pytest.raises(ValueError, match="mindestens eine Mischung"):
            RaceConfig(n_laps=10, pit_loss=20.0, tyres=())

    def test_single_compound_needs_flag_disabled(self):
        with pytest.raises(ValueError, match="Zweimischungs-Regel"):
            RaceConfig(n_laps=10, pit_loss=20.0,
                      tyres=(TyreModel("SOFT", 90.0, 0.05),))
        # mit ausgeschalteter regel geht es
        RaceConfig(n_laps=10, pit_loss=20.0,
                  tyres=(TyreModel("SOFT", 90.0, 0.05),),
                  require_two_compounds=False)

    def test_unknown_start_compound_raises(self):
        with pytest.raises(ValueError, match="unbekannte Startmischung"):
            RaceConfig(n_laps=10, pit_loss=20.0, tyres=self._tyres(),
                      start_compound="MEDIUM")

    def test_zero_laps_raises(self):
        with pytest.raises(ValueError, match="n_laps"):
            RaceConfig(n_laps=0, pit_loss=20.0, tyres=self._tyres())

    def test_zero_min_stint_raises(self):
        with pytest.raises(ValueError, match="min_stint"):
            RaceConfig(n_laps=10, pit_loss=20.0, tyres=self._tyres(),
                      min_stint=0)

    def test_fuel_offset_is_zero_without_fuel_effect(self):
        cfg = RaceConfig(n_laps=20, pit_loss=20.0, tyres=self._tyres())
        assert cfg.fuel_offset == pytest.approx(0.0)

    def test_fuel_offset_scales_with_laps_squared(self):
        cfg = RaceConfig(n_laps=10, pit_loss=20.0, tyres=self._tyres(),
                         fuel_effect=0.1)
        # 0.1 * 10 * 9 / 2 = 4.5
        assert cfg.fuel_offset == pytest.approx(4.5)


class TestStintArcs:
    def test_max_age_caps_stint_length(self):
        """ein TyreModel mit max_age darf keinen Stint erzeugen, der laenger
        laeuft als der Reifen haelt, selbst wenn n_laps/max_stint mehr
        zuliessen."""
        cfg = RaceConfig(
            n_laps=20, pit_loss=20.0, min_stint=1,
            tyres=(TyreModel("SOFT", 90.0, 0.3, max_age=5),
                  TyreModel("HARD", 91.0, 0.05)))
        arcs = stint_arcs(cfg)
        soft_laengen = [end - start + 1 for ci, start, end, _ in arcs
                        if cfg.tyres[ci].compound == "SOFT"]
        assert soft_laengen and max(soft_laengen) <= 5

    def test_start_compound_excludes_other_tyres_at_lap_one(self):
        """nur Stints der Startmischung duerfen bei Runde 1 beginnen."""
        cfg = RaceConfig(
            n_laps=20, pit_loss=20.0, min_stint=1,
            tyres=(TyreModel("SOFT", 90.0, 0.3), TyreModel("HARD", 91.0, 0.05)),
            start_compound="SOFT")
        arcs = stint_arcs(cfg)
        opener = {cfg.tyres[ci].compound for ci, start, _end, _preis in arcs
                  if start == 1}
        assert opener == {"SOFT"}


class TestOptimalStrategy:
    """die Rennen sind klein. sie lassen sich von Hand nachrechnen."""

    def _cfg(self, **kw):
        defaults = dict(
            n_laps=10, pit_loss=20.0, min_stint=1,
            tyres=(TyreModel("SOFT", 90.0, 1.0), TyreModel("HARD", 91.0, 0.1)))
        defaults.update(kw)
        return RaceConfig(**defaults)

    def test_prefers_flatter_degrading_tyre_for_long_stint(self):
        """SOFT baut zehnmal so schnell ab wie HARD. bei freier Stintlaenge
        gewinnt ein frueher Wechsel auf HARD."""
        cfg = self._cfg()
        best = optimal_strategy(cfg)
        # das laengste stueck des rennens sollte auf HARD liegen
        laengster = max(best.stints, key=lambda s: s.length)
        assert laengster.compound == "HARD"

    def test_respects_two_compound_rule(self):
        best = optimal_strategy(self._cfg())
        assert len(set(best.compounds)) >= 2

    def test_respects_min_stint(self):
        cfg = self._cfg(min_stint=4)
        best = optimal_strategy(cfg)
        assert all(s.length >= 4 for s in best.stints)

    def test_total_laps_covers_race_exactly(self):
        best = optimal_strategy(self._cfg())
        assert sum(s.length for s in best.stints) == 10
        assert best.stints[0].start_lap == 1
        assert best.stints[-1].end_lap == 10

    def test_infeasible_min_stint_raises(self):
        """min_stint*2 > n_laps mit Zweimischungs-Pflicht: kein Plan passt."""
        cfg = self._cfg(n_laps=5, min_stint=4)
        with pytest.raises(InfeasibleRace):
            optimal_strategy(cfg)

    def test_total_time_includes_fuel_offset(self):
        cfg = self._cfg(fuel_effect=0.1)
        best = optimal_strategy(cfg)
        assert best.total_time == pytest.approx(best.green_time + cfg.fuel_offset)

    def test_no_legal_stint_arcs_raises_before_the_dp(self):
        """min_stint > n_laps: schon stint_arcs() liefert nichts, die DP
        laeuft gar nicht erst an (andere Fehlermeldung als der Fall oben,
        wo Arcs existieren aber keine Kombination die Zweimischungs-Regel
        erfuellt)."""
        cfg = self._cfg(n_laps=3, min_stint=5, require_two_compounds=False)
        with pytest.raises(InfeasibleRace, match="keine legalen Stints"):
            optimal_strategy(cfg)

    def test_more_pit_stops_never_beats_pit_loss_savings(self):
        """ein Vergleichsplan mit einem zusaetzlichen, unnoetigen Stopp kann
        nie besser sein als das Optimum."""
        cfg = self._cfg()
        best = optimal_strategy(cfg)
        frontier = frontier_by_stops(cfg, up_to=4)
        best_by_stops = min(s.green_time for s in frontier.values()
                            if s is not None)
        assert best_by_stops == pytest.approx(best.green_time)


class TestFrontierByStops:
    def test_each_entry_has_requested_stop_count(self):
        cfg = RaceConfig(n_laps=20, pit_loss=20.0, min_stint=2,
                         tyres=(TyreModel("SOFT", 90.0, 0.3),
                               TyreModel("HARD", 91.0, 0.05)))
        frontier = frontier_by_stops(cfg, up_to=3)
        for n, strat in frontier.items():
            if strat is not None:
                assert strat.n_stops == n

    def test_zero_stops_infeasible_with_two_compound_rule(self):
        cfg = RaceConfig(n_laps=20, pit_loss=20.0,
                         tyres=(TyreModel("SOFT", 90.0, 0.3),
                               TyreModel("HARD", 91.0, 0.05)))
        frontier = frontier_by_stops(cfg, up_to=1)
        assert frontier[0] is None


class TestPitLossCrossovers:
    def test_lower_pit_loss_favours_more_stops(self):
        """ein guenstigerer Boxenstopp macht zusaetzliche Stopps attraktiver.
        die Stoppzahl bei niedrigem Pitloss ist mindestens so hoch wie bei
        hohem Pitloss."""
        cfg = RaceConfig(n_laps=40, pit_loss=20.0, min_stint=3,
                         tyres=(TyreModel("SOFT", 90.0, 0.3),
                               TyreModel("HARD", 91.0, 0.05)))
        from dataclasses import replace
        n_billig = optimal_strategy(replace(cfg, pit_loss=1.0)).n_stops
        n_teuer = optimal_strategy(replace(cfg, pit_loss=100.0)).n_stops
        assert n_billig >= n_teuer

    def test_crossovers_lie_within_range(self):
        cfg = RaceConfig(n_laps=40, pit_loss=20.0, min_stint=3,
                         tyres=(TyreModel("SOFT", 90.0, 0.3),
                               TyreModel("HARD", 91.0, 0.05)))
        grenzen = pit_loss_crossovers(cfg, 5.0, 60.0)
        assert all(5.0 <= g <= 60.0 for g in grenzen)


class TestSafetyCarProcess:
    def test_marginals_are_probabilities(self):
        prozess = SafetyCarProcess(p_deploy=0.05, p_end=0.3)
        q = prozess.marginals(50)
        assert all(0.0 <= v <= 1.0 for v in q[1:51])

    def test_marginals_start_at_zero(self):
        """runde 1 startet immer gruen."""
        prozess = SafetyCarProcess()
        q = prozess.marginals(10)
        assert q[1] == pytest.approx(0.0)

    def test_no_deploy_process_never_shows_sc(self):
        prozess = SafetyCarProcess(p_deploy=0.0, p_end=0.5)
        q = prozess.marginals(20)
        assert all(v == pytest.approx(0.0) for v in q[1:21])

    def test_sample_only_produces_valid_states(self):
        import random
        prozess = SafetyCarProcess(p_deploy=0.1, p_end=0.3)
        folge = prozess.sample(random.Random(1), 30)
        assert all(s in (GRUEN, SC) for s in folge[1:31])

    def test_sample_length_matches_laps(self):
        import random
        prozess = SafetyCarProcess()
        folge = prozess.sample(random.Random(0), 25)
        assert len(folge) == 26  # 1-indexed, index 0 ungenutzt


class TestSolvePolicyAndRollOut:
    def _cfg(self):
        return RaceConfig(n_laps=15, pit_loss=20.0, min_stint=3,
                          tyres=(TyreModel("SOFT", 90.0, 0.3),
                                TyreModel("HARD", 91.0, 0.05)))

    def test_solve_policy_returns_finite_value(self):
        wert, politik = solve_policy(self._cfg(), SafetyCarProcess())
        assert wert < float("inf")
        assert ("start",) in politik

    def test_exact_stops_not_supported(self):
        from dataclasses import replace
        cfg = replace(self._cfg(), exact_stops=1)
        with pytest.raises(NotImplementedError):
            solve_policy(cfg, SafetyCarProcess())

    def test_roll_out_matches_race_distance(self):
        cfg = self._cfg()
        _wert, politik = solve_policy(cfg, SafetyCarProcess())
        folge = [GRUEN] * (cfg.n_laps + 1)
        strat = roll_out(cfg, politik, folge)
        assert sum(s.length for s in strat.stints) == cfg.n_laps

    def test_roll_out_respects_two_compound_rule(self):
        cfg = self._cfg()
        _wert, politik = solve_policy(cfg, SafetyCarProcess())
        folge = [GRUEN] * (cfg.n_laps + 1)
        strat = roll_out(cfg, politik, folge)
        assert len(set(strat.compounds)) >= 2

    def test_start_compound_restricts_opening_tyre(self):
        cfg = RaceConfig(n_laps=15, pit_loss=20.0, min_stint=3,
                         tyres=(TyreModel("SOFT", 90.0, 0.3),
                               TyreModel("HARD", 91.0, 0.05)),
                         start_compound="HARD")
        wert, politik = solve_policy(cfg, SafetyCarProcess())
        assert wert < float("inf")
        ci_start = politik[("start",)]
        assert cfg.tyres[ci_start].compound == "HARD"

    def test_no_feasible_policy_raises(self):
        """min_stint > n_laps: kein Zustand am Renn-Ende ist je 'gut', der
        Erwartungswert bleibt unendlich (andere Stelle als die Arc-basierte
        Infeasibility in TestOptimalStrategy - solve_policy baut seine
        eigene DP, ohne stint_arcs())."""
        cfg = RaceConfig(n_laps=3, pit_loss=20.0, min_stint=5,
                         tyres=(TyreModel("SOFT", 90.0, 0.3),
                               TyreModel("HARD", 91.0, 0.05)),
                         require_two_compounds=False)
        with pytest.raises(InfeasibleRace, match="keine Politik"):
            solve_policy(cfg, SafetyCarProcess())


class TestExpectedCostAndHindsight:
    def _cfg(self):
        return RaceConfig(n_laps=20, pit_loss=20.0, min_stint=3,
                          tyres=(TyreModel("SOFT", 90.0, 0.3),
                                TyreModel("HARD", 91.0, 0.05)))

    def test_expected_cost_without_sc_equals_green_time(self):
        """ohne jede Safety-Car-Wahrscheinlichkeit gibt es keinen Rabatt.
        die erwarteten Kosten sind genau die Fahrzeit des festen Plans."""
        cfg = self._cfg()
        plan = optimal_strategy(cfg)
        prozess = SafetyCarProcess(p_deploy=0.0, p_end=0.5)
        kosten = expected_cost_of_plan(cfg, plan, prozess)
        assert kosten == pytest.approx(plan.green_time)

    def test_hindsight_never_worse_than_expected_cost_of_fixed_plan(self):
        """das Optimum bei bekanntem Verlauf ist eine untere Schranke. im
        Mittel ist es nie schlechter als ein fester Plan der stur
        durchgezogen wird."""
        cfg = self._cfg()
        plan = optimal_strategy(cfg)
        prozess = SafetyCarProcess(p_deploy=0.05, p_end=0.3)
        naiv = expected_cost_of_plan(cfg, plan, prozess)
        blind, _se = hindsight_value(cfg, prozess, n=200, seed=3)
        assert blind <= naiv + 1e-6

    def test_hindsight_without_sc_matches_deterministic_optimum(self):
        cfg = self._cfg()
        prozess = SafetyCarProcess(p_deploy=0.0, p_end=0.5)
        blind, se = hindsight_value(cfg, prozess, n=50, seed=1)
        deterministic = optimal_strategy(cfg).green_time
        assert blind == pytest.approx(deterministic)
        assert se == pytest.approx(0.0, abs=1e-9)


# --------------------------------------------------------- lap_times_for_strategy
class TestLapTimesForStrategy:
    """rundenzeiten je Runde aus einer festen Strategie (P41)."""

    def test_sum_matches_green_time(self):
        cfg = RaceConfig(n_laps=20, pit_loss=20.0, min_stint=3,
                         tyres=(TyreModel("SOFT", 90.0, 0.3),
                               TyreModel("HARD", 91.0, 0.05)))
        plan = optimal_strategy(cfg)
        zeiten = lap_times_for_strategy(cfg, plan)
        assert zeiten.sum() == pytest.approx(plan.green_time)

    def test_length_matches_n_laps(self):
        cfg = RaceConfig(n_laps=15, pit_loss=20.0,
                         tyres=(TyreModel("SOFT", 90.0, 0.3),
                               TyreModel("HARD", 91.0, 0.05)))
        plan = optimal_strategy(cfg)
        assert len(lap_times_for_strategy(cfg, plan)) == 15

    def test_pit_loss_lands_on_pit_lap(self):
        """der Boxenverlust steckt auf der Runde an deren Ende gestoppt wird.
        das ist genau die in Strategy.pit_laps genannte Runde."""
        cfg = RaceConfig(n_laps=10, pit_loss=25.0, min_stint=2,
                         tyres=(TyreModel("SOFT", 90.0, 0.0),
                               TyreModel("HARD", 90.0, 0.0)),
                         exact_stops=1)
        plan = optimal_strategy(cfg)
        zeiten = lap_times_for_strategy(cfg, plan)
        pit_lap = plan.pit_laps[0]
        andere = [zeiten[lap - 1] for lap in range(1, 11) if lap != pit_lap]
        assert zeiten[pit_lap - 1] == pytest.approx(90.0 + 25.0)
        assert all(z == pytest.approx(90.0) for z in andere)


# ------------------------------------------------------------- gap_evolution
class TestGapEvolution:
    """rundenweiser Abstand zweier Autos mit Ueberholwahrscheinlichkeit (P41)."""

    def test_far_apart_never_blocks(self):
        """weit auseinander liegend entspricht die Simulation exakt der
        freien Rechnung. das gilt unabhaengig von p_overtake."""
        hero = np.full(10, 90.0)
        rival = np.full(10, 90.0)
        verlauf, blockiert = gap_evolution(hero, rival, initial_gap=30.0,
                                           p_overtake=0.0,
                                           rng=random.Random(1))
        assert blockiert == 0
        assert verlauf[-1] == pytest.approx(30.0)

    def test_p_overtake_one_matches_free_gap(self):
        """gelingt jeder Versuch macht die Blockade keinen Unterschied mehr.
        die Simulation muss dann der freien Rechnung entsprechen."""
        hero = np.full(20, 89.0)     # 1s/runde schneller
        rival = np.full(20, 90.0)
        verlauf, _ = gap_evolution(hero, rival, initial_gap=0.5,
                                   p_overtake=1.0, block_gap_s=1.0,
                                   rng=random.Random(2))
        frei_ende = 0.5 + float(np.sum(hero - rival))
        assert verlauf[-1] == pytest.approx(frei_ende)

    def test_p_overtake_zero_pins_at_block_gap(self):
        """gelingt kein Versuch kann hero nie unter block_gap_s
        aufschliessen. das gilt auch bei grossem Tempovorteil."""
        hero = np.full(20, 85.0)     # 5s/runde schneller
        rival = np.full(20, 90.0)
        verlauf, blockiert = gap_evolution(hero, rival, initial_gap=0.8,
                                           p_overtake=0.0, block_gap_s=1.0,
                                           rng=random.Random(3))
        assert blockiert > 0
        assert verlauf[-1] == pytest.approx(1.0)
        assert (verlauf[1:] >= 1.0 - 1e-9).all()

    def test_deterministic_with_same_rng_state(self):
        hero = np.full(15, 89.5)
        rival = np.full(15, 90.0)
        v1, _ = gap_evolution(hero, rival, 0.7, 0.4, rng=random.Random(42))
        v2, _ = gap_evolution(hero, rival, 0.7, 0.4, rng=random.Random(42))
        assert np.allclose(v1, v2)

    def test_mismatched_lengths_raise(self):
        with pytest.raises(ValueError, match="gleich lang"):
            gap_evolution([90.0, 90.0], [90.0], initial_gap=1.0, p_overtake=0.5)

    def test_hero_already_ahead_never_blocks(self):
        """negativer Startabstand (hero schon vorn) darf nie in die
        Blockade-Logik laufen. das gilt selbst wenn rival deutlich
        schneller waere."""
        hero = np.full(10, 90.0)
        rival = np.full(10, 85.0)    # rival 5s/runde schneller
        verlauf, blockiert = gap_evolution(hero, rival, initial_gap=-2.0,
                                           p_overtake=0.0, block_gap_s=1.0,
                                           rng=random.Random(4))
        assert blockiert == 0
        assert verlauf[-1] == pytest.approx(-2.0 + 10 * 5.0)


# --------------------------------------------------------------- traffic_cost
class TestTrafficCost:
    """erwarteter Zeitverlust durch Verkehr per Monte Carlo (P41)."""

    def test_zero_when_never_close(self):
        hero = np.full(10, 90.0)
        rival = np.full(10, 90.0)
        mittel, se = traffic_cost(hero, rival, initial_gap=50.0,
                                  p_overtake=0.3, n_sim=100, seed=1)
        assert mittel == pytest.approx(0.0, abs=1e-9)
        assert se == pytest.approx(0.0, abs=1e-9)

    def test_zero_when_overtake_always_succeeds(self):
        hero = np.full(20, 89.0)
        rival = np.full(20, 90.0)
        mittel, _ = traffic_cost(hero, rival, initial_gap=0.5,
                                 p_overtake=1.0, n_sim=100, seed=2)
        assert mittel == pytest.approx(0.0, abs=1e-6)

    def test_higher_p_overtake_costs_less(self):
        """monotonie: eine leichtere Strecke (hohe p_overtake) darf im
        Mittel nie mehr kosten als eine schwere (niedrige p_overtake)."""
        hero = np.full(30, 88.0)
        rival = np.full(30, 90.0)
        schwer, _ = traffic_cost(hero, rival, initial_gap=0.5, p_overtake=0.05,
                                 n_sim=1000, seed=5)
        leicht, _ = traffic_cost(hero, rival, initial_gap=0.5, p_overtake=0.6,
                                 n_sim=1000, seed=5)
        assert leicht <= schwer

    def test_deterministic_with_seed(self):
        hero = np.full(20, 89.0)
        rival = np.full(20, 90.0)
        a = traffic_cost(hero, rival, 0.5, 0.2, n_sim=300, seed=7)
        b = traffic_cost(hero, rival, 0.5, 0.2, n_sim=300, seed=7)
        assert a == b

    def test_cost_is_never_negative(self):
        """verkehr kann hero nur bremsen, nie beschleunigen. der
        Zeitverlust ist per Konstruktion >= 0."""
        hero = np.full(25, 89.0)
        rival = np.full(25, 90.0)
        mittel, _ = traffic_cost(hero, rival, initial_gap=0.5, p_overtake=0.2,
                                 n_sim=500, seed=9)
        assert mittel >= -1e-9


# --------------------------------------------------------- track_curvature
class TestTrackCurvature:
    """kruemmung aus X/Y-Ideallinie (P37)."""

    def test_straight_line_has_zero_curvature(self):
        x = np.linspace(0, 100, 101)
        y = np.zeros_like(x)
        kappa = track_curvature(x, y, x)
        assert np.max(np.abs(kappa[5:-5])) == pytest.approx(0.0, abs=1e-9)

    def test_circle_matches_one_over_radius(self):
        """auf einem Kreis mit Radius R ist kappa(s) konstant 1/R."""
        R = 50.0
        theta = np.linspace(0, np.pi, 300)
        x, y, dist = R * np.cos(theta), R * np.sin(theta), R * theta
        kappa = track_curvature(x, y, dist, window=15)
        mitte = kappa[50:-50]
        assert mitte.mean() == pytest.approx(1.0 / R, rel=1e-5)


# ------------------------------------------------------------- simulate_lap
class TestSimulateLap:
    """quasi-stationaere Punktmassen-Rundenzeitsimulation (P37)."""

    def test_straight_line_runs_at_top_speed(self):
        """ohne Kruemmung ist v_top die einzige Grenze. die Geschwindigkeit
        ist konstant und die Rundenzeit ist exakt Distanz/v_top."""
        dist = np.linspace(0, 1000, 200)
        kappa = np.zeros_like(dist)
        v, t = simulate_lap(dist, kappa, mu_g=20.0, a_accel=8.0, a_brake=25.0,
                            v_top=80.0)
        assert np.allclose(v, 80.0)
        assert t == pytest.approx(1000 / 80.0)

    def test_constant_curvature_runs_at_grip_limited_speed(self):
        """bei konstanter Kruemmung ist die kurvengrip-begrenzte
        Geschwindigkeit sqrt(mu_g/kappa) ueberall bindend. das gilt
        unabhaengig von den Beschleunigungsgrenzen."""
        mu_g, kappa_wert = 20.0, 0.02
        dist = np.linspace(0, 500, 200)
        kappa = np.full_like(dist, kappa_wert)
        v, t = simulate_lap(dist, kappa, mu_g=mu_g, a_accel=8.0, a_brake=25.0,
                            v_top=200.0)
        erwartet = np.sqrt(mu_g / kappa_wert)
        assert np.allclose(v, erwartet, atol=1e-6)
        assert t == pytest.approx(500 / erwartet)

    def test_lower_brake_deceleration_starts_braking_earlier(self):
        """der Rueckwaertspass ist bremsbegrenzt: mit schwaecherer
        Bremsverzoegerung muss frueher vor der Kurve gebremst werden."""
        dist = np.linspace(0, 300, 600)
        kappa = np.where(dist < 200, 1e-6, 0.0125)
        v_top = 100.0
        v_schwach, _ = simulate_lap(dist, kappa, mu_g=20.0, a_accel=10.0,
                                    a_brake=30.0, v_top=v_top)
        v_stark, _ = simulate_lap(dist, kappa, mu_g=20.0, a_accel=10.0,
                                  a_brake=60.0, v_top=v_top)
        beginn_schwach = dist[np.argmax(v_schwach < v_top - 1)]
        beginn_stark = dist[np.argmax(v_stark < v_top - 1)]
        assert 0 < beginn_schwach < beginn_stark < 200


# -------------------------------------------------------- calibrate_lap_model
class TestCalibrateLapModel:
    """kleinste-Quadrate-Kalibrierung der vier Fahrzeugparameter (P37)."""

    def test_recovers_known_parameters_from_noiseless_trace(self):
        """aus einer mit bekannten Parametern simulierten (rauschfreien)
        Geschwindigkeitsspur muss die Kalibrierung dieselben Parameter
        zurueckgewinnen. der Rundtrip simulate_lap -> calibrate_lap_model
        ist die staerkste Pruefung fuer diese Funktion."""
        wahr = {"mu_g": 15.0, "a_accel": 9.0, "a_brake": 28.0, "v_top": 90.0}
        dist = np.linspace(0, 1500, 750)
        kappa = 1e-5 + 0.05 * (np.sin(dist / 120.0) ** 8)
        v_echt, _ = simulate_lap(dist, kappa, **wahr)

        params = calibrate_lap_model(dist, kappa, v_echt)

        assert params["mu_g"] == pytest.approx(wahr["mu_g"], abs=1e-3)
        assert params["a_accel"] == pytest.approx(wahr["a_accel"], abs=1e-3)
        assert params["a_brake"] == pytest.approx(wahr["a_brake"], abs=1e-3)
        assert params["rmse_ms"] == pytest.approx(0.0, abs=1e-3)


# ----------------------------------------------------------- simulate_stint
class TestSimulateStint:
    """rundenzeit ueber einen Stint mit sinkender Kraftstoffmasse (P37)."""

    def _strecke(self):
        dist = np.linspace(0, 1000, 400)
        kappa = 1e-5 + 0.04 * (np.sin(dist / 80.0) ** 8)
        return dist, kappa

    def test_laptimes_fall_monotonically_as_tank_empties(self):
        """mit vollerem Tank ist das Auto langsamer. mehr verbrannter
        Kraftstoff macht die Runde schneller und nie langsamer."""
        dist, kappa = self._strecke()
        zeiten = simulate_stint(dist, kappa, mu_g_ref=20.0, a_accel_ref=10.0,
                                a_brake_ref=28.0, v_top=90.0,
                                fuel_start_kg=100.0, n_laps=40)
        assert np.all(np.diff(zeiten) <= 1e-9)
        assert zeiten[0] > zeiten[-1]

    def test_empty_tank_throughout_matches_plain_simulate_lap(self):
        """ohne Kraftstoff (fuel_start_kg=0) ist jede Runde identisch zur
        unskalierten simulate_lap-Rundenzeit. die Skalierung m_dry/(m_dry+0)
        ist 1."""
        dist, kappa = self._strecke()
        params = dict(mu_g=20.0, a_accel=10.0, a_brake=28.0, v_top=90.0)
        _, t_erwartet = simulate_lap(dist, kappa, **params)

        zeiten = simulate_stint(dist, kappa, mu_g_ref=params["mu_g"],
                                a_accel_ref=params["a_accel"],
                                a_brake_ref=params["a_brake"],
                                v_top=params["v_top"], fuel_start_kg=0.0,
                                n_laps=10)
        assert np.allclose(zeiten, t_erwartet)

    def test_laptimes_constant_once_tank_is_dry(self):
        """sobald der errechnete Rest-Kraftstoff 0 erreicht aendert sich die
        Rundenzeit nicht mehr. der Tank kann nicht negativ werden."""
        dist, kappa = self._strecke()
        zeiten = simulate_stint(dist, kappa, mu_g_ref=20.0, a_accel_ref=10.0,
                                a_brake_ref=28.0, v_top=90.0,
                                fuel_start_kg=10.0, kg_per_lap=5.0, n_laps=6)
        assert zeiten[2] == pytest.approx(zeiten[3])
        assert zeiten[3] == pytest.approx(zeiten[5])


class TestSiegGrund:
    def test_no_rival_means_start_advantage(self):
        assert sieg_grund(None, False, False, False, False) == "Start-Vorteil"

    def test_taking_the_lead_on_lap_one_also_counts_as_start_advantage(self):
        """runde 1 ist noch der start, kein "wechsel" im eigentlichen sinn."""
        assert sieg_grund(1, False, False, False, False) == "Start-Vorteil"

    def test_rival_dnf_wins_over_everything_else(self):
        assert sieg_grund(30, True, True, True, True, True) == "Ausfall des Rivalen"

    def test_safety_car_beats_pit_stop_and_overtake(self):
        assert sieg_grund(30, False, True, True, True, True) == "Safety-Car-Wende"

    def test_pit_stop_beats_overtake_and_collapse(self):
        assert sieg_grund(30, False, False, True, True, True) == "Strategie/Boxenstopp"

    def test_real_overtake_beats_collapse(self):
        assert sieg_grund(30, False, False, False, True, True) == "Erkaempft auf der Strecke"

    def test_collapse_only_when_nothing_harder_applies(self):
        """rivale_bricht_ein ist ein weicher hinweis (reine positions-
        beobachtung) und deshalb absichtlich der am niedrigsten priorisierte
        grund - er darf einen boxenstopp/Safety-Car/DNF/echte Ueberholung nicht
        ueberschreiben (siehe Docstring, Niederlande 2024)."""
        assert sieg_grund(30, False, False, False, False, True) == "Einbruch des Rivalen"

    def test_unexplained_fallback_when_no_signal_fires(self):
        """kein signal traf zu, obwohl ein rivale existierte - ehrlich als
        offen kennzeichnen statt eine falsche kategorie zu erzwingen."""
        assert sieg_grund(30, False, False, False, False) == "Ungeklaert"
