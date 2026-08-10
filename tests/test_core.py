"""Tests fuer f1lab.core.

Laufen ohne Netzzugriff: alle Eingaben sind synthetisch mit bekannter
Wahrheit. Damit ist die Rechnung pruefbar, ohne von der Verfuegbarkeit der
F1-API abzuhaengen.

    pytest -q
"""
from __future__ import annotations

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
        """Ein Ausreisser darf den Median kaum bewegen - das ist der Grund,
        warum hier nicht der Mittelwert steht."""
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
        """Definierender Referenzwert des Elo-Systems."""
        assert elo_expected(1900, 1500) == pytest.approx(0.909, abs=1e-3)


class TestEloUpdate:
    def test_win_raises_rating(self):
        new_a, _ = elo_update(1500, 1500, score_a=1, k=24)
        assert new_a > 1500

    def test_loss_lowers_rating(self):
        new_a, _ = elo_update(1500, 1500, score_a=0, k=24)
        assert new_a < 1500

    def test_zero_sum(self):
        """Elo verschiebt Punkte nur - der Pool bleibt konstant."""
        new_a, new_b = elo_update(1520, 1480, score_a=1, k=24)
        assert (new_a - 1520) == pytest.approx(-(new_b - 1480))

    def test_upset_moves_more_than_expected_win(self):
        """Ein ueberraschender Sieg bewegt das Rating staerker als ein
        erwarteter - die Ueberraschung steckt in der Differenz zur
        erwarteten Punktzahl, nicht im Sieg selbst."""
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
        """Am Rennende ist der Tank leer - dort ist die Korrektur null."""
        out = fuel_correct([90.0], [50], total_laps=50)
        assert out[0] == pytest.approx(90.0)

    def test_first_lap_gets_largest_correction(self):
        out = fuel_correct([90.0] * 3, [1, 25, 50], total_laps=50)
        assert out[0] < out[1] < out[2]

    def test_magnitude_matches_formula(self):
        # 49 Runden Rest * 1.8 kg * 0.03 s = 2.646 s
        out = fuel_correct([90.0], [1], total_laps=50)
        assert out[0] == pytest.approx(90.0 - 2.646, abs=1e-6)

    def test_removes_artificial_speedup(self):
        """Konstante Reifen, nur Spritverbrauch: nach der Korrektur muessen
        die Zeiten flach sein."""
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
        assert not short.is_reliable          # zu wenige Runden

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
        """Flach bis Runde 12, danach steiler - der Knick muss gefunden
        werden und der zweite Abschnitt staerker steigen."""
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
        """Wenn der zweite Abschnitt flacher wird, ist das kein Cliff."""
        x = np.arange(1, 25)
        y = np.where(x <= 12, 90 + 0.25 * x, 90 + 0.25 * 12 + 0.02 * (x - 12))
        cliff, _, right = find_cliff(x, y)
        assert cliff is None and right is None


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
        """Synthetische Runde: zwei Bremszonen, dazwischen Vollgas."""
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


class TestMatchByDistance:
    def test_exact_match(self):
        assert match_by_distance([100, 500], [100, 500], tolerance=1) == \
            [(0, 0), (1, 1)]

    def test_within_tolerance(self):
        assert match_by_distance([100], [107], tolerance=10) == [(0, 0)]

    def test_outside_tolerance_stays_unmatched(self):
        assert match_by_distance([100], [200], tolerance=10) == []

    def test_extra_value_in_b_is_ignored(self):
        """Eine zusaetzliche Bremsung eines Fahrers ist kein Fehlerfall."""
        paare = match_by_distance([100, 500], [100, 300, 500], tolerance=5)
        assert paare == [(0, 0), (1, 2)]

    def test_no_double_booking(self):
        """Zwei nahe a-Werte duerfen sich nicht denselben b-Wert teilen."""
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
    """Streckenlaenge aus Positionsdaten."""

    def test_unit_square_closed(self):
        """Vier Seiten a 1 -> Umfang 4. Das Schlusssegment zaehlt mit."""
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
        """Ein feines Polygon naehert den Kreisumfang von unten an."""
        t = np.linspace(0, 2 * np.pi, 2000, endpoint=False)
        got = path_length(np.cos(t) * 100, np.sin(t) * 100)
        assert got == pytest.approx(2 * np.pi * 100, rel=1e-5)

    def test_resolution_does_not_change_result(self):
        """Doppelt so viele Stuetzpunkte auf derselben Geraden -> gleiche Laenge."""
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


# --------------------------------------------------------------- elevation
class TestElevationProfile:
    """Hoehenmeter mit Hysterese gegen Messrauschen."""

    def test_monotonic_climb(self):
        got = elevation_profile(np.arange(0, 51, 1.0))
        assert got.gain == pytest.approx(50.0)
        assert got.drop == pytest.approx(0.0)
        assert got.span == pytest.approx(50.0)

    def test_climb_then_descent_is_balanced(self):
        """Runde endet, wo sie beginnt -> Anstieg gleich Abstieg."""
        z = np.r_[np.arange(0, 31, 1.0), np.arange(29, -1, -1.0)]
        got = elevation_profile(z)
        assert got.gain == pytest.approx(30.0)
        assert got.drop == pytest.approx(30.0)
        assert got.span == pytest.approx(30.0)

    @staticmethod
    def _messrauschen(rng, n, sigma=0.3, window=25):
        """Korreliertes Rauschen, wie Positionsdaten es tatsaechlich zeigen.

        Weisses Rauschen waere das falsche Modell: aufeinanderfolgende Proben
        eines Positionskanals haengen zusammen, sie springen nicht unabhaengig.
        """
        glatt = np.ones(window) / window
        return np.convolve(rng.normal(0, sigma, n + window), glatt,
                           mode="valid")[:n]

    def test_noise_below_threshold_is_ignored(self):
        """Der eigentliche Zweck: Rauschen darf keine Hoehenmeter erzeugen."""
        z = self._messrauschen(np.random.default_rng(3), 5000)
        got = elevation_profile(z, min_step=1.0)
        assert got.gain == 0.0
        assert got.drop == 0.0

    def test_naive_sum_would_massively_overcount(self):
        """Belegt, warum die Hysterese noetig ist."""
        z = self._messrauschen(np.random.default_rng(3), 5000)
        naiv = float(np.abs(np.diff(z)).sum())
        assert naiv > 50                         # ohne Schwelle: frei erfunden
        assert elevation_profile(z, min_step=1.0).gain == 0.0

    def test_white_noise_still_leaks_above_threshold(self):
        """Grenze des Verfahrens, bewusst festgehalten.

        Weisses Rauschen mit sigma=0.3 ueberschreitet die 1-m-Schwelle ueber
        5000 Proben hunderte Male. Die Hysterese daempft das deutlich, setzt
        es aber nicht auf null - wer die Funktion auf ungeglaettete Daten
        loslaesst, muss min_step anheben.
        """
        rng = np.random.default_rng(0)
        z = rng.normal(0, 0.3, 5000)
        naiv = float(np.abs(np.diff(z)).sum())
        gedaempft = elevation_profile(z, min_step=1.0).gain
        assert gedaempft < naiv / 5              # klar besser als naiv
        assert gedaempft > 0                     # aber eben nicht null
        assert elevation_profile(z, min_step=2.0).gain == 0.0

    def test_real_climb_survives_noise(self):
        """Ein echter Anstieg darf durch ueberlagertes Rauschen nicht verschwinden."""
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
    """Sparses Aenderungs-Log (wie FastF1s track_status) in Intervalle."""

    def test_end_is_next_start_not_own_last_occurrence(self):
        """Der eigentliche Zweck: 'SCDeployed' erscheint einmal, gilt aber
        bis zum naechsten Statuswechsel - nicht nur an seinem eigenen
        Zeitpunkt (das waere ein Intervall der Laenge 0)."""
        status = ["1", "4", "1"]
        zeit = [0.0, 100.0, 500.0]
        s, start, ende = status_intervals(status, zeit)
        assert list(s) == ["1", "4", "1"]
        assert list(start) == [0.0, 100.0, 500.0]
        assert ende[0] == pytest.approx(100.0)
        assert ende[1] == pytest.approx(500.0)
        assert np.isnan(ende[2])                  # letztes Intervall offen

    def test_immediate_repeats_are_merged(self):
        status = ["1", "1", "4", "1"]
        zeit = [0.0, 10.0, 100.0, 500.0]
        s, start, ende = status_intervals(status, zeit)
        assert list(s) == ["1", "4", "1"]
        assert list(start) == [0.0, 100.0, 500.0]

    def test_naive_groupby_would_give_zero_length(self):
        """Dokumentiert den Fehler, den die Funktion vermeidet: Gruppieren
        nach *aufeinanderfolgend* gleichem Status (cumsum ueber Wechsel) und
        dessen eigenem Min/Max ergibt hier ueberall Laenge 0, weil kein
        Status zweimal hintereinander steht - das war die urspruengliche
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
        # Runden 1..3: 90, 91, 92
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
        # mit ausgeschalteter Regel geht es
        RaceConfig(n_laps=10, pit_loss=20.0,
                  tyres=(TyreModel("SOFT", 90.0, 0.05),),
                  require_two_compounds=False)

    def test_unknown_start_compound_raises(self):
        with pytest.raises(ValueError, match="unbekannte Startmischung"):
            RaceConfig(n_laps=10, pit_loss=20.0, tyres=self._tyres(),
                      start_compound="MEDIUM")

    def test_fuel_offset_is_zero_without_fuel_effect(self):
        cfg = RaceConfig(n_laps=20, pit_loss=20.0, tyres=self._tyres())
        assert cfg.fuel_offset == pytest.approx(0.0)

    def test_fuel_offset_scales_with_laps_squared(self):
        cfg = RaceConfig(n_laps=10, pit_loss=20.0, tyres=self._tyres(),
                         fuel_effect=0.1)
        # 0.1 * 10 * 9 / 2 = 4.5
        assert cfg.fuel_offset == pytest.approx(4.5)


class TestOptimalStrategy:
    """Kleine, von Hand nachrechenbare Rennen."""

    def _cfg(self, **kw):
        defaults = dict(
            n_laps=10, pit_loss=20.0, min_stint=1,
            tyres=(TyreModel("SOFT", 90.0, 1.0), TyreModel("HARD", 91.0, 0.1)))
        defaults.update(kw)
        return RaceConfig(**defaults)

    def test_prefers_flatter_degrading_tyre_for_long_stint(self):
        """SOFT baut zehnmal so schnell ab wie HARD - bei freier Stintlaenge
        gewinnt ein frueher Wechsel auf HARD."""
        cfg = self._cfg()
        best = optimal_strategy(cfg)
        # das laengste Stueck des Rennens sollte auf HARD liegen
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

    def test_more_pit_stops_never_beats_pit_loss_savings(self):
        """Ein Vergleichsplan mit einem zusaetzlichen, unnoetigen Stopp kann
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
        """Ein guenstigerer Boxenstopp macht zusaetzliche Stopps attraktiver -
        die Stoppzahl bei niedrigem Pitloss ist mindestens so hoch wie bei
        hohem."""
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
        """Runde 1 startet immer gruen."""
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


class TestExpectedCostAndHindsight:
    def _cfg(self):
        return RaceConfig(n_laps=20, pit_loss=20.0, min_stint=3,
                          tyres=(TyreModel("SOFT", 90.0, 0.3),
                                TyreModel("HARD", 91.0, 0.05)))

    def test_expected_cost_without_sc_equals_green_time(self):
        """Ohne jede Safety-Car-Wahrscheinlichkeit gibt es keinen Rabatt -
        die erwarteten Kosten sind genau die Fahrzeit des festen Plans."""
        cfg = self._cfg()
        plan = optimal_strategy(cfg)
        prozess = SafetyCarProcess(p_deploy=0.0, p_end=0.5)
        kosten = expected_cost_of_plan(cfg, plan, prozess)
        assert kosten == pytest.approx(plan.green_time)

    def test_hindsight_never_worse_than_expected_cost_of_fixed_plan(self):
        """Das Optimum bei bekanntem Verlauf ist eine untere Schranke - im
        Mittel nie schlechter als ein fester Plan, der stur durchgezogen
        wird."""
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
