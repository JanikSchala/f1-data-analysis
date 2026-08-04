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
    Interval,
    bootstrap_median,
    braking_zones,
    elevation_profile,
    estimate_pit_loss,
    find_cliff,
    fit_degradation,
    fuel_correct,
    mad_outlier_mask,
    optimal_undercut_window,
    path_length,
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
