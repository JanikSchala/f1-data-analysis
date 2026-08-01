"""Tests fuer f1lab.session, soweit ohne Netzzugriff moeglich.

Die Ladefunktionen brauchen die F1-API und werden hier nicht getestet.
Was sich aber pruefen laesst, ist die Behandlung der Rohspalten - und genau
dort lauern die Fallstricke, weil der Live-Timing-Feed Lueckenhaftes liefert.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from f1lab.session import TRACK_STATUS, not_deleted_mask


class TestNotDeletedMask:
    """Regressionstests fuer einen Absturz in clean_laps().

    FastF1s pick_not_deleted() invertiert die Spalte 'Deleted' direkt mit ~.
    Ist die Spalte object-dtype - was passiert, sobald None darin steht -
    wirft pandas einen TypeError. In Barcelona 2024 ist genau das der Fall.
    """

    def test_plain_booleans(self):
        out = not_deleted_mask(pd.Series([True, False, False, True]))
        assert out.tolist() == [False, True, True, False]

    def test_none_counts_as_not_deleted(self):
        """Keine Meldung der Rennleitung heisst: Runde zaehlt."""
        out = not_deleted_mask(pd.Series([None, True, None, False]))
        assert out.tolist() == [True, False, True, True]

    def test_object_dtype_does_not_raise(self):
        """Der Fall, der pick_not_deleted() zum Absturz bringt."""
        s = pd.Series([None, True, False], dtype=object)
        with pytest.raises(TypeError):
            _ = ~s                      # so macht es FastF1
        assert not_deleted_mask(s).tolist() == [True, False, True]

    def test_nullable_boolean_dtype(self):
        s = pd.Series([True, None, False], dtype="boolean")
        assert not_deleted_mask(s).tolist() == [False, True, True]

    def test_all_none(self):
        out = not_deleted_mask(pd.Series([None] * 5, dtype=object))
        assert out.all()

    def test_empty_series(self):
        assert not_deleted_mask(pd.Series([], dtype=object)).empty

    def test_mask_length_matches_input(self):
        s = pd.Series([None, True, False, None, True], dtype=object)
        assert len(not_deleted_mask(s)) == len(s)

    def test_no_pandas_warning(self):
        """Kein FutureWarning - sonst bricht das Verhalten mit pandas 3."""
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("error", FutureWarning)
            not_deleted_mask(pd.Series([None, True, False], dtype=object))

    def test_usable_as_numpy_index(self):
        """So wird die Maske in clean_laps() eingesetzt."""
        df = pd.DataFrame({
            "LapTime": [90.1, 90.2, 90.3, 90.4],
            "Deleted": pd.Series([None, True, None, False], dtype=object),
        })
        kept = df[not_deleted_mask(df["Deleted"]).to_numpy()]
        assert kept["LapTime"].tolist() == [90.1, 90.3, 90.4]


class TestTrackStatus:
    def test_known_codes(self):
        assert TRACK_STATUS["1"] == "gruen"
        assert TRACK_STATUS["4"] == "safety car"
        assert TRACK_STATUS["6"] == "vsc"

    def test_codes_are_strings(self):
        """Der Feed liefert Strings, keine Zahlen - ein Mapping auf int
        wuerde still nichts treffen."""
        assert all(isinstance(k, str) for k in TRACK_STATUS)

    def test_green_is_the_filter_used_everywhere(self):
        """clean_laps() filtert auf '1'. Sollte sich das Mapping aendern,
        faellt es hier auf."""
        green = [k for k, v in TRACK_STATUS.items() if v == "gruen"]
        assert green == ["1"]


class TestSessionApiSurface:
    """Stellt sicher, dass die oeffentliche Schnittstelle stabil bleibt."""

    def test_exports_exist(self):
        import f1lab
        for name in ("load", "clean_laps", "race_pace", "pace_table",
                     "degradation", "degradation_by_compound", "pit_loss",
                     "stints", "track_status_phases", "enable_cache"):
            assert hasattr(f1lab, name), f"f1lab.{name} fehlt"

    def test_core_functions_are_numpy_only(self):
        """core darf nicht von FastF1 abhaengen - sonst waeren die Tests
        ohne Netz nicht mehr moeglich."""
        import inspect

        import f1lab.core as core
        src = inspect.getsource(core)
        assert "import fastf1" not in src
        assert "from fastf1" not in src

    def test_pace_entry_is_sortable_by_median(self):
        from f1lab.core import Interval
        from f1lab.session import PaceEntry

        a = PaceEntry("VER", "Red Bull", 30, Interval(90.0, 89.8, 90.2))
        b = PaceEntry("NOR", "McLaren", 30, Interval(89.5, 89.3, 89.7))
        assert sorted([a, b], key=lambda e: e.median_s)[0].driver == "NOR"
        assert np.isclose(a.median_s, 90.0)
