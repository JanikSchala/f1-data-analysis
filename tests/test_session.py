"""tests fuer f1lab.session soweit ohne Netzzugriff moeglich.

die Ladefunktionen brauchen die F1-API und werden hier nicht getestet.
pruefbar ist aber die Behandlung der Rohspalten. genau dort lauern die
Fallstricke weil der Live-Timing-Feed Lueckenhaftes liefert.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import f1lab.session as session_mod
from f1lab.session import (
    TELEMETRY_MARKER,
    TIMING_MARKER,
    TRACK_STATUS,
    cache_ready,
    cached_sessions,
    find_cache,
    not_deleted_mask,
)


class TestNotDeletedMask:
    """regressionstest fuer einen Absturz in clean_laps().

    FastF1s pick_not_deleted() invertiert die Spalte 'Deleted' direkt mit ~.
    object-dtype macht das kaputt. das passiert sobald None in der Spalte
    steht. pandas wirft dann einen TypeError.
    """

    def test_plain_booleans(self):
        out = not_deleted_mask(pd.Series([True, False, False, True]))
        assert out.tolist() == [False, True, True, False]

    def test_none_counts_as_not_deleted(self):
        """keine Meldung der Rennleitung heisst: Runde zaehlt."""
        out = not_deleted_mask(pd.Series([None, True, None, False]))
        assert out.tolist() == [True, False, True, True]

    def test_object_dtype_does_not_raise(self):
        """genau dieser Fall bringt pick_not_deleted() zum Absturz."""
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
        """kein FutureWarning. sonst bricht das Verhalten mit pandas 3."""
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("error", FutureWarning)
            not_deleted_mask(pd.Series([None, True, False], dtype=object))

    def test_usable_as_numpy_index(self):
        """so wird die Maske in clean_laps() eingesetzt."""
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
        """der Feed liefert Strings, keine Zahlen. ein Mapping auf int
        wuerde still nichts treffen."""
        assert all(isinstance(k, str) for k in TRACK_STATUS)

    def test_green_is_the_filter_used_everywhere(self):
        """clean_laps() filtert auf '1'. ein geaendertes Mapping faellt
        hier auf."""
        green = [k for k, v in TRACK_STATUS.items() if v == "gruen"]
        assert green == ["1"]


class TestSessionApiSurface:
    """sichert die Stabilitaet der oeffentlichen Schnittstelle."""

    def test_exports_exist(self):
        import f1lab
        for name in ("load", "clean_laps", "race_pace", "pace_table",
                     "degradation", "degradation_by_compound", "pit_loss",
                     "stints", "track_status_phases", "enable_cache"):
            assert hasattr(f1lab, name), f"f1lab.{name} fehlt"

    def test_core_functions_are_numpy_only(self):
        """core darf nicht von FastF1 abhaengen. sonst waeren die Tests
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


class TestCacheInventory:
    """bestandsaufnahme des FastF1-Caches allein aus der Ordnerstruktur.

    die Oberflaeche soll nur auswertbare Sessions zur Auswahl stellen. ob
    eine Session im Cache liegt steht nirgends geschrieben. es ergibt sich
    aus den abgelegten Dateien. genau daran haengen hier die Erwartungen.
    """

    @staticmethod
    def _session(root, jahr, event_datum, event, ses_datum, ses,
                 timing=True, telemetry=False):
        d = root / str(jahr) / f"{event_datum}_{event}" / f"{ses_datum}_{ses}"
        d.mkdir(parents=True)
        if timing:
            (d / TIMING_MARKER).touch()
        if telemetry:
            (d / TELEMETRY_MARKER).touch()
        return d

    def test_leerer_pfad_gibt_leeren_rahmen(self, tmp_path):
        out = cached_sessions(tmp_path)
        assert out.empty
        assert list(out.columns) == ["season", "event", "event_date",
                                     "ident", "timing", "telemetry"]

    def test_ordnernamen_werden_zu_kennungen(self, tmp_path):
        self._session(tmp_path, 2024, "2024-09-01", "Italian_Grand_Prix",
                      "2024-09-01", "Race")
        self._session(tmp_path, 2024, "2024-09-01", "Italian_Grand_Prix",
                      "2024-08-31", "Qualifying")
        out = cached_sessions(tmp_path)
        assert set(out["ident"]) == {"R", "Q"}
        assert set(out["event"]) == {"Italian Grand Prix"}
        assert out["season"].tolist() == [2024, 2024]

    def test_sprint_quali_heisst_je_nach_jahr_anders(self, tmp_path):
        """2023 heisst es 'Sprint_Shootout', ab 2024 'Sprint_Qualifying'.
        beides ist dieselbe Session und muss auf dieselbe Kennung fallen."""
        self._session(tmp_path, 2023, "2023-04-30", "Azerbaijan_Grand_Prix",
                      "2023-04-29", "Sprint_Shootout")
        self._session(tmp_path, 2024, "2024-04-21", "Chinese_Grand_Prix",
                      "2024-04-20", "Sprint_Qualifying")
        assert cached_sessions(tmp_path)["ident"].tolist() == ["SQ", "SQ"]

    def test_leerer_ordner_zaehlt_nicht_als_geladen(self, tmp_path):
        """FastF1 legt den Ordner schon beim Anfassen an. ohne Timing-Datei
        ist die Session nicht auswertbar und darf nicht als geladen gelten."""
        self._session(tmp_path, 2024, "2024-09-01", "Italian_Grand_Prix",
                      "2024-08-30", "Practice_1", timing=False)
        out = cached_sessions(tmp_path)
        assert len(out) == 1
        assert not out["timing"].iloc[0]

    def test_telemetrie_wird_getrennt_ausgewiesen(self, tmp_path):
        """telemetrie ist ein eigener, viel groesserer Download. eine
        Session kann Timing haben und trotzdem keine Telemetrie."""
        self._session(tmp_path, 2024, "2024-09-01", "Italian_Grand_Prix",
                      "2024-09-01", "Race", telemetry=True)
        self._session(tmp_path, 2024, "2024-09-01", "Italian_Grand_Prix",
                      "2024-08-31", "Qualifying", telemetry=False)
        out = cached_sessions(tmp_path).set_index("ident")
        assert bool(out.loc["R", "telemetry"])
        assert not bool(out.loc["Q", "telemetry"])
        assert out["timing"].all()

    def test_fremde_ordner_stoeren_nicht(self, tmp_path):
        """neben den Jahresordnern liegen im Cache auch die HTTP-Datenbank
        und Systemdateien. die duerfen den Scan nicht aus dem Tritt bringen."""
        (tmp_path / "fastf1_http_cache.sqlite").touch()
        (tmp_path / ".DS_Store").touch()
        (tmp_path / "notizen").mkdir()
        self._session(tmp_path, 2024, "2024-09-01", "Italian_Grand_Prix",
                      "2024-09-01", "Race")
        assert len(cached_sessions(tmp_path)) == 1

    def test_sortiert_nach_saison_und_datum(self, tmp_path):
        self._session(tmp_path, 2024, "2024-09-01", "Italian_Grand_Prix",
                      "2024-09-01", "Race")
        self._session(tmp_path, 2018, "2018-05-13", "Spanish_Grand_Prix",
                      "2018-05-13", "Race")
        self._session(tmp_path, 2024, "2024-03-02", "Bahrain_Grand_Prix",
                      "2024-03-02", "Race")
        out = cached_sessions(tmp_path)
        assert out["season"].tolist() == [2018, 2024, 2024]
        assert out["event"].tolist() == ["Spanish Grand Prix",
                                         "Bahrain Grand Prix",
                                         "Italian Grand Prix"]


class TestFindCache:
    """der Cache liegt je nach Rechner woanders. gesucht wird in fester
    Reihenfolge und ein fehlender Cache ist kein Fehler."""

    def test_explizites_argument_gewinnt(self, tmp_path, monkeypatch):
        monkeypatch.setenv("F1_CACHE", str(tmp_path / "aus_env"))
        (tmp_path / "aus_env").mkdir()
        explizit = tmp_path / "explizit"
        explizit.mkdir()
        assert find_cache(explizit) == explizit

    def test_umgebungsvariable_vor_standardpfad(self, tmp_path, monkeypatch):
        aus_env = tmp_path / "aus_env"
        aus_env.mkdir()
        monkeypatch.setenv("F1_CACHE", str(aus_env))
        assert find_cache() == aus_env

    def test_ohne_treffer_none(self, tmp_path, monkeypatch):
        """kein Cache heisst noch kein Warmup gelaufen. die Oberflaeche
        soll das erklaeren koennen statt an einem Fehler zu scheitern."""
        monkeypatch.setenv("F1_CACHE", str(tmp_path / "gibt_es_nicht"))
        monkeypatch.setattr(session_mod, "CACHE_DIR", tmp_path / "auch_nicht")
        monkeypatch.setattr(
            session_mod, "__file__",
            str(tmp_path / "tief" / "f1lab" / "session.py"))
        assert find_cache() is None


class TestCacheReady:
    """regressionstest fuer einen echten Bug in f1analyze/data.py.

    load_session() rief bei jedem Aufruf enable_cache() ohne Argumente auf.
    das hat einen von aussen gesetzten Offline-Fixture-Cache (siehe
    tests/conftest.py dort) stillschweigend auf den Standardpfad
    zurueckgesetzt. cache_ready() gibt Aufrufern eine Moeglichkeit eine
    bereits gesetzte Konfiguration zu respektieren statt sie zu
    ueberschreiben."""

    def test_false_ohne_vorherigen_enable_cache_aufruf(self, monkeypatch):
        monkeypatch.setattr(session_mod, "_active_cache", None)
        assert cache_ready() is False

    def test_true_nach_enable_cache_aufruf(self, monkeypatch, tmp_path):
        monkeypatch.setattr(session_mod, "_active_cache", tmp_path)
        assert cache_ready() is True
