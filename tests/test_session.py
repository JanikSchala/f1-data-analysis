"""tests fuer f1lab.session soweit ohne Netzzugriff moeglich.

die Ladefunktionen brauchen die F1-API und werden hier nicht getestet.
pruefbar ist aber die Behandlung der Rohspalten. genau dort lauern die
Fallstricke weil der Live-Timing-Feed Lueckenhaftes liefert.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from fastf1.exceptions import ErgastInvalidRequestError

import f1lab.session as session_mod
from f1lab.session import (
    TELEMETRY_MARKER,
    TIMING_MARKER,
    TRACK_STATUS,
    _duelle,
    cache_ready,
    cached_sessions,
    compare_braking_zones,
    dirty_air_effect,
    ergast_retry,
    find_cache,
    not_deleted_mask,
    parse_penalties,
    parse_track_limits,
    sc_compaction,
    season_sessions,
    temperature_effect,
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


class TestErgastRetry:
    """die Wiederholung lag vorher sechsmal als eigene Kopie in den
    Skripten. alle sechs wiederholten auch den einen Fall, der sich durch
    Wiederholen nie aendert: ErgastInvalidRequestError ("Not Cached").
    fuer die laufende Saison kostete das bis zu 30s pro Aufruf und lieferte
    danach trotzdem nichts."""

    @pytest.fixture(autouse=True)
    def _keine_echten_pausen(self, monkeypatch):
        """Wartezeiten protokollieren statt sie abzusitzen."""
        self.pausen: list[float] = []
        monkeypatch.setattr(session_mod.time, "sleep", self.pausen.append)

    def test_erfolg_beim_ersten_versuch_wartet_nicht(self):
        assert ergast_retry(lambda: "da") == "da"
        assert self.pausen == []

    def test_wiederholt_nach_voruebergehendem_fehler(self):
        rufe = []

        def flatterhaft():
            rufe.append(1)
            if len(rufe) < 3:
                raise ConnectionError("429")
            return "endlich"

        assert ergast_retry(flatterhaft) == "endlich"
        assert len(rufe) == 3
        assert self.pausen == [3.0, 6.0]  # backoff waechst linear

    def test_versuche_ist_die_gesamtzahl_nicht_die_der_wiederholungen(self):
        rufe = []

        def immer_kaputt():
            rufe.append(1)
            raise ConnectionError("429")

        with pytest.raises(ConnectionError):
            ergast_retry(immer_kaputt, versuche=3)
        assert len(rufe) == 3

    def test_erschoepfte_versuche_werfen_standardmaessig_weiter(self):
        def immer_kaputt():
            raise ConnectionError("429")

        with pytest.raises(ConnectionError):
            ergast_retry(immer_kaputt, versuche=2)

    def test_erschoepfte_versuche_geben_none_wenn_gewuenscht(self):
        """fuer Scans ueber viele Saisons: ein fehlendes Jahr ueberspringen
        statt die ganze Auswertung abzubrechen (siehe P22/P46/P50)."""
        def immer_kaputt():
            raise ConnectionError("429")

        assert ergast_retry(immer_kaputt, versuche=2,
                            leer_bei_fehlschlag=True) is None

    def test_eindeutige_absage_wird_nicht_wiederholt(self):
        """der eigentliche Fund: die Antwort ist eindeutig, nicht flatterhaft."""
        rufe = []

        def nicht_gecacht():
            rufe.append(1)
            raise ErgastInvalidRequestError("Server response: 'Not Cached'")

        with pytest.raises(ErgastInvalidRequestError):
            ergast_retry(nicht_gecacht)
        assert len(rufe) == 1
        assert self.pausen == []

    def test_eindeutige_absage_auch_ohne_wiederholung_ueberspringbar(self):
        rufe = []

        def nicht_gecacht():
            rufe.append(1)
            raise ErgastInvalidRequestError("Server response: 'Not Cached'")

        assert ergast_retry(nicht_gecacht, leer_bei_fehlschlag=True) is None
        assert len(rufe) == 1
        assert self.pausen == []

    def test_argumente_gehen_unveraendert_durch(self):
        assert ergast_retry(lambda a, b=0: (a, b), 1, b=2) == (1, 2)


class TestSeasonSessions:
    """das Rueckgrat jedes Saison-Scans. lag vorher fuenfmal einzeln im Repo,
    jedes Mal mit demselben nackten except-continue."""

    class _Session:
        def __init__(self, results):
            self.results = results

    @pytest.fixture
    def kalender(self, monkeypatch):
        """zwei Saisons a zwei Runden, ohne Netz."""
        def fake_schedule(jahr, include_testing=False):
            assert include_testing is False   # Testfahrten sind keine Rennen
            return pd.DataFrame({"RoundNumber": [1, 2],
                                 "EventName": [f"GP {jahr}-1", f"GP {jahr}-2"]})
        monkeypatch.setattr(session_mod.fastf1, "get_event_schedule",
                            fake_schedule)

    def _lade(self, monkeypatch, verhalten):
        """verhalten: (jahr, runde) -> Session, None (leer) oder Exception."""
        aufrufe = []

        def fake_load(jahr, rnd, identifier, **kwargs):
            aufrufe.append((jahr, rnd, identifier, kwargs))
            was = verhalten(jahr, rnd)
            if isinstance(was, Exception):
                raise was
            return was
        monkeypatch.setattr(session_mod, "load", fake_load)
        return aufrufe

    def test_liefert_jede_ladbare_session(self, kalender, monkeypatch):
        voll = self._Session(pd.DataFrame({"Position": [1]}))
        self._lade(monkeypatch, lambda j, r: voll)
        got = list(season_sessions([2023, 2024]))
        assert [(j, r) for j, r, _, _ in got] == [
            (2023, 1), (2023, 2), (2024, 1), (2024, 2)]
        assert got[0][2]["EventName"] == "GP 2023-1"

    def test_nicht_ladbare_runde_wird_uebersprungen(self, kalender, monkeypatch):
        """der Cache deckt nie jede Session ab - das ist der Normalfall."""
        voll = self._Session(pd.DataFrame({"Position": [1]}))
        self._lade(monkeypatch,
                   lambda j, r: ValueError("nicht im Cache") if r == 1 else voll)
        got = list(season_sessions([2024]))
        assert [(j, r) for j, r, _, _ in got] == [(2024, 2)]

    def test_leere_ergebnistabelle_wird_uebersprungen(self, kalender, monkeypatch):
        leer = self._Session(pd.DataFrame())
        voll = self._Session(pd.DataFrame({"Position": [1]}))
        self._lade(monkeypatch, lambda j, r: leer if r == 1 else voll)
        assert [r for _, r, _, _ in season_sessions([2024])] == [2]

    def test_leere_ergebnisse_bleiben_wenn_gewuenscht(self, kalender, monkeypatch):
        """Auswertungen auf laps statt results brauchen die Session trotzdem."""
        leer = self._Session(pd.DataFrame())
        self._lade(monkeypatch, lambda j, r: leer)
        got = list(season_sessions([2024], mit_ergebnis=False))
        assert len(got) == 2

    def test_identifier_geht_durch(self, kalender, monkeypatch):
        voll = self._Session(pd.DataFrame({"Position": [1]}))
        aufrufe = self._lade(monkeypatch, lambda j, r: voll)
        list(season_sessions([2024], "Q"))
        assert {a[2] for a in aufrufe} == {"Q"}

    def test_laedt_ohne_telemetrie_wetter_meldungen(self, kalender, monkeypatch):
        """ein Scan ueber drei Jahre braucht die Ergebnistabelle, nicht den
        um ein Vielfaches groesseren Rest."""
        voll = self._Session(pd.DataFrame({"Position": [1]}))
        aufrufe = self._lade(monkeypatch, lambda j, r: voll)
        list(season_sessions([2024]))
        for _, _, _, kwargs in aufrufe:
            assert kwargs == {"telemetry": False, "weather": False,
                              "messages": False}

    def test_ist_ein_generator_laedt_erst_beim_iterieren(self, kalender,
                                                         monkeypatch):
        """sonst wuerde ein Scan ueber viele Saisons alles auf einmal laden."""
        voll = self._Session(pd.DataFrame({"Position": [1]}))
        aufrufe = self._lade(monkeypatch, lambda j, r: voll)
        gen = season_sessions([2024])
        assert aufrufe == []
        next(gen)
        assert len(aufrufe) == 1


class TestRaceControlParser:
    """die Strafen-Regex traf bei ihrer Einfuehrung 0 von 6 echten Meldungen
    (siehe P19). die Beispiele hier sind deshalb keine erfundenen Strings,
    sondern woertliche Meldungen aus Oesterreich 2024."""

    ECHTE_STRAFEN = [
        "FIA STEWARDS: 10 SECOND TIME PENALTY FOR CAR 14 (ALO) - CAUSING A COLLISION",
        "FIA STEWARDS: 5 SECOND TIME PENALTY FOR CAR 11 (PER) - SPEEDING IN THE PIT LANE",
        "FIA STEWARDS: 5 SECOND TIME PENALTY FOR CAR 4 (NOR) - TRACK LIMITS",
        "FIA STEWARDS: 10 SECOND TIME PENALTY FOR CAR 1 (VER) - CAUSING A COLLISION",
    ]
    ECHTE_LIMITS = [
        "CAR 3 (RIC) LAP DELETED - TRACK LIMITS AT TURN 3 LAP 1 15:03:32",
        "CAR 27 (HUL) TIME 1:29.202 DELETED - TRACK LIMITS AT TURN 3 LAP 12 15:16:58",
        "CAR 4 (NOR) TIME 1:11.751 DELETED - TRACK LIMITS AT TURN 3 LAP 10 15:14:03",
    ]

    @staticmethod
    def _rcm(nachrichten):
        return pd.DataFrame({"Lap": list(range(1, len(nachrichten) + 1)),
                             "Message": nachrichten})

    def test_echte_strafmeldungen_werden_alle_getroffen(self):
        got = parse_penalties(self._rcm(self.ECHTE_STRAFEN))
        assert len(got) == len(self.ECHTE_STRAFEN)
        assert got["driver"].tolist() == ["ALO", "PER", "NOR", "VER"]
        assert got["nr"].tolist() == ["14", "11", "4", "1"]

    def test_praefix_vor_der_strafe_stoert_nicht(self):
        """echte Meldungen beginnen mit "FIA STEWARDS: ". ein Wechsel von
        search() auf match() wuerde genau hier alles stillschweigend
        verlieren - das war der urspruengliche Fund in P19."""
        ohne = "10 SECOND TIME PENALTY FOR CAR 14 (ALO) - CAUSING A COLLISION"
        mit = "FIA STEWARDS: " + ohne
        assert len(parse_penalties(self._rcm([ohne]))) == 1
        assert len(parse_penalties(self._rcm([mit]))) == 1

    def test_strafmass_und_grund_getrennt(self):
        got = parse_penalties(self._rcm([self.ECHTE_STRAFEN[1]]))
        assert got["strafmass"].iloc[0] == "5 SECOND TIME PENALTY"
        assert got["grund"].iloc[0] == "SPEEDING IN THE PIT LANE"

    def test_grund_mit_gedankenstrich_bleibt_vollstaendig(self):
        """ein echter Grund enthaelt einen Halbgeviertstrich, nicht nur
        ASCII - er darf den Text nicht abschneiden."""
        msg = ("FIA STEWARDS: 5 SECOND TIME PENALTY FOR CAR 44 (HAM) - FAILING "
               "TO FOLLOW RACE DIRECTORS INSTRUCTIONS – CROSSING THE LINE "
               "AT PIT ENTRY")
        got = parse_penalties(self._rcm([msg]))
        assert got["grund"].iloc[0].endswith("CROSSING THE LINE AT PIT ENTRY")

    def test_andere_meldungen_ergeben_keine_strafe(self):
        harmlos = ["GREEN LIGHT - PIT EXIT OPEN", "DRS ENABLED",
                   "CAR 3 (RIC) LAP DELETED - TRACK LIMITS AT TURN 3 LAP 1"]
        assert parse_penalties(self._rcm(harmlos)).empty

    def test_echte_track_limit_meldungen(self):
        got = parse_track_limits(self._rcm(self.ECHTE_LIMITS))
        assert got["driver"].tolist() == ["RIC", "HUL", "NOR"]
        assert got["turn"].tolist() == [3, 3, 3]

    def test_rundenzeit_in_der_meldung_verwirrt_die_kurve_nicht(self):
        """"TIME 1:29.202" steht vor der Kurvennummer und enthaelt selbst
        Ziffern - die Kurve muss trotzdem 3 sein, nicht 1 oder 29."""
        got = parse_track_limits(self._rcm([self.ECHTE_LIMITS[1]]))
        assert got["turn"].iloc[0] == 3

    def test_leeres_ergebnis_behaelt_seine_spalten(self):
        """ein Rennen ohne Strafen ist normal. ein spaltenloses DataFrame
        wuerde Aufrufer mit einem KeyError treffen statt mit einer leeren
        Tabelle - genau der Fehler, der sieg_attribution() schon einmal
        zerlegt hat."""
        leer = self._rcm(["GREEN LIGHT - PIT EXIT OPEN"])
        pen, lim = parse_penalties(leer), parse_track_limits(leer)
        assert pen.empty and lim.empty
        assert list(pen.columns) == ["lap", "strafmass", "nr", "driver", "grund"]
        assert list(lim.columns) == ["lap", "nr", "driver", "turn"]
        # so greifen die Aufrufer darauf zu (13_RaceControl.py, P19)
        assert pen.groupby("driver").size().empty
        assert lim.groupby("turn").size().empty


class TestCompareBrakingZones:
    """paart die Bremszonen zweier Fahrer ueber die Distanz."""

    @staticmethod
    def _zonen(*starts):
        return pd.DataFrame({"start_m": list(starts)})

    def test_paare_nach_naehe(self):
        got = compare_braking_zones(self._zonen(100.0, 900.0),
                                    self._zonen(120.0, 880.0))
        assert got["start_m_a"].tolist() == [100.0, 900.0]
        assert got["delta_m"].tolist() == [20.0, -20.0]

    def test_nach_distanz_sortiert(self):
        got = compare_braking_zones(self._zonen(900.0, 100.0),
                                    self._zonen(880.0, 120.0))
        assert got["start_m_a"].tolist() == sorted(got["start_m_a"].tolist())

    def test_ausserhalb_der_toleranz_kein_paar(self):
        got = compare_braking_zones(self._zonen(100.0), self._zonen(400.0),
                                    tolerance_m=150.0)
        assert got.empty

    def test_kein_einziges_paar_stuerzt_nicht_ab(self):
        """beide Fahrer haben Zonen, nur passt keine zur anderen. vorher
        brach das Sortieren hier mit KeyError('start_m_a') ab - erreichbar
        ueber die Fahrerauswahl auf der Telemetrie-Seite."""
        got = compare_braking_zones(self._zonen(100.0, 900.0),
                                    self._zonen(500.0, 1400.0),
                                    tolerance_m=150.0)
        assert got.empty
        assert list(got.columns) == ["start_m_a", "start_m_b", "delta_m"]

    def test_leere_eingabe_behaelt_spalten(self):
        got = compare_braking_zones(self._zonen(), self._zonen(100.0))
        assert list(got.columns) == ["start_m_a", "start_m_b", "delta_m"]


class TestScCompaction:
    """Feldstreckung vor gegen waehrend einer Neutralisation (siehe P18)."""

    @staticmethod
    def _phase(start, ende):
        return pd.DataFrame([{"lap_start": start, "lap_end": ende}])

    def test_baseline_ist_median_der_drei_runden_davor(self):
        spread = pd.Series({1: 30.0, 2: 20.0, 3: 40.0, 4: 10.0, 5: 12.0})
        got = sc_compaction(self._phase(4, 5), spread)
        assert got["baseline_s"].iloc[0] == 30.0     # median(30, 20, 40)

    def test_nimmt_das_minimum_waehrend_der_phase_nicht_den_schnitt(self):
        """bewusste Entscheidung: der Mittelwert waere vom ausloesenden
        Zwischenfall verzerrt, die Phase beginnt ja mit gestrecktem Feld."""
        spread = pd.Series({1: 30.0, 2: 30.0, 3: 30.0, 4: 28.0, 5: 6.0})
        got = sc_compaction(self._phase(4, 5), spread)
        assert got["minimum_s"].iloc[0] == 6.0
        assert got["kompaktierung_pct"].iloc[0] == pytest.approx(80.0)

    def test_phase_am_rennstart_wird_uebersprungen(self):
        """ein Safety Car in Runde 1 hat keine drei gruenen Runden davor."""
        spread = pd.Series({1: 30.0, 2: 10.0, 3: 12.0})
        assert sc_compaction(self._phase(1, 2), spread).empty

    def test_leeres_ergebnis_behaelt_spalten(self):
        spread = pd.Series({1: 30.0, 2: 10.0})
        got = sc_compaction(self._phase(1, 2), spread)
        assert list(got.columns) == ["start", "ende", "baseline_s",
                                     "minimum_s", "kompaktierung_pct"]

    def test_mehrere_phasen_ergeben_mehrere_zeilen(self):
        spread = pd.Series({i: 30.0 for i in range(1, 21)})
        spread[10] = 5.0
        spread[18] = 6.0
        phasen = pd.DataFrame([{"lap_start": 10, "lap_end": 11},
                               {"lap_start": 18, "lap_end": 19}])
        assert len(sc_compaction(phasen, spread)) == 2


class TestDirtyAirEffect:
    """Rundenzeit gegen Nahanteil, nach Herausrechnen der Degradation (P32)."""

    @staticmethod
    def _df(n=40, *, steigung_nah=0.0, steigung_reifen=0.0):
        import numpy as np
        anteil = np.linspace(0, 1, n)
        reifen = np.arange(n, dtype=float)
        return pd.DataFrame({
            "gap_median_m": np.full(n, 50.0),
            "tyre_life": reifen,
            "anteil_nah": anteil,
            "sec_fuel": 90.0 + steigung_nah * anteil + steigung_reifen * reifen,
        })

    def test_zu_wenige_runden_geben_nan(self):
        slope, inter, r2, _ = dirty_air_effect(self._df(n=4))
        assert np.isnan(slope) and np.isnan(inter) and np.isnan(r2)

    def test_konstantes_reifenalter_gibt_nan(self):
        """ohne Variation im Reifenalter laesst sich die Degradation nicht
        herausrechnen."""
        df = self._df()
        df["tyre_life"] = 5.0
        slope, *_ = dirty_air_effect(df)
        assert np.isnan(slope)

    def test_weite_abstaende_fliegen_raus(self):
        df = self._df()
        df.loc[df.index[:20], "gap_median_m"] = 900.0
        _, _, _, d = dirty_air_effect(df)
        assert len(d) == 20

    def test_reine_degradation_ergibt_keinen_dirty_air_effekt(self):
        """der eigentliche Zweck der Korrektur: haengt die Rundenzeit nur am
        Reifenalter, darf am Ende kein Effekt des Nahanteils uebrig bleiben.
        genau diese Verwechslung war der Befund in P32."""
        slope, _, _, _ = dirty_air_effect(self._df(steigung_reifen=0.05))
        assert abs(slope) < 1e-6

    def test_echter_effekt_bleibt_erhalten(self):
        """Nahanteil sauber unabhaengig vom Reifenalter: jedes Reifenalter
        kommt genau einmal mit freier und einmal mit naher Fahrt vor. nur so
        sind die beiden Groessen wirklich trennbar, und dann muss der Effekt
        die Degradations-Korrektur unveraendert ueberleben."""
        alter = np.arange(1.0, 11.0)
        df = pd.DataFrame({
            "gap_median_m": np.full(20, 50.0),
            "tyre_life": np.concatenate([alter, alter]),
            "anteil_nah": np.concatenate([np.zeros(10), np.ones(10)]),
        })
        df["sec_fuel"] = 90.0 + 2.0 * df["anteil_nah"]
        assert np.corrcoef(df["tyre_life"], df["anteil_nah"])[0, 1] == \
            pytest.approx(0.0, abs=1e-12)
        slope, _, _, _ = dirty_air_effect(df)
        assert slope == pytest.approx(2.0, abs=1e-6)

    def test_kollinearer_nahanteil_wird_von_der_korrektur_geschluckt(self):
        """methodische Grenze, kein Fehler: steigt der Nahanteil im
        Gleichschritt mit dem Reifenalter, kann keine Rechnung der Welt die
        beiden auseinanderhalten. die Degradations-Korrektur nimmt den
        Effekt dann fuer sich in Anspruch und uebrig bleibt null."""
        slope, _, _, _ = dirty_air_effect(self._df(steigung_nah=2.0))
        assert abs(slope) < 1e-6

    def test_korrigierte_spalte_wird_ergaenzt(self):
        _, _, _, d = dirty_air_effect(self._df(steigung_reifen=0.05))
        assert "sec_corr" in d.columns


class TestTemperatureEffect:
    """Streckentemperatur-Effekt, kontrolliert um Fahrer und Reifenalter
    (siehe P17: gepoolt geht der Effekt fast immer in der Streuung unter)."""

    @staticmethod
    def _merged(n=40, *, temp=None, tyre=None, coef_temp=0.0, coef_tyre=0.0,
                regen=False):
        temp = np.linspace(30.0, 40.0, n) if temp is None else temp
        tyre = np.tile(np.arange(1.0, 11.0), n // 10) if tyre is None else tyre
        return pd.DataFrame({
            "Rainfall": np.full(n, regen),
            "TrackTemp": temp,
            "TyreLife": tyre,
            "Driver": np.tile(["VER", "NOR"], n // 2),
            "corr": 90.0 + coef_temp * temp + coef_tyre * tyre,
        })

    def test_zu_wenige_trockene_runden(self):
        assert self._temp_n(self._merged(n=10)) == 0

    @staticmethod
    def _temp_n(df):
        from f1lab.session import temperature_effect
        return temperature_effect(df)["n"]

    def test_regen_wird_ausgeschlossen(self):
        assert self._temp_n(self._merged(regen=True)) == 0

    def test_konstante_temperatur_stuerzt_nicht_ab(self):
        """ohne Variation ist die Temperaturspalte identisch zum
        Achsenabschnitt - die Regressionsmatrix wird singulaer. vorher
        LinAlgError, jetzt ein ehrliches leeres Ergebnis."""
        got = temperature_effect(self._merged(temp=np.full(40, 31.0)))
        assert got == {"n": 0}

    def test_konstantes_reifenalter_stuerzt_nicht_ab(self):
        got = temperature_effect(self._merged(tyre=np.full(40, 5.0)))
        assert got == {"n": 0}

    def test_findet_den_gesetzten_temperatureffekt(self):
        got = temperature_effect(self._merged(coef_temp=0.2))
        assert got["n"] > 0
        assert got["coef_temp"] == pytest.approx(0.2, abs=1e-6)

    def test_reifenalter_wird_herausgerechnet(self):
        """haengt die Rundenzeit nur am Reifenalter, darf kein
        Temperatureffekt uebrig bleiben - der Befund aus P17 in klein."""
        got = temperature_effect(self._merged(coef_tyre=0.05))
        assert abs(got["coef_temp"]) < 1e-6

    def test_partial_spalte_fuer_den_plot(self):
        got = temperature_effect(self._merged(coef_temp=0.2))
        assert "partial" in got["dry"].columns


class TestDuelle:
    """teaminterne Duelle: nur wer genau zwei Fahrer im Team hat."""

    @staticmethod
    def _tab(zeilen):
        return pd.DataFrame(zeilen, columns=["Team", "Driver", "LapTime"])

    def test_schnellerer_fahrer_ist_a(self):
        got = _duelle(self._tab([("RBR", "VER", 90.0), ("RBR", "PER", 91.0)]),
                      "Team", "Driver", "LapTime")
        assert got[0]["a"] == "VER" and got[0]["b"] == "PER"
        assert got[0]["score_a"] == 1.0

    def test_delta_in_prozent(self):
        got = _duelle(self._tab([("RBR", "VER", 100.0), ("RBR", "PER", 101.0)]),
                      "Team", "Driver", "LapTime")
        assert got[0]["delta_pct"] == pytest.approx(1.0)

    def test_team_mit_nur_einem_fahrer_faellt_raus(self):
        """kommt real vor: wenn alle Runden des Teamkollegen gestrichen oder
        unplausibel sind, bleibt nur ein Fahrer - allein gibt es kein Duell."""
        got = _duelle(self._tab([("RBR", "VER", 90.0),
                                 ("MCL", "NOR", 90.5), ("MCL", "PIA", 90.7)]),
                      "Team", "Driver", "LapTime")
        assert [d["team"] for d in got] == ["MCL"]

    def test_team_mit_drei_fahrern_faellt_raus(self):
        """Fahrerwechsel innerhalb einer Saison - ein Dreieck ist kein Duell."""
        got = _duelle(self._tab([("RB", "RIC", 90.0), ("RB", "TSU", 90.2),
                                 ("RB", "LAW", 90.4)]),
                      "Team", "Driver", "LapTime")
        assert got == []
