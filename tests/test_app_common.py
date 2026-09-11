"""Tests fuer app/common.py - die Schicht, die alle Dashboard-Seiten teilen.

Die Datei hat 118 Anweisungen und bis hierhin keinen einzigen direkten
Test. Was an Abdeckung anfiel, fiel nebenbei beim Seiten-Rauchtest ab
(tests/test_app_seiten.py) - und der prueft nur, dass eine Seite ohne
Exception durchlaeuft, nicht, ob die Auswahl das Richtige anbietet.

Gearbeitet wird gegen ein schlankes Streamlit-Double statt gegen
``AppTest``: geprueft werden soll die Auswahl-Logik, nicht das Rendern.
Das Double haelt fest, mit welchen Optionen und welchem Index das
Auswahlfeld aufgerufen wurde - genau daran haengen beide Zusicherungen,
die ``common`` in seinen Docstrings gibt.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

WURZEL = Path(__file__).resolve().parents[1]
if str(WURZEL / "app") not in sys.path:
    sys.path.insert(0, str(WURZEL / "app"))

import common  # noqa: E402


class _FakeSidebar:
    """nimmt die selectbox-aufrufe entgegen, statt sie zu zeichnen."""

    def __init__(self, aufrufe: list):
        self.aufrufe = aufrufe

    def selectbox(self, label, optionen, index=0, format_func=str, help=None):
        optionen = list(optionen)
        # format_func wird hier bewusst wirklich aufgerufen: eine
        # Beschriftung, die auf einer unbekannten Kennung scheitert,
        # faellt sonst erst im Browser auf
        beschriftungen = [format_func(o) for o in optionen]
        self.aufrufe.append({"label": label, "optionen": optionen,
                             "index": index, "beschriftungen": beschriftungen})
        return optionen[index]

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def __getattr__(self, _name):
        return lambda *a, **k: None


class _FakeSt:
    def __init__(self):
        self.session_state: dict = {}
        self.aufrufe: list = []
        self.sidebar = _FakeSidebar(self.aufrufe)
        self.meldungen: list = []

    def info(self, text, *a, **k):
        self.meldungen.append(("info", text))

    def success(self, text, *a, **k):
        self.meldungen.append(("success", text))

    def __getattr__(self, _name):
        return lambda *a, **k: None


@pytest.fixture
def st_fake(monkeypatch):
    fake = _FakeSt()
    monkeypatch.setattr(common, "st", fake)
    return fake


class TestWahlBehaeltDenWert:
    """``_wahl`` verspricht im Docstring einen Rueckfall.

    Woertlich: "ein gespeicherter wert, der in der neuen liste nicht mehr
    vorkommt, wuerde sonst einen fehler ausloesen statt zurueckzufallen".
    Genau diese Sorte Zusage - im Docstring behauptet, nirgends geprueft -
    hat in diesem Repo schon mehrfach zugeschlagen (``wet_dry_classifier``s
    "das prueft der aufrufer", der fehlende ``dauer``-Schluessel in
    ``mini_sectors``).
    """

    def test_gespeicherter_wert_wird_wieder_ausgewaehlt(self, st_fake):
        st_fake.session_state["k"] = "Monaco"
        wert = common._wahl("Rennen", ["Imola", "Monaco", "Spa"], "k")
        assert wert == "Monaco"
        assert st_fake.aufrufe[-1]["index"] == 1

    def test_verschwundener_wert_faellt_auf_die_erste_option_zurueck(
            self, st_fake):
        """real erreichbar: Saison wechseln, danach steht das vorher
        gewaehlte Rennen nicht mehr in der Liste."""
        st_fake.session_state["k"] = "Spain"
        wert = common._wahl("Rennen", ["Imola", "Monaco"], "k")
        assert wert == "Imola"
        assert st_fake.aufrufe[-1]["index"] == 0

    def test_ohne_vorgeschichte_steht_die_erste_option(self, st_fake):
        wert = common._wahl("Rennen", ["Imola", "Monaco"], "k")
        assert wert == "Imola"
        assert st_fake.aufrufe[-1]["index"] == 0

    def test_die_wahl_wird_fuer_den_seitenwechsel_gemerkt(self, st_fake):
        common._wahl("Rennen", ["Imola", "Monaco"], "k")
        assert st_fake.session_state["k"] == "Imola"


def _inventar(zeilen: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(zeilen, columns=["season", "event", "event_date",
                                         "ident", "timing", "telemetry"])


@pytest.fixture
def inv(monkeypatch):
    """bestandsaufnahme setzen, ohne einen echten Cache zu brauchen."""
    def setzen(zeilen):
        monkeypatch.setattr(common, "inventar", lambda _p: _inventar(zeilen))
    return setzen


class TestSidebarSessionBietetNurAuswertbares:
    """``sidebar_session`` verspricht: "angeboten wird nur, was auch
    auswertbar ist" - weil FastF1 offline nicht beim Laden scheitert,
    sondern erst beim Zugriff auf die Runden, wenn die Oberflaeche schon
    steht."""

    @staticmethod
    def _zeile(event, ident, *, timing=True, telemetry=False, tag="2024-03-02"):
        return {"season": 2024, "event": event, "event_date": pd.Timestamp(tag),
                "ident": ident, "timing": timing, "telemetry": telemetry}

    def test_sessions_ohne_timing_werden_nicht_angeboten(self, st_fake, inv):
        inv([self._zeile("Bahrain", "R"),
             self._zeile("Bahrain", "FP1", timing=False)])
        common.sidebar_session(Path("/egal"))
        session_feld = [a for a in st_fake.aufrufe if a["label"] == "Session"][0]
        assert session_feld["optionen"] == ["R"]

    def test_ohne_telemetrie_kommt_kein_leerer_dialog(self, st_fake, inv):
        """die Seite muss ``None`` liefern und es sagen, nicht eine
        Auswahl ueber einer leeren Liste anbieten (``.iloc[0]`` waere
        sonst ein IndexError)."""
        inv([self._zeile("Bahrain", "R", telemetry=False)])
        assert common.sidebar_session(Path("/egal"),
                                      nur_mit_telemetrie=True) is None
        assert any(art == "info" for art, _ in st_fake.meldungen)

    def test_reihenfolge_ist_die_des_wochenendes_nicht_alphabetisch(
            self, st_fake, inv):
        inv([self._zeile("Bahrain", i)
             for i in ("FP1", "Q", "R", "FP2")])
        common.sidebar_session(Path("/egal"))
        session_feld = [a for a in st_fake.aufrufe if a["label"] == "Session"][0]
        assert session_feld["optionen"] == ["R", "Q", "FP2", "FP1"]

    def test_unbekannte_kennung_faellt_heraus(self, st_fake, inv):
        """eine Kennung, die IDENT_ORDER nicht kennt, wird still
        verworfen - dokumentiert, damit der naechste Session-Typ
        (wie seinerzeit "SQ") nicht unbemerkt fehlt."""
        inv([self._zeile("Bahrain", "R"), self._zeile("Bahrain", "XX")])
        common.sidebar_session(Path("/egal"))
        session_feld = [a for a in st_fake.aufrufe if a["label"] == "Session"][0]
        assert session_feld["optionen"] == ["R"]

    def test_neueste_saison_steht_oben(self, st_fake, inv):
        inv([self._zeile("Bahrain", "R") | {"season": s}
             for s in (2021, 2026, 2024)])
        common.sidebar_session(Path("/egal"))
        saison_feld = [a for a in st_fake.aufrufe if a["label"] == "Saison"][0]
        assert saison_feld["optionen"] == [2026, 2024, 2021]

    def test_rennen_stehen_in_kalenderreihenfolge(self, st_fake, inv):
        inv([self._zeile("Monaco", "R", tag="2024-05-26"),
             self._zeile("Bahrain", "R", tag="2024-03-02"),
             self._zeile("Monza", "R", tag="2024-09-01")])
        common.sidebar_session(Path("/egal"))
        renn_feld = [a for a in st_fake.aufrufe if a["label"] == "Rennen"][0]
        assert renn_feld["optionen"] == ["Bahrain", "Monaco", "Monza"]

    def test_die_auswahl_traegt_das_telemetrie_flag_der_zeile(
            self, st_fake, inv):
        inv([self._zeile("Bahrain", "R", telemetry=True)])
        auswahl = common.sidebar_session(Path("/egal"))
        assert auswahl is not None
        assert (auswahl.season, auswahl.event, auswahl.ident) == (
            2024, "Bahrain", "R")
        assert auswahl.telemetrie is True
