"""jede Dashboard-Seite muss mit leerem Cache eine Meldung zeigen, keinen Traceback.

Das ist der Zustand jedes Erstnutzers: Repository geklont, Dashboard
gestartet, noch kein einziges Rennen heruntergeladen. 23 der 26 Seiten
fangen ihn ueber common.kein_cache_hinweis() ab - dass es wirklich alle
tun und auch so bleibt, stand bis hier nirgends geschrieben. Die App
hatte ueberhaupt keine Tests.

Laeuft ohne Netz und ohne Cache und braucht damit nichts, was ein
frischer CI-Runner nicht haette. F1_CACHE zeigt auf ein leeres tmp-
Verzeichnis, sonst wuerde der Test auf einem Entwicklungsrechner den
warmen ~/f1_cache finden und etwas voellig anderes pruefen.

Bewusst nur ein Rauchtest: geprueft wird, dass keine Ausnahme bis zur
Oberflaeche durchschlaegt. Was die Seiten inhaltlich rechnen, gehoert
nach f1lab und wird dort getestet ("gerechnet wird ausschliesslich in
f1lab").
"""
from __future__ import annotations

import pathlib

import pytest

WURZEL = pathlib.Path(__file__).resolve().parents[1]
APP = WURZEL / "app"
SEITEN = [APP / "Start.py", *sorted((APP / "pages").glob("*.py"))]


@pytest.fixture(scope="module")
def leerer_cache(tmp_path_factory, monkeypatch_module):
    leer = tmp_path_factory.mktemp("f1_cache_leer")
    monkeypatch_module.setenv("F1_CACHE", str(leer))
    return leer


@pytest.fixture(scope="module")
def monkeypatch_module():
    """monkeypatch gibt es nur je Test; die Seiten teilen sich aber einen
    Streamlit-Zwischenspeicher, deshalb hier modulweit."""
    from _pytest.monkeypatch import MonkeyPatch
    mp = MonkeyPatch()
    yield mp
    mp.undo()


def test_es_gibt_seiten():
    """Schutz gegen einen Test, der gruen ist, weil er nichts gefunden hat."""
    assert len(SEITEN) >= 20


def test_der_cache_ist_wirklich_leer(leerer_cache, monkeypatch):
    """ohne diesen Test koennte der ganze Rauchtest gegen einen warmen
    ~/f1_cache laufen und etwas voellig anderes pruefen als er behauptet.

    2_Pace.py ruft common.kein_cache_hinweis() auf und muss deshalb genau
    dessen Meldung zeigen, nicht ein Ergebnis.
    """
    at = pytest.importorskip(
        "streamlit.testing.v1", reason="streamlit-Extra nicht installiert")
    monkeypatch.syspath_prepend(str(APP))
    monkeypatch.syspath_prepend(str(WURZEL))

    app = at.AppTest.from_file(str(APP / "pages" / "2_Pace.py"),
                               default_timeout=120)
    app.run()
    meldungen = [e.value for e in app.error]
    assert any("keine Renndaten" in m for m in meldungen), meldungen


@pytest.mark.parametrize("seite", SEITEN, ids=lambda p: p.name)
def test_seite_ueberlebt_leeren_cache(seite, leerer_cache, monkeypatch):
    at = pytest.importorskip(
        "streamlit.testing.v1", reason="streamlit-Extra nicht installiert")
    monkeypatch.syspath_prepend(str(APP))       # die Seiten importieren "common"
    monkeypatch.syspath_prepend(str(WURZEL))

    app = at.AppTest.from_file(str(seite), default_timeout=120)
    app.run()
    assert not app.exception, (
        f"{seite.name} wirft bei leerem Cache: "
        f"{app.exception[0].value.splitlines()[0]}")
