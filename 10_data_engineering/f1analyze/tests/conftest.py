"""Fixtures gegen einen mitgelieferten FastF1-Cache-Ausschnitt, ohne
Netzzugriff (VORGEHEN 4).

offline=True laesst jede Session scheitern, die nicht bereits im Cache
liegt - schnell und eindeutig statt einer minutenlangen haengenden
Netzanfrage, wenn das Fixture-Rennen einmal fehlt (siehe
f1lab.session.enable_cache()).

FIXTURE_CACHE zeigt bewusst auf tests/fixtures/f1_cache_bahrain2024/ (im
Repo committet, ~147 MB: car_data/position_data lassen sich nicht weiter
verkleinern, siehe CLAUDE.md) statt auf den Standardpfad ``~/f1_cache`` -
zwei Gruende. Erstens: das war lange Zeit der eigentliche Bug hinter einem
CI-Fehlschlag, der nie auffiel, weil die CI-Workflow-Datei am falschen Ort
lag (siehe CLAUDE.md) - ein frischer GitHub-Actions-Runner hat keinen
warmen ``~/f1_cache`` und nicht einmal den Saisonkalender, `offline=True`
scheiterte dort komplett. Zweitens, unabhaengig von CI: Tests, die vom
Zufall abhaengen, was gerade zufaellig im eigenen ``~/f1_cache`` liegt,
sind keine echten Fixtures - der committete Ausschnitt ist deterministisch
fuer jeden, der das Repo klont.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import pytest

import f1lab

FIXTURE_SEASON, FIXTURE_EVENT = 2024, "Bahrain"
FIXTURE_CACHE = Path(__file__).resolve().parent / "fixtures" / "f1_cache_bahrain2024"


@pytest.fixture(scope="session")
def race_session():
    f1lab.enable_cache(path=FIXTURE_CACHE, offline=True)
    return f1lab.load(FIXTURE_SEASON, FIXTURE_EVENT, "R", telemetry=False)


@pytest.fixture(scope="session")
def quali_session():
    f1lab.enable_cache(path=FIXTURE_CACHE, offline=True)
    return f1lab.load(FIXTURE_SEASON, FIXTURE_EVENT, "Q", telemetry=False)


@pytest.fixture(scope="session")
def race_session_tel():
    """Wie race_session, aber mit Telemetrie - fuer P37/P39-Bausteine
    (lap_simulation, overtake_summary), die ohne nicht auskommen."""
    f1lab.enable_cache(path=FIXTURE_CACHE, offline=True)
    return f1lab.load(FIXTURE_SEASON, FIXTURE_EVENT, "R", telemetry=True)


@pytest.fixture(scope="session")
def quali_session_tel():
    f1lab.enable_cache(path=FIXTURE_CACHE, offline=True)
    return f1lab.load(FIXTURE_SEASON, FIXTURE_EVENT, "Q", telemetry=True)
