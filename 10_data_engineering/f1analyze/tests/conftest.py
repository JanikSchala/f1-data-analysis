"""Fixtures gegen einen mitgelieferten FastF1-Cache-Ausschnitt, ohne
Netzzugriff.

offline=True laesst eine Session ohne vorhandenen Cache-Eintrag scheitern.
das ist schnell und eindeutig statt einer minutenlangen haengenden
Netzanfrage. siehe f1lab.session.enable_cache().

FIXTURE_CACHE zeigt bewusst auf tests/fixtures/f1_cache_bahrain2024/ statt
auf den Standardpfad ``~/f1_cache``. ein frischer Runner hat keinen warmen
Cache. Tests gegen den eigenen ``~/f1_cache`` haengen sonst vom Zufall ab
und sind keine echten Fixtures. der committete Ausschnitt ist
deterministisch fuer jeden, der das Repo klont.
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
    """wie race_session, aber mit Telemetrie. fuer lap_simulation/
    overtake_summary, die ohne nicht auskommen."""
    f1lab.enable_cache(path=FIXTURE_CACHE, offline=True)
    return f1lab.load(FIXTURE_SEASON, FIXTURE_EVENT, "R", telemetry=True)


@pytest.fixture(scope="session")
def quali_session_tel():
    f1lab.enable_cache(path=FIXTURE_CACHE, offline=True)
    return f1lab.load(FIXTURE_SEASON, FIXTURE_EVENT, "Q", telemetry=True)
