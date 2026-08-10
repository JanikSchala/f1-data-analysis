"""Fixtures gegen den lokalen FastF1-Cache, ohne Netzzugriff (VORGEHEN 4).

offline=True laesst jede Session scheitern, die nicht bereits im Cache
liegt - schnell und eindeutig statt einer minutenlangen haengenden
Netzanfrage, wenn das Fixture-Rennen einmal fehlt (siehe
f1lab.session.enable_cache()).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import pytest

import f1lab

FIXTURE_SEASON, FIXTURE_EVENT = 2024, "Bahrain"


@pytest.fixture(scope="session")
def race_session():
    f1lab.enable_cache(offline=True)
    return f1lab.load(FIXTURE_SEASON, FIXTURE_EVENT, "R", telemetry=False)


@pytest.fixture(scope="session")
def quali_session():
    f1lab.enable_cache(offline=True)
    return f1lab.load(FIXTURE_SEASON, FIXTURE_EVENT, "Q", telemetry=False)
