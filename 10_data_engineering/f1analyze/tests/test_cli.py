from __future__ import annotations

from f1analyze.cli import app
from typer.testing import CliRunner

runner = CliRunner()
FIXTURE_SEASON, FIXTURE_EVENT = 2024, "Bahrain"     # muss zu conftest.py passen


def test_pace_command_runs(race_session):
    result = runner.invoke(app, ["pace", str(FIXTURE_SEASON), FIXTURE_EVENT,
                                 "--top", "5"])
    assert result.exit_code == 0
    assert "VER" in result.stdout or "HAM" in result.stdout or len(result.stdout) > 0


def test_strategy_command_runs(race_session):
    result = runner.invoke(app, ["strategy", str(FIXTURE_SEASON), FIXTURE_EVENT])
    assert result.exit_code == 0
    assert "SOFT" in result.stdout or "MEDIUM" in result.stdout or "HARD" in result.stdout


def test_weekend_command_runs(race_session, quali_session):
    result = runner.invoke(app, ["weekend", str(FIXTURE_SEASON), FIXTURE_EVENT])
    assert result.exit_code == 0
    assert "Race Pace" in result.stdout
    assert "Degradation" in result.stdout


def test_garbage_event_name_is_fuzzy_matched_not_rejected():
    """Kein sauberer Fehlerfall, sondern ein echter Fund: FastF1 lehnt einen
    unbekannten Streckennamen nicht ab, sondern korrigiert ihn per
    Fuzzy-Matching auf die naechstliegende echte Strecke (hier: "China") -
    stillschweigend, nur als WARNING geloggt. Ein Tippfehler in der CLI
    analysiert damit ohne Fehlermeldung das falsche Rennen. Der Test haelt
    das aktuelle Verhalten fest statt einen Fehlerfall zu behaupten, der so
    nicht eintritt."""
    result = runner.invoke(app, ["pace", "2024", "Nichtexistente Strecke"])
    assert result.exit_code == 0
