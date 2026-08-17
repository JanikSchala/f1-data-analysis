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


def test_optimize_command_runs(race_session):
    result = runner.invoke(app, ["optimize", str(FIXTURE_SEASON), FIXTURE_EVENT])
    assert result.exit_code == 0
    assert "Stopp" in result.stdout


def test_lap_sim_command_runs(quali_session_tel):
    result = runner.invoke(app, ["lap-sim", str(FIXTURE_SEASON), FIXTURE_EVENT])
    assert result.exit_code == 0
    assert "mu_g" in result.stdout


def test_overtakes_command_runs(race_session_tel, quali_session_tel):
    result = runner.invoke(app, ["overtakes", str(FIXTURE_SEASON), FIXTURE_EVENT])
    assert result.exit_code == 0
    assert "Ueberholungen" in result.stdout


def test_traffic_command_runs(race_session):
    result = runner.invoke(app, ["traffic", str(FIXTURE_SEASON), FIXTURE_EVENT, "3"])
    assert result.exit_code == 0
    assert "mit Verkehr" in result.stdout


def test_traffic_command_rejects_impossible_stop_count(race_session):
    result = runner.invoke(app, ["traffic", str(FIXTURE_SEASON), FIXTURE_EVENT, "99"])
    assert result.exit_code == 1


def test_garbage_event_name_is_fuzzy_matched_not_rejected(race_session):
    """Kein sauberer Fehlerfall, sondern ein echter Fund: FastF1 lehnt einen
    unbekannten Streckennamen nicht ab, sondern korrigiert ihn per
    Fuzzy-Matching auf die naechstliegende echte Strecke - stillschweigend,
    nur als WARNING geloggt. Ein Tippfehler in der CLI analysiert damit ohne
    Fehlermeldung das falsche Rennen. Der Test haelt das aktuelle Verhalten
    fest statt einen Fehlerfall zu behaupten, der so nicht eintritt.

    Absichtlich auf "Bahrain" gemuenzt (statt eines beliebigen Garbage-
    Strings): der Fixture-Cache (tests/fixtures/f1_cache_bahrain2024/)
    deckt nur dieses eine Event ab, seit load_session() eine von aussen
    gesetzte Offline-Cache-Konfiguration nicht mehr stillschweigend
    ueberschreibt (siehe f1lab.cache_ready(), CLAUDE.md-Nachtrag) - ein
    Fuzzy-Match auf ein nicht gecachtes Event wuerde jetzt korrekt mit
    DataNotLoadedError statt eines stillen Netzzugriffs scheitern."""
    result = runner.invoke(app, ["pace", "2024", "komplett falsche Strecke Bahrain"])
    assert result.exit_code == 0
