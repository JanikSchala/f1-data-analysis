"""CLI mit Subcommands (VORGEHEN 3): weekend, pace, strategy, telemetry,
report.

Aufruf, sobald installiert (siehe README): f1analyze weekend 2024 Monza
"""
from __future__ import annotations

from pathlib import Path

import typer

import f1lab

from . import analysis, viz
from .data import load_session

app = typer.Typer(help="Analyse-Werkzeug fuer Formel-1-Renndaten")


@app.command()
def pace(year: int, gp: str, session: str = "R", top: int = 20) -> None:
    """Bereinigte Race Pace als Rangliste."""
    ses = load_session(year, gp, session)
    df = analysis.race_pace(ses)
    typer.echo(df.head(top).to_string(index=False))


@app.command()
def strategy(year: int, gp: str) -> None:
    """Stints und Compounds je Fahrer."""
    ses = load_session(year, gp, "R")
    typer.echo(analysis.stint_summary(ses).to_string(index=False))


@app.command()
def telemetry(year: int, gp: str, driver_a: str, driver_b: str,
             session: str = "Q") -> None:
    """Schnellste Runden zweier Fahrer vergleichen: Zeit, Speed-Trap,
    Bremszonen-Zeitpunkte."""
    ses = load_session(year, gp, session, telemetry=True)
    la = ses.laps.pick_drivers(driver_a.upper()).pick_fastest()
    lb = ses.laps.pick_drivers(driver_b.upper()).pick_fastest()
    typer.echo(f"{driver_a.upper()}: {la['LapTime']}   "
              f"{driver_b.upper()}: {lb['LapTime']}")

    za = f1lab.driver_braking_zones(ses, driver_a.upper())
    zb = f1lab.driver_braking_zones(ses, driver_b.upper())
    vgl = f1lab.compare_braking_zones(za, zb)
    typer.echo(f"\n{len(za)} Bremszonen {driver_a.upper()}, {len(zb)} "
              f"Bremszonen {driver_b.upper()}, {len(vgl)} gepaart:")
    typer.echo(vgl.round(1).to_string(index=False))


@app.command()
def weekend(year: int, gp: str) -> None:
    """Komplette Wochenendanalyse als Text: Quali, Race Pace, Degradation."""
    race = load_session(year, gp, "R")
    quali = load_session(year, gp, "Q")

    typer.secho(f"{race.event['EventName']} {year}", fg="red", bold=True)
    typer.echo("\n--- Qualifying ---")
    typer.echo(quali.results[["Position", "Abbreviation", "TeamName",
                              "Q3"]].head(10).to_string(index=False))
    typer.echo("\n--- Race Pace ---")
    typer.echo(analysis.race_pace(race).head(10).to_string(index=False))
    typer.echo("\n--- Degradation je Mischung ---")
    typer.echo(analysis.degradation_by_compound(race).to_string())


@app.command()
def report(year: int, gp: str,
          out: Path = typer.Option(  # noqa: B008 - typers eigenes Muster
              Path("report.pdf"), help="PDF-Ausgabepfad")
          ) -> None:
    """PDF-Report mit Race Pace und Reifenstrategie erzeugen."""
    race = load_session(year, gp, "R")
    quali = load_session(year, gp, "Q")
    pace_df = analysis.race_pace(race)
    stints_df = analysis.stint_summary(race)
    viz.build_pdf(race, quali, pace_df, stints_df, out)
    typer.secho(f"Report geschrieben: {out}", fg="green")


if __name__ == "__main__":
    app()
