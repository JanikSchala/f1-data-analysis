"""
P34 - Der komplette Wochenend-Analyzer als CLI-Tool
===================================================

Ein installierbares Python-Paket, das alle vorherigen Analysen buendelt: f1analyze weekend 2024 Monza --report.

Kategorie:   Data Engineering
Niveau:      Profi
Aufwand:     8-10 h
Rollen:      ENG, DS, STRAT
Zusaetzliche Pakete: typer

WARUM DAS ZAEHLT
Das Abschlussprojekt. Zeigt Software-Engineering statt Skript-Sammlung: Paketstruktur, CLI, Tests, Typannotationen, README. Genau so arbeiten Teams.

VORGEHEN
  1. Paketstruktur anlegen: f1analyze/{data,analysis,viz,cli}.py
  2. Analysen aus den Vorprojekten als Funktionen mit Typannotationen
  3. CLI mit Subcommands: weekend, pace, strategy, telemetry, report
  4. pytest mit gecachten Fixture-Sessions (kein Netz im Test)
  5. pyproject.toml, README mit Beispielbildern, auf GitHub veroeffentlichen

GENUTZTE FASTF1-BAUSTEINE
  - fastf1 gesamt
  - typer/argparse
  - pytest
  - pyproject.toml

AUSBAUSTUFE
Publiziere das Paket auf PyPI und richte eine GitHub Action ein, die bei jedem Push Tests, Linting (ruff) und Typecheck (mypy) laufen laesst.
"""

# f1analyze/cli.py   -   Aufruf: f1analyze weekend 2024 Monza --report out.pdf
from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
import pandas as pd
import fastf1

from .analysis import race_pace, degradation, stint_summary
from .viz import plot_pace, plot_strategy, build_pdf

app = typer.Typer(help="Analyse-Werkzeug fuer Formel-1-Renndaten")
fastf1.Cache.enable_cache(str(Path.home() / "f1_cache"))


@app.command()
def pace(year: int, gp: str, session: str = "R", top: int = 20) -> None:
    """Bereinigte Race Pace als Rangliste."""
    ses = fastf1.get_session(year, gp, session)
    ses.load(telemetry=False)
    df = race_pace(ses)
    typer.echo(df.head(top).to_string(index=False))


@app.command()
def strategy(year: int, gp: str) -> None:
    """Stints und Compounds je Fahrer."""
    ses = fastf1.get_session(year, gp, "R")
    ses.load(telemetry=False)
    typer.echo(stint_summary(ses).to_string(index=False))


@app.command()
def deg(year: int, gp: str) -> None:
    """Degradation je Stint und Compound."""
    ses = fastf1.get_session(year, gp, "R")
    ses.load(telemetry=False)
    typer.echo(degradation(ses).to_string(index=False))


@app.command()
def weekend(year: int, gp: str,
            report: Optional[Path] = typer.Option(None,
                                                  help="PDF-Ausgabepfad")) -> None:
    """Komplette Wochenendanalyse, optional als PDF."""
    race = fastf1.get_session(year, gp, "R")
    race.load(telemetry=False)
    quali = fastf1.get_session(year, gp, "Q")
    quali.load(telemetry=False)

    typer.secho(f"{race.event['EventName']} {year}", fg="red", bold=True)
    typer.echo("\n--- Qualifying ---")
    typer.echo(quali.results[["Position", "Abbreviation", "TeamName",
                              "Q3"]].head(10).to_string(index=False))
    typer.echo("\n--- Race Pace ---")
    typer.echo(race_pace(race).head(10).to_string(index=False))
    typer.echo("\n--- Degradation ---")
    typer.echo(degradation(race).groupby("Compound")["Deg"]
               .mean().round(4).to_string())

    if report:
        build_pdf(race, quali, report)
        typer.secho(f"Report geschrieben: {report}", fg="green")


if __name__ == "__main__":
    app()
