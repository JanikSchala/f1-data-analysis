"""CLI mit Subcommands: weekend, pace, strategy, telemetry, report,
optimize, lap-sim, overtakes, traffic.

Aufruf, sobald installiert (siehe README): f1analyze weekend 2024 Monza

Ueberholschwierigkeit je Strecke und Startplatz-Paritaet sind bewusst
keine Subcommands. beide brauchen einen Season-Scan ueber Dutzende bis
hunderte Sessions statt Sekunden. ein Einzelaufruf passt nicht zu dieser
Kostenstruktur, sie leben deshalb nur als Dashboard-Seiten mit
``persist="disk"``-Cache.
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
    """bereinigte Race Pace als Rangliste."""
    ses = load_session(year, gp, session)
    df = analysis.race_pace(ses)
    typer.echo(df.head(top).to_string(index=False))


@app.command()
def strategy(year: int, gp: str) -> None:
    """stints und Compounds je Fahrer."""
    ses = load_session(year, gp, "R")
    typer.echo(analysis.stint_summary(ses).to_string(index=False))


@app.command()
def telemetry(year: int, gp: str, driver_a: str, driver_b: str,
             session: str = "Q") -> None:
    """schnellste Runden zweier Fahrer vergleichen: Zeit, Speed-Trap,
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
    """komplette Wochenendanalyse als Text: Quali, Race Pace, Degradation."""
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
def optimize(year: int, gp: str) -> None:
    """exakt bester Boxenstopp-Plan, aus echter Degradation/Pitloss."""
    ses = load_session(year, gp, "R")
    try:
        cfg, plan = analysis.optimal_plan(ses)
    except ValueError as exc:
        typer.secho(f"Kein Modell bildbar: {exc}", fg="red")
        raise typer.Exit(1) from exc
    typer.echo(f"{cfg.n_laps} Runden, Pitloss {cfg.pit_loss:.1f}s, "
              f"{len(cfg.tyres)} belastbare Mischungen\n")
    typer.echo(plan.describe())


@app.command()
def lap_sim(year: int, gp: str, session: str = "Q") -> None:
    """punktmassen-Rundenzeitsimulation der schnellsten Runde."""
    ses = load_session(year, gp, session, telemetry=True)
    erg = analysis.lap_simulation(ses)
    typer.echo(f"mu_g={erg['mu_g']:.2f} m/s^2 ({erg['mu_g'] / 9.81:.2f}g)  "
              f"a_accel={erg['a_accel']:.2f} m/s^2  "
              f"a_brake={erg['a_brake']:.2f} m/s^2  "
              f"v_top={erg['v_top'] * 3.6:.1f} km/h")
    typer.echo(f"real {erg['t_real_s']:.3f}s  simuliert {erg['t_sim_s']:.3f}s  "
              f"({erg['diff_pct']:+.2f}%)")


@app.command()
def overtakes(year: int, gp: str) -> None:
    """ueberholungen und ihr Anteil in der DRS-Zone."""
    race = load_session(year, gp, "R", telemetry=True)
    quali = load_session(year, gp, "Q", telemetry=True)
    erg = analysis.overtake_summary(race, quali)
    typer.echo(f"{erg['ueberholungen']} Ueberholungen, "
              f"{erg['lokalisiert']} lokalisiert")
    if erg["lokalisiert"]:
        typer.echo(f"davon {100 * erg['drs_anteil']:.1f}% in einer DRS-Zone")


@app.command()
def traffic(year: int, gp: str, alt_stops: int,
           delta: float = 0.15, gap: float = 3.0,
           p_overtake: float = 0.15) -> None:
    """Verkehrs-Simulation: DAG-Optimum gegen eine Alternative mit
    Stoppzahl ALT_STOPS, beide gegen einen Rivalen simuliert. Rivalen-Tempo,
    Startabstand und Ueberholwahrscheinlichkeit sind Szenario-Annahmen,
    keine Messwerte."""
    ses = load_session(year, gp, "R")
    try:
        erg = analysis.traffic_scenario(ses, alt_stops, delta, gap, p_overtake)
    except ValueError as exc:
        typer.secho(str(exc), fg="red")
        raise typer.Exit(1) from exc
    typer.echo(f"{erg['hero_stops']}-Stopp (Optimum):  "
              f"{erg['hero_frei']:8.2f}s frei  {erg['hero_verkehr']:8.2f}s "
              "mit Verkehr")
    typer.echo(f"{erg['alt_stops']}-Stopp (Alternative): "
              f"{erg['alt_frei']:8.2f}s frei  {erg['alt_verkehr']:8.2f}s "
              "mit Verkehr")
    diff = erg["alt_verkehr"] - erg["hero_verkehr"]
    typer.echo(f"Abstand mit Verkehr: {diff:+.2f}s -> "
              f"{'Optimum bleibt besser' if diff > 0 else 'Alternative waere besser!'}")


@app.command()
def report(year: int, gp: str,
          out: Path = typer.Option(  # noqa: B008 (typers eigenes Muster)
              Path("report.pdf"), help="PDF-Ausgabepfad")
          ) -> None:
    """pdf-Report mit Race Pace und Reifenstrategie erzeugen."""
    race = load_session(year, gp, "R")
    quali = load_session(year, gp, "Q")
    pace_df = analysis.race_pace(race)
    stints_df = analysis.stint_summary(race)
    viz.build_pdf(race, quali, pace_df, stints_df, out)
    typer.secho(f"Report geschrieben: {out}", fg="green")


if __name__ == "__main__":
    app()
