"""erzeugt einen mehrseitigen pdf-rennbericht mit ergebnis, pace, strategie und positionsverlauf"""
from __future__ import annotations

import argparse
import logging
import sys
import warnings
from datetime import date, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")

import fastf1
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

import f1lab
from f1lab.design import BG_HELL, COMPOUND, FG, GRID, MUTED, SERIEN, matplotlib_stil

# Bewusst dauerhaft, nicht nur zur einmaligen Diagnose. FastF1s eigener
# Handler blieb im GitHub-Actions-Runner ohne erkennbaren Grund stumm. Ein
# erzwungener Handler zeigte erst die eigentliche Ursache (siehe
# _lade_mit_wiederholung()). Ein kuenftiger Fehlschlag soll sofort im
# Actions-Log diagnostizierbar sein.
logging.basicConfig(level=logging.INFO, force=True, stream=sys.stdout,
                    format="%(name)s %(levelname)s %(message)s")

warnings.filterwarnings("ignore")
fastf1.set_log_level("ERROR")

SEITENGROESSE = (11.7, 8.3)   # A4 quer

plt.rcParams.update(matplotlib_stil())


def _fusszeile(fig, ses, seite: int, seiten_gesamt: int) -> None:
    fig.text(0.06, 0.965, f"{ses.event['EventName']} {ses.event.year}",
             fontsize=11, color=MUTED, ha="left")
    fig.text(0.06, 0.02,
             f"Erstellt am {date.today().isoformat()} - f1-data-analysis",
             fontsize=8, color=MUTED, ha="left")
    fig.text(0.94, 0.02, f"Seite {seite}/{seiten_gesamt}", fontsize=8,
             color=MUTED, ha="right")


def page_results(pdf, ses, seite: int, gesamt: int) -> None:
    fig, ax = plt.subplots(figsize=SEITENGROESSE)
    ax.axis("off")
    res = ses.results[["Position", "Abbreviation", "TeamName",
                       "GridPosition", "Status", "Points"]].copy()
    res["Gewinn"] = res["GridPosition"] - res["Position"]
    tbl = ax.table(cellText=res.round(0).astype(str).values,
                   colLabels=res.columns, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1, 1.5)
    for cell in tbl.get_celld().values():
        cell.set_edgecolor(GRID)
        cell.set_facecolor(BG_HELL)
        cell.set_text_props(color=FG)
    ax.set_title(f"{ses.event['EventName']} {ses.event.year} - Ergebnis",
                fontsize=16, pad=20, color=FG, loc="left")
    _fusszeile(fig, ses, seite, gesamt)
    pdf.savefig(fig)
    plt.close(fig)


def page_pace(pdf, ses, seite: int, gesamt: int) -> None:
    """nutzt f1lab.pace_table() (Treibstoffkorrektur, Bootstrap-Konfidenzintervall) statt einer eigenen Median-Rechnung."""
    df = f1lab.pace_table(ses)
    fig, ax = plt.subplots(figsize=SEITENGROESSE)
    if df.empty:
        ax.text(0.5, 0.5, "keine auswertbare Pace", ha="center", color=MUTED,
               transform=ax.transAxes)
        ax.axis("off")
    else:
        farben = [SERIEN[0] if i < 3 else MUTED for i in range(len(df))]
        ax.barh(df["driver"], df["delta_s"], color=farben, height=0.6,
               xerr=[df["delta_s"] - df["ci_lo"], df["ci_hi"] - df["delta_s"]],
               error_kw={"ecolor": MUTED, "elinewidth": 1, "capsize": 2})
        ax.invert_yaxis()
        ax.set_xlabel("Delta zum Schnellsten [s]")
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.grid(axis="x", alpha=0.3, linewidth=0.8, color=GRID)
        ax.set_axisbelow(True)
    ax.set_title("Bereinigte Race Pace", fontsize=16, pad=20, color=FG,
                loc="left")
    _fusszeile(fig, ses, seite, gesamt)
    pdf.savefig(fig)
    plt.close(fig)


def page_strategy(pdf, ses, seite: int, gesamt: int) -> None:
    stints = f1lab.stints(ses)
    order = (ses.results.sort_values("Position")["Abbreviation"]
            .loc[lambda s: s.isin(stints["Driver"].unique())].tolist())
    fig, ax = plt.subplots(figsize=SEITENGROESSE)
    for drv in order:
        prev = 0
        for s in stints[stints["Driver"] == drv].sort_values("stint").itertuples():
            farbe = COMPOUND.get(str(s.Compound).upper(), MUTED)
            ax.barh(drv, s.laps, left=prev, color=farbe,
                   edgecolor="#00000055", height=0.7)
            prev += s.laps
    ax.invert_yaxis()
    ax.set_xlabel("Runde")
    ax.set_title("Reifenstrategien", fontsize=16, pad=20, color=FG, loc="left")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="x", alpha=0.3, linewidth=0.8, color=GRID)
    ax.set_axisbelow(True)
    _fusszeile(fig, ses, seite, gesamt)
    pdf.savefig(fig)
    plt.close(fig)


def page_positions(pdf, ses, seite: int, gesamt: int) -> None:
    pos = f1lab.position_progression(ses)
    order = ses.results.sort_values("Position")["Abbreviation"].tolist()
    fig, ax = plt.subplots(figsize=SEITENGROESSE)
    top3 = [d for d in order if d in pos.columns][:3]
    for drv in pos.columns:
        if drv in top3:
            continue
        ax.plot(pos.index, pos[drv], color=MUTED, lw=0.8, alpha=0.5)
    for i, drv in enumerate(top3):
        ax.plot(pos.index, pos[drv], color=SERIEN[i], lw=2.2, label=drv)
    ax.invert_yaxis()
    ax.set_xlabel("Runde")
    ax.set_ylabel("Position")
    ax.legend(loc="upper center", ncol=3, frameon=False, labelcolor=FG)
    ax.set_title("Positionsverlauf (Podium hervorgehoben)", fontsize=16,
                pad=20, color=FG, loc="left")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(alpha=0.3, linewidth=0.8, color=GRID)
    ax.set_axisbelow(True)
    _fusszeile(fig, ses, seite, gesamt)
    pdf.savefig(fig)
    plt.close(fig)


def neuestes_rennen(jahr: int) -> str:
    """für den unbeaufsichtigten Wochenauftrag: wählt ohne --gp automatisch das jüngste bereits gefahrene Rennen der Saison."""
    sched = fastf1.get_event_schedule(jahr, include_testing=False)
    vergangen = sched[sched["EventDate"] < pd.Timestamp(datetime.now())]
    if vergangen.empty:
        raise ValueError(f"noch kein abgeschlossenes Rennen in {jahr}")
    return vergangen.iloc[-1]["EventName"]


def _lade_mit_wiederholung(year: int, gp: str, versuche: int = 2):
    """f1lab.load() plus eine Absicherung. Session.load() kann ohne Exception zurückkehren.
    session.laps ist danach trotzdem manchmal leer/nicht geladen. FastF1 fängt einen
    Teil seiner eigenen API-Fehler intern ab und loggt nur eine WARNING statt eine
    Exception zu werfen.

    Ein Fehlschlag ist meist ``fastf1._api.SessionNotAvailableError``: FastF1s eigene
    Live-Timing-API meldet die Session als nicht verfügbar. Das ist kein Fehler in
    diesem Repo. Ein zweiter Versuch innerhalb desselben Laufs behebt das nicht
    zuverlässig, hilft aber bei echter Transienz und kostet nichts."""
    letzter_fehler: Exception | None = None
    for versuch in range(1, versuche + 1):
        ses = f1lab.load(year, gp, "R", telemetry=False)
        try:
            if not ses.laps.empty:
                return ses
            letzter_fehler = RuntimeError("session.laps ist leer")
        except fastf1.exceptions.DataNotLoadedError as exc:
            letzter_fehler = exc
        print(f"Warnung: Rundendaten fuer {gp} {year} nicht geladen "
             f"(Versuch {versuch}/{versuche}): {letzter_fehler} - "
             f"f1_api_support={ses.f1_api_support}")
    raise RuntimeError(
        f"Rundendaten fuer {gp} {year} liessen sich nach {versuche} "
        f"Versuchen nicht laden (f1_api_support={ses.f1_api_support}) - "
        "vermutlich meldet FastF1s Live-Timing-API die Session von dieser "
        "Netzwerkroute aus als nicht verfuegbar (SessionNotAvailableError, "
        "siehe Docstring), kein Code-Fehler in diesem Repo. Naechster "
        "Montags-Lauf hat gute Chancen, anders zu laufen.") from letzter_fehler


def build(year: int, gp: str | None, out: Path) -> None:
    f1lab.enable_cache()
    if gp is None:
        gp = neuestes_rennen(year)
        print(f"kein --gp angegeben, juengstes Rennen gewaehlt: {gp}")
    ses = _lade_mit_wiederholung(year, gp)

    seiten = [page_results, page_pace, page_strategy, page_positions]
    with PdfPages(out) as pdf:
        for i, seite_fn in enumerate(seiten, start=1):
            seite_fn(pdf, ses, i, len(seiten))
        info = pdf.infodict()
        info["Title"] = f"F1 Report {ses.event['EventName']} {year}"
        info["Author"] = "f1-data-analysis"
        info["CreationDate"] = datetime.now()
    print(f"geschrieben: {out} ({date.today().isoformat()})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, default=2024)
    ap.add_argument("--gp", default=None,
                    help="ohne Angabe: juengstes abgeschlossenes Rennen der Saison")
    ap.add_argument("--out", type=Path,
                    default=Path(__file__).parent / "out" / "race_report.pdf")
    a = ap.parse_args()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    build(a.year, a.gp, a.out)
