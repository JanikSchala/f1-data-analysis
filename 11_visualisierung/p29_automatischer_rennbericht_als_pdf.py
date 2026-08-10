"""
P29 - Automatischer Rennbericht als PDF
=======================================

Ein Skript, das nach jedem Rennen einen mehrseitigen Report mit Grafiken und Kennzahlen erzeugt.

Kategorie:   Visualisierung & Apps
Niveau:      Fortgeschritten
Aufwand:     4-5 h
Schwerpunkt: Datenanalyse, Engineering

WARUM DAS LOHNT
Ein Skript, das nach jedem Rennen von selbst durchlaeuft, ist mehr wert als zehn Notebooks, die man erst wieder verstehen muss. Der Unterschied zwischen Bastelei und Werkzeug.

VORGEHEN
  1. Report-Funktion pro Seite kapseln (Ergebnis, Pace, Strategie, Positionen)
  2. PdfPages nutzen, um alles in eine Datei zu schreiben
  3. Kopf- und Fusszeile mit Event und Erstellungsdatum
  4. Per CLI-Argument Jahr und Rennen waehlbar machen

GENUTZTE FASTF1-BAUSTEINE
  - fastf1 gesamt
  - matplotlib PdfPages
  - pandas

AUSBAUSTUFE  [umgesetzt]
Lass das Skript per GitHub Action jeden Montag automatisch laufen und den
Report als Artefakt hochladen.

VORGEHEN 1 nannte vier Seiten, die Vorlage baute drei - "Positionen" fehlte.
Ergaenzt mit f1lab.position_progression() (derselbe Baustein wie P20/die
Rennverlauf-Dashboardseite). VORGEHEN 3 (Kopf-/Fusszeile mit Event und
Erstellungsdatum) fehlte komplett: jede Seite hatte nur ihren eigenen Titel,
keine Fusszeile, und `info["CreationDate"] = None` loeschte das
PDF-Metadatenfeld sogar aktiv statt es zu setzen - das Gegenteil der
Anforderung. Jetzt traegt jede Seite dieselbe Fusszeile
(Event, Erstellungsdatum, Seitenzahl).

Pace- und Strategie-Seite riefen urspruenglich fastf1.plotting fuer Farben
und rechneten die Rundenzeiten selbst statt f1lab.pace_table()/stints() zu
nutzen - umgestellt auf f1lab (Treibstoffkorrektur, Bootstrap-Konfidenz-
intervall, dieselbe Quelle wie App und alle anderen Skripte) und
f1lab.design fuer den Hausstil statt fastf1s eigenem Farbschema.

AUSBAUSTUFE: .github/workflows/weekly-report.yml, jeden Montag 06:00 UTC.
Ein "jeden Montag automatisch" Job kann keinen festen Strecken-Namen im
Code haben (die Rennen wechseln) - --gp ist deshalb jetzt optional, ohne
Angabe waehlt das Skript automatisch das juengste bereits stattgefundene
Rennen der angegebenen Saison. Der Workflow selbst laueft hier nicht -
das wuerde echte, wiederkehrende GitHub-Actions-Minuten auf dem oeffentlichen
Repository verbrauchen, eine Entscheidung, die ausserhalb dieses Skripts
getroffen werden sollte, nicht automatisch beim Bauen des Skripts.
"""
from __future__ import annotations

import argparse
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
from f1lab.design import COMPOUND, FG, GRID, MUTED, SERIEN, matplotlib_stil

warnings.filterwarnings("ignore")
fastf1.set_log_level("ERROR")

SEITENGROESSE = (11.7, 8.3)   # A4 quer

plt.rcParams.update(matplotlib_stil())


def _fusszeile(fig, ses, seite: int, seiten_gesamt: int) -> None:
    """VORGEHEN 3: dieselbe Fuss-/Kopfzeile auf jeder Seite."""
    fig.text(0.06, 0.965, f"{ses.event['EventName']} {ses.event.year}",
             fontsize=11, color=MUTED, ha="left")
    fig.text(0.06, 0.02,
             f"Erstellt am {date.today().isoformat()} - f1-data-analysis",
             fontsize=8, color=MUTED, ha="left")
    fig.text(0.94, 0.02, f"Seite {seite}/{seiten_gesamt}", fontsize=8,
             color=MUTED, ha="right")


def page_results(pdf, ses, seite: int, gesamt: int) -> None:
    """VORGEHEN 1: Ergebnis."""
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
        cell.set_facecolor("#1e1e2a")
        cell.set_text_props(color=FG)
    ax.set_title(f"{ses.event['EventName']} {ses.event.year} - Ergebnis",
                fontsize=16, pad=20, color=FG, loc="left")
    _fusszeile(fig, ses, seite, gesamt)
    pdf.savefig(fig)
    plt.close(fig)


def page_pace(pdf, ses, seite: int, gesamt: int) -> None:
    """VORGEHEN 1: Pace, ueber f1lab.pace_table() (Treibstoffkorrektur,
    Bootstrap-Konfidenzintervall) statt eigener Median-Rechnung."""
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
    """VORGEHEN 1: Strategie, ueber f1lab.stints()."""
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
    """VORGEHEN 1: Positionen - in der Vorlage komplett uebersehene vierte
    Seite, ueber f1lab.position_progression() (siehe P20)."""
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
    """AUSBAUSTUFE: fuer den Wochenauftrag - ohne --gp automatisch das
    juengste bereits gefahrene Rennen der Saison waehlen."""
    sched = fastf1.get_event_schedule(jahr, include_testing=False)
    vergangen = sched[sched["EventDate"] < pd.Timestamp(datetime.now())]
    if vergangen.empty:
        raise ValueError(f"noch kein abgeschlossenes Rennen in {jahr}")
    return vergangen.iloc[-1]["EventName"]


def build(year: int, gp: str | None, out: Path) -> None:
    f1lab.enable_cache()
    if gp is None:
        gp = neuestes_rennen(year)
        print(f"kein --gp angegeben, juengstes Rennen gewaehlt: {gp}")
    ses = f1lab.load(year, gp, "R", telemetry=False)

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
