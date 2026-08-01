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

AUSBAUSTUFE
Lass das Skript per GitHub Action jeden Montag automatisch laufen und den Report als Artefakt hochladen.
"""

import argparse
from datetime import date

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import fastf1
import fastf1.plotting as f1plt

fastf1.Cache.enable_cache("~/f1_cache")
f1plt.setup_mpl(mpl_timedelta_support=False, color_scheme="fastf1")


def page_results(pdf, ses):
    fig, ax = plt.subplots(figsize=(11.7, 8.3))
    ax.axis("off")
    res = ses.results[["Position", "Abbreviation", "TeamName",
                       "GridPosition", "Status", "Points"]].copy()
    res["Gewinn"] = res["GridPosition"] - res["Position"]
    tbl = ax.table(cellText=res.round(0).astype(str).values,
                   colLabels=res.columns, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(8); tbl.scale(1, 1.5)
    ax.set_title(f"{ses.event['EventName']} {ses.event.year} - Ergebnis",
                 fontsize=16, pad=20)
    pdf.savefig(fig); plt.close(fig)


def page_pace(pdf, ses):
    laps = (ses.laps.pick_wo_box().pick_accurate()
            .pick_track_status("1").pick_quicklaps())
    med = (laps.groupby("Driver")["LapTime"].median()
           .dt.total_seconds().sort_values())
    fig, ax = plt.subplots(figsize=(11.7, 8.3))
    colors = [f1plt.get_driver_color(d, session=ses) for d in med.index]
    ax.barh(med.index, med - med.min(), color=colors)
    ax.invert_yaxis()
    ax.set_xlabel("Delta zur besten Median-Pace [s]")
    ax.set_title("Bereinigte Race Pace")
    pdf.savefig(fig); plt.close(fig)


def page_strategy(pdf, ses):
    stints = (ses.laps.groupby(["Driver", "Stint", "Compound"])["LapNumber"]
              .count().reset_index(name="Laenge"))
    order = ses.results.sort_values("Position")["Abbreviation"].tolist()
    fig, ax = plt.subplots(figsize=(11.7, 8.3))
    for drv in [d for d in order if d in stints["Driver"].values]:
        prev = 0
        for _, s in stints[stints["Driver"] == drv].sort_values("Stint").iterrows():
            ax.barh(drv, s["Laenge"], left=prev, edgecolor="black",
                    color=f1plt.get_compound_color(s["Compound"], session=ses))
            prev += s["Laenge"]
    ax.invert_yaxis(); ax.set_xlabel("Runde"); ax.set_title("Reifenstrategien")
    pdf.savefig(fig); plt.close(fig)


def build(year, gp, out):
    ses = fastf1.get_session(year, gp, "R")
    ses.load(telemetry=False)
    with PdfPages(out) as pdf:
        page_results(pdf, ses)
        page_pace(pdf, ses)
        page_strategy(pdf, ses)
        info = pdf.infodict()
        info["Title"] = f"F1 Report {gp} {year}"
        info["Author"] = "Janik Schala"
        info["CreationDate"] = None
    print(f"geschrieben: {out} ({date.today()})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, default=2024)
    ap.add_argument("--gp", default="Monza")
    ap.add_argument("--out", default="race_report.pdf")
    a = ap.parse_args()
    build(a.year, a.gp, a.out)
