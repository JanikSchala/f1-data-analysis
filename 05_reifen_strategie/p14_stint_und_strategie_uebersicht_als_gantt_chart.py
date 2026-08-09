"""
P14 - Stint- und Strategie-Uebersicht als Gantt-Chart
=====================================================

Die klassische Strategiegrafik: jeder Fahrer eine Zeile, jeder Stint ein Balken in Compound-Farbe.

Kategorie:   Reifen & Strategie
Niveau:      Einsteiger
Aufwand:     2-3 h
Schwerpunkt: Strategie, Datenanalyse

WARUM DAS LOHNT
Diese Grafik hat jedes Team an der Boxenmauer. Schnell gebaut, sofort verstaendlich - und danach schaust du jedes Rennen anders.

VORGEHEN
  1. Stints je Fahrer aggregieren: Start-Runde, Laenge, Compound
  2. Nach Endposition sortieren
  3. Gestapelte horizontale Balken mit Compound-Farben
  4. Compound-Kuerzel und Stintlaenge beschriften

GENUTZTE FASTF1-BAUSTEINE
  - Laps groupby Stint
  - Laps Compound/FreshTyre
  - fastf1.plotting.get_compound_color

AUSBAUSTUFE  [umgesetzt]
Ergaenze eine zweite Spalte mit dem Positionsgewinn/-verlust je Stint,
damit man sieht, welche Strategie sich ausgezahlt hat.

VORGEHEN 1 nutzt f1lab.stints() statt einer eigenen Kopie der
Stint-Aggregation - dieselbe Funktion wie die Reifen-Seite im Dashboard.
Kompaktere Compound-Farben aus f1lab.design.COMPOUND statt
fastf1.plotting.get_compound_color(), aus demselben Grund wie ueberall im
Projekt: eine Quelle der Wahrheit fuer Farbe, die Skript und App teilen.

Die AUSBAUSTUFE-Spalte rechts vom Gantt zeigt je Stint die Positions-
Differenz zwischen erster und letzter Runde des Stints (nicht Overtakes -
siehe P20/f1lab.overtakes_matrix() dafuer -, sondern die rohe
Positionsverschiebung inklusive Boxenstopp-Artefakt: ein Fahrer taucht
direkt nach seinem eigenen Stopp meist kurz weiter hinten auf, was der
naechste Stint dann wieder "aufholt"). Ungarn 2024 R: Perez und Sargeant
gewinnen am meisten Positionen waehrend ihrer Stints (+15 bzw. +14 ueber
das Rennen), aber nur Perez uebersetzt das in ein besseres Zielergebnis als
den Ausgangspunkt (Startplatz 16, Ziel 7). Sargeants Stint-Gewinne loesen
sich wieder auf - netto verliert er ueber das ganze Rennen drei Plaetze
(Start P14, Ziel P17): ein gutes Beispiel dafuer, dass die Spalte
Stint-Pace zeigt, nicht das Endergebnis.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")                      # kein Fenster, nur Dateien

import matplotlib.pyplot as plt
import pandas as pd

import f1lab
from f1lab.design import COMPOUND, FG, GRID, MUTED, NEGATIV, POSITIV, matplotlib_stil

warnings.filterwarnings("ignore")

OUT = Path(__file__).parent / "out"
OUT.mkdir(exist_ok=True)

SEASON, EVENT, IDENT = 2024, "Hungary", "R"

plt.rcParams.update(matplotlib_stil())


def positionswechsel(ses, stints: pd.DataFrame) -> pd.DataFrame:
    """AUSBAUSTUFE: Position bei Stint-Beginn gegen Stint-Ende."""
    pos = ses.laps.set_index(["Driver", "LapNumber"])["Position"]

    def bei(drv, lap):
        try:
            v = pos.loc[(drv, lap)]
        except KeyError:
            return float("nan")
        return float(v.iloc[0]) if isinstance(v, pd.Series) else float(v)

    out = stints.copy()
    out["pos_start"] = [bei(r.Driver, r.start) for r in out.itertuples()]
    out["pos_end"] = [bei(r.Driver, r.end) for r in out.itertuples()]
    out["gain"] = out["pos_start"] - out["pos_end"]
    return out


def zeichne_gantt(ax, stints: pd.DataFrame, order: list[str]) -> None:
    """VORGEHEN 3-4: gestapelte Balken, Compound-Kuerzel ab 4 Runden Laenge."""
    for drv in order:
        prev = 0
        for s in stints[stints["Driver"] == drv].sort_values("stint").itertuples():
            farbe = COMPOUND.get(str(s.Compound).upper(), MUTED)
            ax.barh(drv, s.laps, left=prev, color=farbe, edgecolor="#00000055",
                    height=0.7)
            if s.laps >= 4:
                ax.text(prev + s.laps / 2, drv, str(s.Compound)[0], ha="center",
                        va="center", fontsize=8, fontweight="bold", color="#111")
            prev += s.laps
    ax.invert_yaxis()
    ax.set_xlabel("Runde")
    ax.set_title(f"{EVENT} {SEASON} - Reifenstrategien (sortiert nach Zielposition)",
                loc="left", color=FG, fontsize=13, pad=10)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="x", alpha=0.3, linewidth=0.8, color=GRID)
    ax.set_axisbelow(True)


def zeichne_positionsspalte(ax, wechsel: pd.DataFrame, order: list[str]) -> None:
    """AUSBAUSTUFE: zweite Spalte, ein Punkt je Stint mit Positionsgewinn/
    -verlust (POSITIV/NEGATIV aus f1lab.design - Wertung, keine Kategorie)."""
    for drv in order:
        g = wechsel[wechsel["Driver"] == drv].sort_values("stint")
        for s in g.itertuples():
            if pd.isna(s.gain):
                continue
            farbe = POSITIV if s.gain > 0 else (NEGATIV if s.gain < 0 else MUTED)
            ax.scatter(s.stint, drv, s=90, color=farbe, zorder=3,
                      edgecolors=GRID, linewidths=0.8)
            if s.gain != 0:
                ax.text(s.stint, drv, f"{s.gain:+.0f}", fontsize=6.5,
                       ha="center", va="center", color="#0c0c12", zorder=4,
                       fontweight="bold")
    ax.tick_params(axis="y", left=False, labelleft=False)
    # kein eigenes invert_yaxis(): ax teilt die y-Achse mit dem Gantt (sharey)
    # und uebernimmt dessen bereits invertierte Reihenfolge automatisch.
    ax.set_xticks(range(1, int(wechsel["stint"].max()) + 1))
    ax.set_xlabel("Stint")
    ax.set_title("Positions-\ndelta", loc="left", color=FG, fontsize=13, pad=10)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="x", alpha=0.2, linewidth=0.8, color=GRID)
    ax.set_axisbelow(True)


def main():
    f1lab.enable_cache()

    print(f"[1/3] {EVENT} {SEASON} {IDENT} laden ...")
    ses = f1lab.load(SEASON, EVENT, IDENT, telemetry=False)

    print("[2/3] Stints aggregieren (VORGEHEN 1), nach Zielposition sortieren "
         "(VORGEHEN 2) ...")
    stints = f1lab.stints(ses)
    order = (ses.results.sort_values("Position")["Abbreviation"]
             .loc[lambda s: s.isin(stints["Driver"].unique())].tolist())
    print(f"      {len(stints)} Stints, {len(order)} Fahrer, "
         f"Mischungen: {sorted(stints['Compound'].unique())}")
    print(stints.groupby("Compound")["laps"].agg(["mean", "max", "count"]).round(1))

    print("\n[3/3] AUSBAUSTUFE: Positionswechsel je Stint ...")
    wechsel = positionswechsel(ses, stints)
    netto = wechsel.groupby("Driver")["gain"].sum().sort_values(ascending=False)
    print("      Groesster Positionsgewinn ueber alle Stints:")
    print(netto.head(5).to_string())
    res = ses.results.set_index("Abbreviation")
    for drv in netto.head(2).index:
        start = res.loc[drv, "GridPosition"]
        ziel = res.loc[drv, "Position"]
        print(f"      {drv}: Start P{start:.0f}, Ziel P{ziel:.0f} "
             f"(Netto laut Grid: {start - ziel:+.0f})")

    print("\nGrafik ...")
    fig, (links, rechts) = plt.subplots(
        1, 2, figsize=(13, max(4.5, 0.42 * len(order))),
        gridspec_kw={"width_ratios": [5, 1]}, sharey=True)
    zeichne_gantt(links, stints, order)
    zeichne_positionsspalte(rechts, wechsel, order)
    plt.tight_layout()
    path = OUT / "strategie_gantt.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"\n      -> {path}")


if __name__ == "__main__":
    main()
