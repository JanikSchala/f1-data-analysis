"""
P03 - Rundenzeit-Qualitaetsfilter: Was ist eine 'saubere' Runde?
================================================================

Rohdaten luegen. Out-Laps, In-Laps, Safety Car und geloeschte Runden verzerren jede Pace-Analyse.

Kategorie:   Timing & Rundenanalyse
Niveau:      Einsteiger
Aufwand:     2-3 h
Schwerpunkt: Datenanalyse, Strategie

WARUM DAS LOHNT
Das ist der haeufigste Anfaengerfehler in F1-Analysen. Wenn du im Gespraech erklaeren kannst, warum du track_status '1' filterst, klingst du wie jemand aus dem Fach.

VORGEHEN
  1. Alle Runden eines Rennens laden
  2. Schrittweise filtern und nach jedem Schritt zaehlen, wie viel wegfaellt
  3. Verteilung vorher/nachher als Boxplot vergleichen
  4. Eine wiederverwendbare clean_laps()-Funktion schreiben

GENUTZTE FASTF1-BAUSTEINE
  - Laps.pick_quicklaps
  - Laps.pick_accurate
  - Laps.pick_wo_box
  - Laps.pick_track_status
  - Laps.pick_not_deleted (indirekt - siehe Hinweis unten)

AUSBAUSTUFE  [umgesetzt]
Ausreisser-Detektor auf Basis der Median Absolute Deviation (MAD), verglichen
mit der 107%-Regel - auf denselben "Kandidaten"-Runden (nach Box/Genauigkeit/
gruener Flagge/Streichung, vor der Breitenfilterung).

Zwei Abweichungen von der urspruenglichen Formulierung:

Erstens nutzt clean_laps() nicht Laps.pick_not_deleted(), sondern ein eigenes
not_deleted_mask() aus f1lab.session. FastF1 invertiert die Spalte 'Deleted'
direkt mit '~'; ist sie object-dtype - was passiert, sobald ein Wert fehlt -
wirft pandas einen TypeError. Genau das tritt in diesem Rennen auf (Spanien
2024: die komplette Spalte ist object-dtype mit lauter None). Der Fehler ist
per Regressionstest in tests/test_session.py abgesichert.

Zweitens laeuft die MAD ueber die Kandidaten je Fahrer, nicht ueber das ganze
Feld. Global gerechnet meldet sie auf dieser Renndistanz fast keine Ausreisser
(1 von 1206 Runden) - der Grund ist nicht, dass MAD schlecht waere, sondern
dass Reifenabbau und Spritgewicht ueber ein ganzes Rennen eine derart grosse,
aber voellig legitime Streuung erzeugen, dass eine einzelne langsame Runde
darin nicht mehr auffaellt. Je Fahrer gerechnet misst MAD an der Streuung
genau dieses einen Fahrers - ein fairer Vergleich mit der 107%-Regel, die
ebenfalls fahrerunabhaengig eine einzige Grenze zieht.

Ergebnis fuer Spanien 2024: die 107%-Regel verwirft 82 von 1206
Kandidaten-Runden. Die MAD stimmt dem nur in einem einzigen Fall zu - die
uebrigen 81 sind Runden, die zwar langsamer als 107% der schnellsten Runde
des Rennens sind, aber innerhalb der normalen Streuung des jeweiligen
Fahrers liegen (spaete, hoch beanspruchte Reifen zum Beispiel). Die
107%-Regel misst also nicht "Ausreisser", sondern "langsamer als eine feste,
globale Grenze" - beides faellt nur zusammen, wenn die Grenze zur Streuung
des Fahrers passt.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")                      # kein Fenster, nur Dateien

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import f1lab
from f1lab.design import FG, GRID, MUTED, SERIEN, matplotlib_stil
from f1lab.session import not_deleted_mask

warnings.filterwarnings("ignore")

OUT = Path(__file__).parent / "out"
OUT.mkdir(exist_ok=True)

SEASON, EVENT = 2024, "Spain"

plt.rcParams.update(matplotlib_stil())


def funnel(ses) -> tuple[list[tuple[str, int]], pd.DataFrame]:
    """Zaehlt, wie viele Runden jeder Filterschritt kostet.

    Die letzte Stufe sind die "Kandidaten": alles, was fuer eine Pace-Analyse
    grundsaetzlich infrage kommt, aber noch nicht auf Rundenzeit-Breite
    gefiltert ist - genau die Menge, auf der sich 107%-Regel und MAD fair
    vergleichen lassen.
    """
    stufen = []
    laps = ses.laps
    stufen.append(("roh", laps))

    laps = laps.pick_wo_box()
    stufen.append(("ohne Boxenrunden", laps))

    laps = laps.pick_accurate()
    stufen.append(("akkurat markiert", laps))

    laps = laps.pick_track_status("1")
    stufen.append(("gruene Flagge", laps))

    laps = laps[not_deleted_mask(laps["Deleted"]).to_numpy()]
    stufen.append(("nicht gestrichen (Kandidaten)", laps))

    counts = [(name, len(df)) for name, df in stufen]
    return counts, laps


def vergleiche_ausreisser(kandidaten: pd.DataFrame) -> pd.DataFrame:
    """107%-Regel gegen fahrerweise MAD, Runde fuer Runde.

    Beide Methoden liefern eine Ja/Nein-Entscheidung je Runde. Die vier
    moeglichen Kombinationen sind die eigentliche Aussage: wo stimmen sie
    zu, und - wichtiger - wo widersprechen sie sich und warum.
    """
    kand = kandidaten.reset_index(drop=True)
    sekunden = kand["LapTime"].dt.total_seconds().to_numpy()

    behalten_107 = kand.index.isin(kand.pick_quicklaps(1.07).index)

    ausreisser_mad = np.zeros(len(kand), dtype=bool)
    for _, idx in kand.groupby("Driver").groups.items():
        werte = sekunden[idx]
        ausreisser_mad[idx] = f1lab.mad_outlier_mask(werte)
    behalten_mad = ~ausreisser_mad

    kategorie = np.select(
        [behalten_107 & behalten_mad,
         ~behalten_107 & ~behalten_mad,
         ~behalten_107 & behalten_mad,
         behalten_107 & ~behalten_mad],
        ["beide behalten", "beide verwerfen",
         "nur 107%-Regel verwirft", "nur MAD verwirft"],
        default="unklar")

    out = kand[["Driver", "LapNumber"]].copy()
    out["sekunden"] = sekunden
    out["kategorie"] = kategorie
    return out


FARBE = {
    "beide behalten": SERIEN[0],
    "nur 107%-Regel verwirft": SERIEN[1],
    "nur MAD verwirft": SERIEN[2],
    "beide verwerfen": MUTED,
}


def zeichne(ax_funnel, stufen, ax_vergleich, vergleich, titel: str) -> None:
    namen = [n for n, _ in stufen]
    werte = [v for _, v in stufen]
    balken = ax_funnel.barh(namen, werte, color=SERIEN[0], height=0.6)
    for b, v in zip(balken, werte, strict=True):
        ax_funnel.text(v + max(werte) * 0.015, b.get_y() + b.get_height() / 2,
                       f"{v}", va="center", color=FG, fontsize=10)
    ax_funnel.invert_yaxis()
    ax_funnel.set_xlabel("Runden")
    ax_funnel.set_title("Wie viel jeder Filterschritt kostet", loc="left",
                        color=FG, fontsize=13, pad=12)
    for side in ("top", "right"):
        ax_funnel.spines[side].set_visible(False)
    ax_funnel.grid(axis="x", alpha=0.35, linewidth=0.8)
    ax_funnel.set_axisbelow(True)

    for kat, farbe in FARBE.items():
        teil = vergleich[vergleich["kategorie"] == kat]
        ax_vergleich.scatter(teil["LapNumber"], teil["sekunden"], s=16,
                             color=farbe, label=f"{kat} (n={len(teil)})",
                             alpha=0.85, linewidth=0)
    ax_vergleich.set_xlabel("Runde")
    ax_vergleich.set_ylabel("Rundenzeit [s]")
    ax_vergleich.set_title("107%-Regel gegen fahrerweise MAD", loc="left",
                           color=FG, fontsize=13, pad=12)
    ax_vergleich.legend(loc="upper left", bbox_to_anchor=(0, -0.14), ncol=2,
                        frameon=False, labelcolor=FG, fontsize=9,
                        handletextpad=0.4, columnspacing=1.4)
    ax_vergleich.grid(alpha=0.35, linewidth=0.8, color=GRID)
    ax_vergleich.set_axisbelow(True)
    for side in ("top", "right"):
        ax_vergleich.spines[side].set_visible(False)


def main():
    f1lab.enable_cache()

    print(f"[1/3] {EVENT} {SEASON} laden ...")
    ses = f1lab.load(SEASON, EVENT, "R")

    print("[2/3] Filterstufen ...")
    stufen, kandidaten = funnel(ses)
    for i, (name, n) in enumerate(stufen):
        verlust = "" if i == 0 else f" (-{stufen[i - 1][1] - n})"
        print(f"      {name:<30} n={n:>4}{verlust}")

    print("[3/3] MAD gegen 107%-Regel ...")
    vergleich = vergleiche_ausreisser(kandidaten)
    zusammenfassung = vergleich["kategorie"].value_counts()
    for kat in FARBE:
        print(f"      {kat:<26} {zusammenfassung.get(kat, 0)}")

    fig, ax = plt.subplots(1, 2, figsize=(13, 5.5),
                           gridspec_kw={"width_ratios": [1, 1.3]})
    zeichne(ax[0], stufen, ax[1], vergleich,
           f"{EVENT} {SEASON} - Datenqualitaet")
    fig.suptitle(f"{EVENT} {SEASON} - Rundenzeit-Qualitaetsfilter", x=0.125,
                ha="left", fontsize=16, color=FG, y=1.01)
    plt.tight_layout()
    path = OUT / "qualitaetsfilter.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"\n      -> {path}")

    strittig = vergleich[vergleich["kategorie"] == "nur 107%-Regel verwirft"]
    if not strittig.empty:
        print("\nBeispiele, die nur die 107%-Regel verwirft (fuer MAD "
              "normale Streuung des jeweiligen Fahrers):")
        beispiel = strittig.nlargest(5, "sekunden")[["Driver", "LapNumber", "sekunden"]]
        print(beispiel.to_string(index=False))


if __name__ == "__main__":
    main()
