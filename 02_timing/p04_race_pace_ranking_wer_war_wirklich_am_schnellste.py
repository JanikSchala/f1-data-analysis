"""
P04 - Race Pace Ranking: Wer war wirklich am schnellsten?
=========================================================

Endergebnis != Pace. Rangliste nach bereinigter Median-Pace pro Fahrer und Team, mit Konfidenzintervall.

Kategorie:   Timing & Rundenanalyse
Niveau:      Fortgeschritten
Aufwand:     3-4 h
Schwerpunkt: Datenanalyse, Strategie

WARUM DAS LOHNT
Genau diese Frage stellt sich an der Boxenmauer nach jedem Rennen. Ein Ranking ohne Unsicherheitsangabe ist nur eine Meinung mit Nachkommastellen.

VORGEHEN
  1. Saubere Runden filtern
  2. Median-Pace je Fahrer berechnen, Delta zum Schnellsten
  3. Bootstrap-Konfidenzintervall (1000 Resamples)
  4. Horizontaler Barplot in Team-Farben

GENUTZTE FASTF1-BAUSTEINE
  - Session.laps
  - Laps.pick_quicklaps
  - Laps.groupby

AUSBAUSTUFE  [umgesetzt]
Treibstoffeffekt herausrechnen (0.03 s/Runde/kg, siehe f1lab.core.fuel_correct)
und alle Runden auf leeren Tank normieren.

Steckt jetzt in f1lab.race_pace() selbst statt nur in diesem Skript - Grund:
pace_table() ist bereits die Grundlage der Pace-Seite im Dashboard, und eine
unkorrigierte Pace ist keine Fussnote, sondern schlicht falsch: der Median
haengt sonst nicht nur vom Tempo ab, sondern zusaetzlich davon, *wann* im
Rennen die sauberen Runden eines Fahrers liegen. race_pace(kg_per_lap=0)
schaltet die Korrektur ab und liefert damit exakt das alte, unkorrigierte
Verhalten - der Vergleich unten nutzt genau das.

Silverstone 2024 zeigt, warum das mehr als Kosmetik ist: unkorrigiert fuehrt
Hamilton, korrigiert fuehrt Russell - beide Mercedes-Piloten desselben
Rennens, das Ranking dreht sich an der Spitze allein durch die Korrektur.
Der Grund liegt nicht in unterschiedlichem Tempo, sondern darin, wann die
jeweils sauberen (nicht durch Verkehr, Undercut oder Ausreisser verzerrten)
Runden im Rennen liegen - bei vollerem oder leererem Tank.

Eine zweite, bewusste Abweichung von VORGEHEN Punkt 4: der Barplot faerbt
nicht nach Team, sondern danach, ob sich ein Fahrer vom Schnellsten
statistisch absetzt (Intervall ueberlappt nicht). 20 Fahrer brauchen 10
Teamfarben - das sprengt die CVD-geprueften drei Kategoriefarben aus
f1lab.design (MAX_SERIEN, siehe dortige Dokumentation) und waere an dieser
Stelle auch die schwaechere Information: die Farbe der Pace-Seite im
Dashboard beantwortet dieselbe Frage genauso, und genau die - nicht die
Teamzugehoerigkeit - ist die eigentliche Aussage dieses Diagramms. Team-
Farben zeigt stattdessen assets/race_pace.png (make_assets.py), wo die
Zuordnung selbst die Aussage ist.
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
from f1lab.design import FG, GRID, MUTED, POSITIV, SERIEN, matplotlib_stil

warnings.filterwarnings("ignore")

OUT = Path(__file__).parent / "out"
OUT.mkdir(exist_ok=True)

SEASON, EVENT = 2024, "Silverstone"

plt.rcParams.update(matplotlib_stil())


def zeichne_ranking(ax, tab: pd.DataFrame) -> None:
    """Horizontaler Barplot der korrigierten Pace, wie auf der Pace-Seite."""
    p = tab.iloc[::-1]
    bester_hi = tab["ci_hi"].iloc[0]
    farben = [POSITIV if lo > bester_hi else SERIEN[0] for lo in p["ci_lo"]]
    farben[-1] = FG

    err = [p["delta_s"] - p["ci_lo"], p["ci_hi"] - p["delta_s"]]
    ax.barh(p["driver"], p["delta_s"], xerr=err, color=farben,
           error_kw={"ecolor": MUTED, "elinewidth": 1.2, "capsize": 2})
    ax.set_xlabel("Rueckstand auf den Schnellsten [s]")
    ax.set_title("Treibstoffkorrigierte Race Pace", loc="left", color=FG,
                fontsize=13, pad=12)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="x", alpha=0.35, linewidth=0.8)
    ax.set_axisbelow(True)


def zeichne_rangwechsel(ax, roh: pd.DataFrame, korr: pd.DataFrame) -> None:
    """Slope-Chart: Platzierung roh gegen korrigiert, verbunden je Fahrer.

    Die Reihenfolge einer Rangliste allein zeigt nicht, *wie viel* sich
    verschiebt - zwei benachbarte Linien, die sich kreuzen, sagen mehr als
    die Zahl daneben.
    """
    r = roh.reset_index(drop=True)
    k = korr.reset_index(drop=True)
    platz_roh = {d: i for i, d in enumerate(r["driver"])}
    platz_korr = {d: i for i, d in enumerate(k["driver"])}
    fahrer = [d for d in platz_roh if d in platz_korr]

    n = len(fahrer)
    for d in fahrer:
        y0, y1 = platz_roh[d], platz_korr[d]
        verschoben = abs(y0 - y1) >= 2
        farbe = SERIEN[1] if verschoben else GRID
        ax.plot([0, 1], [y0, y1], color=farbe,
               linewidth=1.8 if verschoben else 1.0,
               alpha=0.95 if verschoben else 0.55, zorder=2 if verschoben else 1)
        for x, y in ((0, y0), (1, y1)):
            ax.scatter([x], [y], s=26, color=farbe, zorder=3,
                      edgecolor="none")
        if verschoben:
            ax.text(-0.04, y0, d, ha="right", va="center", color=FG,
                   fontsize=9)
            ax.text(1.04, y1, d, ha="left", va="center", color=FG,
                   fontsize=9)

    ax.set_xlim(-0.35, 1.35)
    ax.set_ylim(n - 0.5, -0.5)
    ax.set_xticks([0, 1], ["roh", "korrigiert"])
    ax.set_yticks([])
    ax.set_title("Wer wechselt den Platz durch die Korrektur", loc="left",
                color=FG, fontsize=13, pad=12)
    for side in ax.spines.values():
        side.set_visible(False)
    ax.text(0.5, -0.07, "grau: Platz haelt (< 2 Positionen) - "
           "orange, beschriftet: >= 2 Positionen Unterschied",
           ha="center", va="top", color=MUTED, fontsize=9,
           transform=ax.transAxes)


def main():
    f1lab.enable_cache()

    print(f"[1/2] {EVENT} {SEASON} laden ...")
    ses = f1lab.load(SEASON, EVENT, "R")

    roh = f1lab.pace_table(ses, kg_per_lap=0)
    korr = f1lab.pace_table(ses)

    print("\nRoh (unkorrigiert):")
    print(roh[["driver", "team", "laps", "delta_s"]].to_string(index=False))
    print("\nKorrigiert (Treibstoffeffekt herausgerechnet):")
    print(korr[["driver", "team", "laps", "delta_s"]].to_string(index=False))

    fuehrend_roh, fuehrend_korr = roh["driver"].iloc[0], korr["driver"].iloc[0]
    print(f"\nFuehrend roh: {fuehrend_roh}  |  Fuehrend korrigiert: "
         f"{fuehrend_korr}"
         + ("  <- dreht sich durch die Korrektur"
            if fuehrend_roh != fuehrend_korr else ""))

    print("[2/2] Grafik ...")
    fig, ax = plt.subplots(1, 2, figsize=(13, 8),
                           gridspec_kw={"width_ratios": [1, 0.85]})
    zeichne_ranking(ax[0], korr)
    zeichne_rangwechsel(ax[1], roh, korr)
    fig.suptitle(f"{EVENT} {SEASON} - Race Pace Ranking", x=0.125, ha="left",
                fontsize=16, color=FG, y=1.01)
    plt.tight_layout()
    path = OUT / "race_pace_ranking.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"\n      -> {path}")


if __name__ == "__main__":
    main()
