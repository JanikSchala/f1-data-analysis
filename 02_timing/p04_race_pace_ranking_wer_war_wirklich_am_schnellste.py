"""rankt fahrer nach treibstoffkorrigierter median-pace mit konfidenzintervall"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")                      # nur dateien statt fenster

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
    """horizontaler barplot der korrigierten pace wie auf der pace-seite."""
    if tab.empty:
        # kein fahrer mit genug sauberen runden - kommt bei abgebrochenen
        # regenrennen real vor (siehe pace_table-Docstring).
        ax.text(0.5, 0.5, "keine auswertbare Pace", ha="center", va="center",
               color=MUTED)
        ax.axis("off")
        return
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
    """slope-chart: platzierung roh gegen korrigiert je fahrer. zeigt auch
    wie stark sich der platz verschiebt, nicht nur die neue reihenfolge."""
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

    if roh.empty or korr.empty:
        print("\n      keine auswertbare Pace - zu wenige saubere Runden je "
             "Fahrer (abgebrochenes Rennen?)")
        return
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
