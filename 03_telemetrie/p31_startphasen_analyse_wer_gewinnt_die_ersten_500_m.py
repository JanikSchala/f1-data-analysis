"""misst launch-distanz und positionsgewinn nach dem start und vergleicht die launch-staerke ueber die saison"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")                      # schreibt nur dateien ohne fenster

import matplotlib.pyplot as plt
import pandas as pd

import f1lab
from f1lab.design import FG, GRID, MUTED, SERIEN, matplotlib_stil

warnings.filterwarnings("ignore")

OUT = Path(__file__).parent / "out"
OUT.mkdir(exist_ok=True)

SEASON, EVENT, IDENT = 2024, "Austria", "R"
SAISON_EVENTS = ["Bahrain", "Saudi Arabia", "Australia", "Japan", "China",
                "Miami", "Emilia Romagna", "Monaco", "Italy"]
FENSTER_S = 8.0

plt.rcParams.update(matplotlib_stil())


def saison_scan(events: list[str], year: int = SEASON) -> pd.DataFrame:
    """berechnet dieselben startkennzahlen ueber mehrere rennen."""
    zeilen = []
    for gp in events:
        ses = f1lab.load(year, gp, "R", telemetry=True)
        sp = f1lab.start_performance(ses, fenster_s=FENSTER_S)
        if not sp.empty:
            zeilen.append(sp.assign(round=int(ses.event["RoundNumber"]),
                                    event=ses.event["EventName"]))
        print(f"      {ses.event['EventName']} ausgewertet")
    return pd.concat(zeilen, ignore_index=True) if zeilen else pd.DataFrame()


def zeichne_positionsgewinn(ax, df: pd.DataFrame) -> None:
    d = df.dropna(subset=["m_nach_5s", "gewinn"])
    ax.scatter(d["m_nach_5s"], d["gewinn"], s=70, color=SERIEN[0], zorder=3)
    for _, r in d.iterrows():
        ax.annotate(r["driver"], (r["m_nach_5s"], r["gewinn"]), xytext=(6, 4),
                   textcoords="offset points", color=FG, fontsize=8)
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.set_xlabel("Distanz nach 5 s [m]")
    ax.set_ylabel("Positionsgewinn Runde 1")
    ax.set_title(f"Start-Distanz gegen Positionsgewinn - {EVENT} {SEASON}",
                loc="left", color=FG, fontsize=13, pad=10)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(alpha=0.3, linewidth=0.8, color=GRID)
    ax.set_axisbelow(True)


def zeichne_saisontrend(ax, saison: pd.DataFrame, top_n: int = 3) -> None:
    """zeigt die abweichung vom feld-mittel des jeweiligen rennens statt der rohen distanz.
    rohe distanz haengt stark von der streckengeometrie ab und ist ueber die saison nicht vergleichbar."""
    saison = saison.copy()
    saison["rel_m"] = saison.groupby("event")["m_nach_5s"].transform(
        lambda s: s - s.mean())

    pivot = saison.pivot_table(index="round", columns="driver", values="rel_m")
    vollstaendig = pivot.dropna(axis=1)
    top = (vollstaendig.mean().sort_values(ascending=False).index[:top_n]
          if not vollstaendig.empty else pivot.mean().sort_values(
              ascending=False).index[:top_n])

    for drv in pivot.columns:
        if drv not in top:
            ax.plot(pivot.index, pivot[drv], color=GRID, lw=1.0, alpha=0.8)
    for i, drv in enumerate(top):
        serie = pivot[drv].dropna()
        ax.plot(serie.index, serie, color=SERIEN[i], lw=2.2, marker="o",
               markersize=4, label=drv)

    ax.axhline(0, color=MUTED, lw=0.8)
    ax.set_xlabel("Rennwoche (Saison 2024)")
    ax.set_ylabel("Abweichung vom Feld-Mittel [m]")
    ax.set_title(f"Launch-Staerke relativ zum Feld, Top {top_n} im Mittel "
                f"hervorgehoben", loc="left", color=FG, fontsize=13, pad=10)
    ax.legend(loc="lower left", frameon=False, labelcolor=FG, fontsize=9,
             ncol=top_n)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(alpha=0.3, linewidth=0.8, color=GRID)
    ax.set_axisbelow(True)


def main():
    f1lab.enable_cache()

    print(f"[1/3] {EVENT} {SEASON} {IDENT} laden (mit Telemetrie) ...")
    ses = f1lab.load(SEASON, EVENT, IDENT, telemetry=True)
    start = f1lab.start_performance(ses, fenster_s=FENSTER_S)
    print(start.to_string(index=False))

    print(f"\n[2/3] Saison-Scan ueber {len(SAISON_EVENTS)} Rennen mit "
         f"Telemetrie ...")
    saison = saison_scan(SAISON_EVENTS)
    ranking = (saison.groupby("driver")["m_nach_5s"]
              .agg(["mean", "count"]).sort_values("mean", ascending=False))
    print("\nMittlere Startdistanz nach 5s, ganze Saison (Top 8):")
    print(ranking.head(8).round(1).to_string())

    print("\n[3/3] Grafik ...")
    fig, ax = plt.subplots(1, 2, figsize=(15, 6))
    zeichne_positionsgewinn(ax[0], start)
    zeichne_saisontrend(ax[1], saison)
    fig.suptitle("Startphasen-Analyse: Launch-Distanz, Positionsgewinn, "
                "Saisonverlauf", x=0.09, ha="left", fontsize=16, color=FG,
                y=1.02)
    plt.tight_layout()
    path = OUT / "startphasen_analyse.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"\n      -> {path}")


if __name__ == "__main__":
    main()
