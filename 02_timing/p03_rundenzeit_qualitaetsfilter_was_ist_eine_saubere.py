"""filtert runden auf sauberkeit und vergleicht die 107%-regel mit einem
fahrerweisen MAD-ausreisserdetektor"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")                      # nur dateien statt fenster

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
    """zaehlt wie viele runden jeder filterschritt kostet. die letzte stufe
    sind die "kandidaten": pace-taugliche runden vor der breitenfilterung,
    auf denen sich 107%-regel und MAD fair vergleichen lassen."""
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
    """vergleicht 107%-regel gegen fahrerweise MAD runde fuer runde und
    kategorisiert jede runde nach den vier moeglichen kombinationen."""
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


def zeichne_funnel(ax_funnel, stufen) -> None:
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


def zeichne_boxplot(ax, roh, sauber) -> None:
    """zeigt die verteilung der rundenzeiten vor und nach dem filtern,
    nicht nur die anzahl wie der funnel daneben."""
    bp = ax.boxplot([roh, sauber], tick_labels=["roh", "sauber"],
                    patch_artist=True, widths=0.55,
                    medianprops={"color": FG, "linewidth": 1.6},
                    whiskerprops={"color": MUTED}, capprops={"color": MUTED},
                    flierprops={"markeredgecolor": MUTED, "markersize": 3,
                               "alpha": 0.5})
    for patch, farbe in zip(bp["boxes"], (MUTED, SERIEN[0]), strict=True):
        patch.set_facecolor(farbe)
        patch.set_edgecolor(farbe)
        patch.set_alpha(0.85)

    ax.set_ylabel("Rundenzeit [s]")
    ax.set_title("Verteilung vorher/nachher", loc="left", color=FG,
                fontsize=13, pad=12)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="y", alpha=0.35, linewidth=0.8, color=GRID)
    ax.set_axisbelow(True)


def zeichne_vergleich(ax_vergleich, vergleich) -> None:
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

    roh_sekunden = ses.laps["LapTime"].dt.total_seconds().dropna()
    sauber_sekunden = f1lab.clean_laps(ses)["LapTime"].dt.total_seconds()
    print(f"\nVerteilung: Median roh {roh_sekunden.median():.3f}s (Streuung "
         f"{roh_sekunden.std():.3f}s) -> sauber {sauber_sekunden.median():.3f}s "
         f"(Streuung {sauber_sekunden.std():.3f}s)")

    fig, ax = plt.subplots(1, 3, figsize=(17, 5.5),
                           gridspec_kw={"width_ratios": [1, 0.6, 1.3]})
    zeichne_funnel(ax[0], stufen)
    zeichne_boxplot(ax[1], roh_sekunden, sauber_sekunden)
    zeichne_vergleich(ax[2], vergleich)
    fig.suptitle(f"{EVENT} {SEASON} - Rundenzeit-Qualitaetsfilter", x=0.09,
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
