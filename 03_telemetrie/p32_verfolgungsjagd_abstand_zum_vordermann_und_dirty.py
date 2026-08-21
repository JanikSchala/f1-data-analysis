"""regressiert bereinigte rundenzeit gegen abstand zum vordermann und vergleicht den dirty-air-effekt ueber mehrere strecken"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")                      # schreibt nur dateien ohne fenster

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

import f1lab
from f1lab.design import FG, GRID, MUTED, RAMPE, SERIEN, matplotlib_stil

RAMPE_CMAP = LinearSegmentedColormap.from_list("rampe", RAMPE)

warnings.filterwarnings("ignore")

OUT = Path(__file__).parent / "out"
OUT.mkdir(exist_ok=True)

SEASON, EVENT, IDENT = 2024, "Spain", "R"
DRIVER = "SAI"
STRECKEN = ["Bahrain", "Monaco", "Italy"]           # schnell -> langsam/eng
FAHRER_JE_STRECKE = 8
NAH_SCHWELLE_M = 50.0

plt.rcParams.update(matplotlib_stil())


def zeichne_regression(ax, d: pd.DataFrame, slope: float, inter: float,
                       r2: float, titel: str) -> None:
    sc = ax.scatter(d["anteil_nah"], d["sec_corr"], c=d["tyre_life"],
                    cmap=RAMPE_CMAP, s=55, zorder=3)
    xs = np.linspace(d["anteil_nah"].min(), d["anteil_nah"].max(), 50)
    ax.plot(xs, slope * xs + inter, color=SERIEN[1], lw=2.2)
    cb = ax.figure.colorbar(sc, ax=ax, pad=0.02)
    cb.set_label("Reifenalter [Runden]", color=MUTED, fontsize=9)
    cb.ax.tick_params(colors=MUTED, length=0)
    cb.outline.set_visible(False)

    ax.set_xlabel(f"Zeitanteil unter {NAH_SCHWELLE_M:.0f} m Abstand [%]")
    ax.set_ylabel("Bereinigte Rundenzeit [s]")
    ax.set_title(f"{titel}: {slope:+.3f} s je 1%, R²={r2:.2f}", loc="left",
                color=FG, fontsize=13, pad=10)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(alpha=0.3, linewidth=0.8, color=GRID)
    ax.set_axisbelow(True)


def zeichne_streckenvergleich(ax, ergebnisse: pd.DataFrame) -> None:
    e = ergebnisse.sort_values("slope")
    farben = [SERIEN[1] if s > 0 else SERIEN[2] for s in e["slope"]]
    ax.barh(e["strecke"], e["slope"], color=farben, height=0.6)
    for y, (_s, r2, n) in enumerate(zip(e["slope"], e["r2"], e["n"], strict=True)):
        ax.text(0, y + 0.38, f"R²={r2:.2f}, n={n}", va="bottom",
               ha="center", color=MUTED, fontsize=8.5)
    ax.axvline(0, color=MUTED, lw=0.8)
    ax.set_xlabel(f"Dirty-Air-Effekt [s je 1% Zeit unter "
                 f"{NAH_SCHWELLE_M:.0f} m]")
    ax.set_title("Streckenvergleich: Highspeed gegen Winkelkurs", loc="left",
                color=FG, fontsize=13, pad=10)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="x", alpha=0.3, linewidth=0.8, color=GRID)
    ax.set_axisbelow(True)


def main():
    f1lab.enable_cache()

    print(f"[1/3] {EVENT} {SEASON} {IDENT} laden, Fahrer {DRIVER} ...")
    ses = f1lab.load(SEASON, EVENT, IDENT, telemetry=True)
    df = f1lab.close_following(ses, DRIVER, nah_schwelle_m=NAH_SCHWELLE_M)
    slope, inter, r2, d = f1lab.dirty_air_effect(df)
    print(f"      Dirty-Air-Effekt: {slope:+.4f} s je 1% Zeit unter "
         f"{NAH_SCHWELLE_M:.0f} m (R²={r2:.3f}, n={len(d)})")
    print(d[["lap", "sec_fuel", "sec_corr", "gap_median_m", "anteil_nah",
             "compound", "tyre_life"]].round(2).to_string(index=False))

    print(f"\n[2/3] Streckenvergleich ueber {len(STRECKEN)} Strecken "
         f"({FAHRER_JE_STRECKE} Fahrer je Strecke) ...")
    ergebnisse = []
    for gp in STRECKEN:
        s = f1lab.load(SEASON, gp, "R", telemetry=True)
        fahrer = list(s.drivers)[:FAHRER_JE_STRECKE]
        teile = [f1lab.close_following(s, drv, nah_schwelle_m=NAH_SCHWELLE_M)
                for drv in fahrer]
        teile = [t for t in teile if not t.empty]
        gesamt = pd.concat(teile, ignore_index=True) if teile else pd.DataFrame()
        sl, ic, rr, dd = f1lab.dirty_air_effect(gesamt)
        ergebnisse.append({"strecke": s.event["EventName"], "slope": sl,
                           "r2": rr, "n": len(dd)})
        print(f"      {s.event['EventName']:<28} {sl:+.4f} s/%  "
             f"R²={rr:.3f}  n={len(dd)}")
    ergebnisse = pd.DataFrame(ergebnisse)

    print("\n[3/3] Grafik ...")
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    zeichne_regression(ax[0], d, slope, inter, r2, f"{DRIVER} - {EVENT} {SEASON}")
    zeichne_streckenvergleich(ax[1], ergebnisse)
    fig.suptitle("Dirty Air: Abstand zum Vordermann und Rundenzeit-Effekt",
                x=0.09, ha="left", fontsize=16, color=FG, y=1.02)
    plt.tight_layout()
    path = OUT / "dirty_air.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"\n      -> {path}")


if __name__ == "__main__":
    main()
