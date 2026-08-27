"""prueft ob der erwartete sieger (startplatz 1) im regen seltener gewinnt und ob das feld staerker durcheinandergewirbelt wird"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")                      # schreibt nur dateien ohne fenster

import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import fisher_exact, mannwhitneyu

import f1lab
from f1lab.design import FG, GRID, MUTED, SERIEN, matplotlib_stil

warnings.filterwarnings("ignore")

OUT = Path(__file__).parent / "out"
OUT.mkdir(exist_ok=True)

ERSTE_SAISON = 2018   # davor ist Wetterdaten-Cache duenn (siehe P33)
LETZTE_SAISON = 2026
NASS_SCHWELLE = 0.20   # anteil der wetter-messpunkte mit Rainfall=True

plt.rcParams.update(matplotlib_stil())


def saison_scan() -> pd.DataFrame:
    inv = f1lab.cached_sessions()
    zeilen = []
    for saison in range(ERSTE_SAISON, LETZTE_SAISON + 1):
        rennen = sorted(inv[(inv["season"] == saison) & (inv["ident"] == "R")]
                        ["event"].unique())
        for gp in rennen:
            try:
                ses = f1lab.load(saison, gp, "R", telemetry=False, weather=True)
                w = ses.weather_data
                r = ses.results
            except Exception:
                continue
            if w is None or w.empty or r is None or r.empty:
                continue
            nass_anteil = float(w["Rainfall"].mean())
            gueltig = r.dropna(subset=["GridPosition", "Position"])
            if gueltig.empty:
                continue
            pole = gueltig[gueltig["GridPosition"] == 1]
            pole_gewinnt = (bool((pole["Position"] == 1).iloc[0])
                            if not pole.empty else None)
            durcheinander = float(
                (gueltig["GridPosition"] - gueltig["Position"]).abs().mean())
            zeilen.append({
                "season": saison, "gp": gp, "nass_anteil": nass_anteil,
                "nass": nass_anteil >= NASS_SCHWELLE,
                "pole_gewinnt": pole_gewinnt, "durcheinander": durcheinander,
            })
    return pd.DataFrame(zeilen)


def zeichne_polequote(ax, daten: pd.DataFrame) -> None:
    d = daten.dropna(subset=["pole_gewinnt"])
    quote = d.groupby("nass")["pole_gewinnt"].agg(["mean", "count"])
    quote = quote.reindex([False, True])
    labels = ["Trocken", "Nass"]
    farben = [MUTED, SERIEN[1]]
    ax.bar(labels, quote["mean"] * 100, color=farben, width=0.55)
    for i, (_lbl, row) in enumerate(quote.iterrows()):
        ax.text(i, row["mean"] * 100 + 1.5, f"n={int(row['count'])}",
               ha="center", color=MUTED, fontsize=9)
    ax.set_ylabel("Pole -> Sieg [%]")
    ax.set_ylim(0, max(quote["mean"].max() * 100 + 12, 50))
    ax.set_title("Pole-to-Win: trocken gegen nass", loc="left", color=FG,
                fontsize=13, pad=10)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="y", alpha=0.3, linewidth=0.8, color=GRID)
    ax.set_axisbelow(True)


def zeichne_durcheinander(ax, daten: pd.DataFrame) -> None:
    nass = daten.loc[daten["nass"], "durcheinander"]
    trocken = daten.loc[~daten["nass"], "durcheinander"]
    bp = ax.boxplot([trocken, nass], tick_labels=["Trocken", "Nass"],
                    patch_artist=True, widths=0.5, showmeans=True)
    for patch, farbe in zip(bp["boxes"], [MUTED, SERIEN[1]]):
        patch.set_facecolor(farbe)
        patch.set_alpha(0.6)
    ax.set_ylabel("Mittlere |Start - Ziel| Positionsaenderung")
    ax.set_title("Wie stark wird das Feld durcheinandergewirbelt?",
                loc="left", color=FG, fontsize=13, pad=10)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="y", alpha=0.3, linewidth=0.8, color=GRID)
    ax.set_axisbelow(True)


def main():
    f1lab.enable_cache()

    print(f"[1/3] Saisonverlauf {ERSTE_SAISON}-{LETZTE_SAISON} laden "
         "(Wetterdaten + Ergebnisse je Rennen, VORGEHEN 1) ...")
    daten = saison_scan()
    print(f"      {len(daten)} Rennen, davon {int(daten['nass'].sum())} nass "
         f"(Rainfall-Anteil >= {NASS_SCHWELLE:.0%})")

    print("\n[2/3] Pole-to-Win: trocken gegen nass (VORGEHEN 2) ...")
    d = daten.dropna(subset=["pole_gewinnt"])
    for nass, label in ((False, "Trocken"), (True, "Nass")):
        teil = d[d["nass"] == nass]
        siege = int(teil["pole_gewinnt"].sum())
        print(f"      {label}: {siege}/{len(teil)} = "
             f"{siege / len(teil):.1%}" if len(teil) else f"      {label}: n=0")

    tabelle = pd.crosstab(d["nass"], d["pole_gewinnt"])
    odds, p_fisher = fisher_exact(tabelle)
    print(f"      Fisher-Exact-Test (2x2, Pole gewinnt x nass/trocken): "
         f"p={p_fisher:.3f}")

    print("\n[3/3] Positions-Durcheinander: trocken gegen nass (VORGEHEN 3, "
         "AUSBAUSTUFE) ...")
    nass_d = daten.loc[daten["nass"], "durcheinander"]
    trocken_d = daten.loc[~daten["nass"], "durcheinander"]
    print(f"      Trocken: Median {trocken_d.median():.2f} Plaetze  |  "
         f"Nass: Median {nass_d.median():.2f} Plaetze")
    u_stat, p_mwu = mannwhitneyu(nass_d, trocken_d, alternative="greater")
    print(f"      Mann-Whitney-U (nass > trocken): p={p_mwu:.3f}")

    print("\nGrafik ...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5.5))
    zeichne_polequote(ax1, daten)
    zeichne_durcheinander(ax2, daten)
    fig.suptitle(f"Regen-Variance, {ERSTE_SAISON}-{LETZTE_SAISON}", x=0.06,
                ha="left", fontsize=16, color=FG, y=1.02)
    plt.tight_layout()
    path = OUT / "regen_variance.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"      -> {path}")


if __name__ == "__main__":
    main()
