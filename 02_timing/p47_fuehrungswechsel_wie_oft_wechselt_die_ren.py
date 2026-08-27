"""zaehlt echte fuehrungswechsel je rennen einer saison und prueft ob mehr ueberholungen insgesamt auch mehr spannung an der spitze bedeuten"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")                      # schreibt nur dateien ohne fenster

import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import pearsonr, spearmanr

import f1lab
from f1lab.design import FG, GRID, MUTED, SERIEN, matplotlib_stil

warnings.filterwarnings("ignore")

OUT = Path(__file__).parent / "out"
OUT.mkdir(exist_ok=True)

SEASON = 2024

plt.rcParams.update(matplotlib_stil())


def zeichne_ranking(ax, daten: pd.DataFrame) -> None:
    d = daten.sort_values("fuehrungswechsel")
    farben = [SERIEN[0] if v == d["fuehrungswechsel"].max() else MUTED
             for v in d["fuehrungswechsel"]]
    ax.barh(d["gp"], d["fuehrungswechsel"], color=farben, height=0.65)
    ax.set_xlabel("Echte Fuehrungswechsel (P1 -> P1, ohne Boxenstopp-Effekt)")
    ax.set_title(f"Fuehrungswechsel je Rennen, Saison {SEASON} "
                f"({len(d)} Rennen)", loc="left", color=FG, fontsize=13,
                pad=10)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="x", alpha=0.3, linewidth=0.8, color=GRID)
    ax.set_axisbelow(True)


def zeichne_streuung(ax, daten: pd.DataFrame) -> None:
    ax.scatter(daten["overtakes"], daten["fuehrungswechsel"], s=70,
              color=SERIEN[0], zorder=3)
    for _, r in daten.iterrows():
        ax.annotate(r["gp"].replace(" Grand Prix", ""),
                    (r["overtakes"], r["fuehrungswechsel"]),
                    textcoords="offset points", xytext=(6, 4),
                    fontsize=7.5, color=MUTED)
    ax.set_xlabel("Ueberholungen insgesamt")
    ax.set_ylabel("Davon Fuehrungswechsel")
    ax.set_title("Gesamtspannung gegen Spitzenspannung", loc="left",
                color=FG, fontsize=13, pad=10)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(alpha=0.3, linewidth=0.8, color=GRID)
    ax.set_axisbelow(True)


def main():
    f1lab.enable_cache()

    print(f"[1/3] Fuehrungswechsel und Ueberholungen je Rennen, Saison "
         f"{SEASON} (VORGEHEN 1-2, keine Telemetrie noetig) ...")
    inv = f1lab.cached_sessions()
    rennen = sorted(inv[(inv["season"] == SEASON) & (inv["ident"] == "R")]
                    ["event"].unique())
    zeilen = []
    for gp in rennen:
        try:
            ses = f1lab.load(SEASON, gp, "R", telemetry=False)
        except Exception:
            continue
        n_overtakes = int(f1lab.overtakes_matrix(ses).values.sum())
        n_lead = len(f1lab.lead_changes(ses))
        zeilen.append({"gp": gp, "overtakes": n_overtakes,
                       "fuehrungswechsel": n_lead})
    daten = pd.DataFrame(zeilen)
    print(daten.sort_values("fuehrungswechsel", ascending=False)
         .to_string(index=False))

    print(f"\n[2/3] Uebersicht ueber {len(daten)} Rennen (VORGEHEN 3) ...")
    print(f"      Fuehrungswechsel gesamt: {daten['fuehrungswechsel'].sum()}, "
         f"{(daten['fuehrungswechsel'] == 0).sum()} Rennen ganz ohne "
         "Wechsel an der Spitze")
    anteil = daten["fuehrungswechsel"].sum() / daten["overtakes"].sum()
    print(f"      Nur {anteil:.1%} aller Ueberholungen aendern auch die "
         "Rennfuehrung - der Rest passiert weiter hinten im Feld")

    print("\n[3/3] Haengt mehr Ueberholen insgesamt mit mehr Wechseln an "
         "der Spitze zusammen? (VORGEHEN 4, AUSBAUSTUFE) ...")
    r_p, p_p = pearsonr(daten["overtakes"], daten["fuehrungswechsel"])
    r_s, p_s = spearmanr(daten["overtakes"], daten["fuehrungswechsel"])
    print(f"      Pearson:  r={r_p:+.3f}  p={p_p:.3f}")
    print(f"      Spearman: r={r_s:+.3f}  p={p_s:.3f}")

    print("\nGrafik ...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7),
                                   gridspec_kw={"width_ratios": [3, 2]})
    zeichne_ranking(ax1, daten)
    zeichne_streuung(ax2, daten)
    fig.suptitle(f"Fuehrungswechsel, Saison {SEASON}", x=0.06, ha="left",
                fontsize=16, color=FG, y=1.02)
    plt.tight_layout()
    path = OUT / "fuehrungswechsel.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"      -> {path}")


if __name__ == "__main__":
    main()
