"""prueft ueber einen saison-scan ob ueberholzahl und safety-car-dauer mit der streckengeometrie korrelieren"""
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
from scipy.stats import pearsonr, spearmanr

import f1lab
from f1lab.design import FG, GRID, MUTED, SERIEN, matplotlib_stil

warnings.filterwarnings("ignore")

OUT = Path(__file__).parent / "out"
OUT.mkdir(exist_ok=True)

SEASON = 2024

plt.rcParams.update(matplotlib_stil())


def zeichne_saison(ax, ueberholungen: pd.DataFrame) -> None:
    """zeigt ueberholungen je rennen ueber die saison."""
    u = ueberholungen.sort_values("overtakes")
    farben = [SERIEN[0] if gp != "Monaco Grand Prix" else SERIEN[1]
             for gp in u["gp"]]
    ax.barh(u["gp"], u["overtakes"], color=farben, height=0.65)
    ax.set_xlabel("Ueberholungen (gruene Flagge, ohne Boxenstopp-Effekt)")
    ax.set_title(f"Ueberholungen je Rennen, Saison {SEASON} "
                f"({len(u)} Rennen)", loc="left", color=FG, fontsize=13,
                pad=10)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="x", alpha=0.3, linewidth=0.8, color=GRID)
    ax.set_axisbelow(True)


def zeichne_korrelation(ax, geo: pd.DataFrame, r: float, p: float) -> None:
    """zeigt kurven pro kilometer gegen ueberholungen fuer die strecken mit telemetrie."""
    ax.scatter(geo["kurven_pro_km"], geo["overtakes"], s=70, color=SERIEN[0],
              zorder=3)
    for _, row in geo.iterrows():
        ax.annotate(row["circuit"], (row["kurven_pro_km"], row["overtakes"]),
                   xytext=(6, 4), textcoords="offset points", color=MUTED,
                   fontsize=9)
    steigung, achse0 = np.polyfit(geo["kurven_pro_km"], geo["overtakes"], 1)
    x_fit = np.linspace(geo["kurven_pro_km"].min(), geo["kurven_pro_km"].max(), 50)
    ax.plot(x_fit, steigung * x_fit + achse0, color=MUTED, lw=1.2, ls="--")
    ax.set_xlabel("Kurven pro Kilometer")
    ax.set_ylabel("Ueberholungen")
    ax.set_title(f"Kurvendichte gegen Ueberholungen, {len(geo)} Strecken mit "
                f"Telemetrie (Pearson r={r:.2f}, p={p:.3f})", loc="left",
                color=FG, fontsize=13, pad=10)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(alpha=0.3, linewidth=0.8, color=GRID)
    ax.set_axisbelow(True)


def zeichne_sc_korrelation(ax, geo: pd.DataFrame, r: float, p: float) -> None:
    """zeigt kurven pro kilometer gegen safety-car-/vsc-dauer fuer dieselben strecken."""
    ax.scatter(geo["kurven_pro_km"], geo["sc_dauer_s"], s=70, color=SERIEN[2],
              zorder=3)
    for _, row in geo.iterrows():
        ax.annotate(row["circuit"], (row["kurven_pro_km"], row["sc_dauer_s"]),
                   xytext=(6, 4), textcoords="offset points", color=MUTED,
                   fontsize=9)
    ax.set_xlabel("Kurven pro Kilometer")
    ax.set_ylabel("Safety-Car-/VSC-Dauer [s]")
    ax.set_title(f"ZWEITE AUSBAUSTUFE: Kurvendichte gegen SC-Dauer "
                f"(Pearson r={r:+.2f}, p={p:.2f}) - keine Streckeneigenschaft",
                loc="left", color=FG, fontsize=13, pad=10)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(alpha=0.3, linewidth=0.8, color=GRID)
    ax.set_axisbelow(True)


def main():
    f1lab.enable_cache()

    print(f"[1/3] Ueberholungen je Rennen, Saison {SEASON} (VORGEHEN 1, keine "
         "Telemetrie noetig) ...")
    inv = f1lab.cached_sessions()
    rennen = sorted(inv[(inv["season"] == SEASON) & (inv["ident"] == "R")]
                    ["event"].unique())
    zeilen = []
    for gp in rennen:
        try:
            ses = f1lab.load(SEASON, gp, "R", telemetry=False)
        except Exception:
            continue
        n = int(f1lab.overtakes_matrix(ses).values.sum())
        zeilen.append({"gp": gp, "overtakes": n})
    ueberholungen = pd.DataFrame(zeilen)
    print(f"      {len(ueberholungen)} Rennen, "
         f"{ueberholungen['overtakes'].min()}-{ueberholungen['overtakes'].max()} "
         "Ueberholungen (Min-Max)")

    print("\n[2/3] Streckengeometrie fuer die telemetriefaehigen Rennen "
         "(VORGEHEN 2-3) ...")
    tel_rennen = sorted(inv[(inv["season"] == SEASON) & (inv["ident"] == "R")
                            & inv["telemetry"]]["event"].unique())
    geo = f1lab.circuit_dimension([(SEASON, gp) for gp in tel_rennen])
    geo = geo.merge(ueberholungen, on="gp", how="inner")
    geo["kurven_pro_km"] = geo["corners"] / (geo["length_m"] / 1000)
    print(geo[["gp", "circuit", "corners", "length_m", "kurven_pro_km",
              "overtakes"]].to_string(index=False))

    print(f"\n[3/3] Korrelationen ueber {len(geo)} Strecken (VORGEHEN 4, "
         "AUSBAUSTUFE) ...")
    r_len, p_len = pearsonr(geo["length_m"], geo["overtakes"])
    r_corn, p_corn = pearsonr(geo["corners"], geo["overtakes"])
    r_kpk, p_kpk = pearsonr(geo["kurven_pro_km"], geo["overtakes"])
    print(f"      Laenge:        r={r_len:+.3f}  p={p_len:.3f}")
    print(f"      Kurvenzahl:    r={r_corn:+.3f}  p={p_corn:.3f}")
    print(f"      Kurven/km:     r={r_kpk:+.3f}  p={p_kpk:.3f}")

    ohne_monaco = geo[geo["circuit"] != "Monaco"]
    r_om, p_om = pearsonr(ohne_monaco["kurven_pro_km"], ohne_monaco["overtakes"])
    r_sp, p_sp = spearmanr(geo["kurven_pro_km"], geo["overtakes"])
    print(f"\n      AUSBAUSTUFE ohne Monaco (n={len(ohne_monaco)}): "
         f"r={r_om:+.3f}  p={p_om:.3f}")
    print(f"      AUSBAUSTUFE Spearman (n={len(geo)}):        "
         f"r={r_sp:+.3f}  p={p_sp:.3f}")

    print("\nZWEITE AUSBAUSTUFE: Safety-Car-/VSC-Dauer gegen dieselbe "
         "Streckengeometrie ...")
    sc_zeilen = []
    for gp in rennen:
        ses = f1lab.load(SEASON, gp, "R", telemetry=False)
        ph = f1lab.track_status_phases(ses)
        neutral = ph[ph["label"].isin(["safety car", "vsc"])]
        sc_zeilen.append({"gp": gp, "sc_phasen": len(neutral),
                          "sc_dauer_s": round(float(neutral["duration_s"].sum()), 0)})
    sc = pd.DataFrame(sc_zeilen)
    print(f"      {(sc['sc_phasen'] > 0).sum()}/{len(sc)} Rennen mit "
         "SC/VSC")
    geo = geo.merge(sc, on="gp", how="inner")
    r_sc, p_sc = pearsonr(geo["kurven_pro_km"], geo["sc_dauer_s"])
    r_sc_sp, p_sc_sp = spearmanr(geo["kurven_pro_km"], geo["sc_dauer_s"])
    print(f"      Kurven/km gegen SC-Dauer (n={len(geo)}): "
         f"Pearson r={r_sc:+.3f} p={p_sc:.3f}, "
         f"Spearman r={r_sc_sp:+.3f} p={p_sc_sp:.3f}")

    print("\nGrafik ...")
    fig, ax = plt.subplots(3, 1, figsize=(13, 17))
    zeichne_saison(ax[0], ueberholungen)
    zeichne_korrelation(ax[1], geo, r_kpk, p_kpk)
    zeichne_sc_korrelation(ax[2], geo, r_sc, p_sc)
    fig.suptitle("Ueberholschwierigkeit je Strecke: Saison-Scan", x=0.09,
                ha="left", fontsize=16, color=FG, y=0.995)
    plt.tight_layout()
    path = OUT / "ueberholschwierigkeit_saison_scan.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"\n      -> {path}")


if __name__ == "__main__":
    main()
