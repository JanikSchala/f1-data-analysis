"""baut eine heatmap der apex-speed je fahrer und kurve und clustert fahrer per k-means nach ihrem kurvenprofil"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")                      # schreibt nur dateien ohne fenster

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from sklearn.cluster import KMeans

import f1lab
from f1lab.design import FG, GRID, MUTED, RAMPE, SERIEN, matplotlib_stil

warnings.filterwarnings("ignore")

OUT = Path(__file__).parent / "out"
OUT.mkdir(exist_ok=True)

SEASON, EVENT, IDENT = 2024, "Hungary", "Q"
RAMPE_CMAP = LinearSegmentedColormap.from_list("rampe", RAMPE)

plt.rcParams.update(matplotlib_stil())


def zeichne_heatmap(ax, rel) -> None:
    """zeigt das apex-speed-delta zum kurvenbesten. faerbung ist auf dem 10. perzentil gekappt statt auf dem minimum.
    eine einzelne unruhige kurve wuerde sonst allein die farbskala strecken."""
    vmin = float(np.percentile(rel.to_numpy(), 10))
    im = ax.imshow(rel.to_numpy(), cmap=RAMPE_CMAP, aspect="auto",
                   vmin=vmin, vmax=0)
    ax.set_xticks(range(len(rel.columns)), rel.columns, rotation=90, fontsize=8)
    ax.set_yticks(range(len(rel.index)), rel.index, fontsize=9)
    ax.set_title("Apex-Speed, Delta zum Kurvenbesten", loc="left", color=FG,
                fontsize=13, pad=12)
    for side in ax.spines.values():
        side.set_visible(False)
    cb = ax.figure.colorbar(im, ax=ax, pad=0.015)
    cb.set_label("Delta [km/h]", color=MUTED, fontsize=9)
    cb.ax.tick_params(colors=MUTED, length=0)
    cb.outline.set_visible(False)


def zeichne_cluster(ax, profil, labels: np.ndarray) -> None:
    """zeigt die mittlere abweichung vom eigenen kurvenschnitt eingefaerbt nach cluster.
    das ist der wert der die cluster tatsaechlich trennt."""
    streuung = profil.abs().mean(axis=1).sort_values()
    farbe = {0: SERIEN[0], 1: SERIEN[1]}
    labels_je_fahrer = dict(zip(profil.index, labels, strict=True))
    farben = [farbe[labels_je_fahrer[f]] for f in streuung.index]

    ax.barh(streuung.index, streuung.to_numpy(), color=farben, height=0.65)
    ax.set_xlabel("Mittlere |Abweichung| vom eigenen Kurvenschnitt [km/h]")
    ax.set_title("k-Means (k=2): Cluster trennt nach Streuung, nicht Kurventyp",
                loc="left", color=FG, fontsize=13, pad=12)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="x", alpha=0.35, linewidth=0.8, color=GRID)
    ax.set_axisbelow(True)

    for lbl, name in ((0, "Cluster A"), (1, "Cluster B")):
        ax.scatter([], [], color=farbe[lbl], s=80,
                  label=f"{name} (n={int((labels == lbl).sum())})")
    ax.legend(loc="lower right", frameon=False, labelcolor=FG, fontsize=9)


def main():
    f1lab.enable_cache()

    print(f"[1/3] {EVENT} {SEASON} {IDENT} laden (mit Telemetrie) ...")
    ses = f1lab.load(SEASON, EVENT, IDENT, telemetry=True)

    print("[2/3] Kurvengeschwindigkeiten (f1lab.corner_speeds) ...")
    mat = f1lab.corner_speeds(ses)
    rel = mat.sub(mat.max(axis=0), axis=1)
    order = rel.mean(axis=1).sort_values(ascending=False).index
    rel = rel.loc[order]
    print(f"      {mat.shape[0]} Fahrer x {mat.shape[1]} Kurven, "
         f"{mat.isna().sum().sum()} fehlende Werte")

    print("\n[3/3] k-Means-Clustering (AUSBAUSTUFE) ...")
    median_speed = mat.median(axis=0)
    profil = rel.sub(rel.mean(axis=1), axis=0)
    korrelation = profil.apply(
        lambda row: np.corrcoef(row, median_speed.reindex(row.index))[0, 1],
        axis=1)
    print("      Korrelation Profil x Kurvengeschwindigkeit je Fahrer "
         "(negativ = staerker in langsamen Kurven):")
    print(f"      {korrelation.min():.2f} bis {korrelation.max():.2f}, "
         f"Median {korrelation.median():.2f}")

    labels = KMeans(n_clusters=2, random_state=42, n_init=10).fit_predict(
        profil.to_numpy())
    for lbl in (0, 1):
        idx = profil.index[labels == lbl]
        print(f"      Cluster {lbl}: {list(idx)}")
        print(f"        mittlere |Abweichung|: "
             f"{profil.loc[idx].abs().mean().mean():.2f} km/h")

    print("\nGrafik ...")
    fig, ax = plt.subplots(1, 2, figsize=(16, 8),
                           gridspec_kw={"width_ratios": [1.3, 1]})
    zeichne_heatmap(ax[0], rel)
    zeichne_cluster(ax[1], profil, labels)
    fig.suptitle(f"{ses.event['EventName']} {SEASON} - "
                f"Kurvengeschwindigkeits-Profil", x=0.09, ha="left",
                fontsize=16, color=FG, y=1.02)
    plt.tight_layout()
    path = OUT / "kurvenprofil.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"\n      -> {path}")


if __name__ == "__main__":
    main()
