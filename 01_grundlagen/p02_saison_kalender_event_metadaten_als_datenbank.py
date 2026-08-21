"""baut event- und streckengeometrie-tabellen für den saison-kalender"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

# python legt beim skriptstart nur den ordner des skripts auf den pfad.
# das repo-wurzelverzeichnis fehlt dadurch und "import f1lab" scheitert hier.
# diese zwei zeilen fixen das fuer direkt gestartete skripte in unterordnern.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")                      # nur dateien statt fenster

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import FancyBboxPatch

import f1lab
from f1lab.design import (
    BG,
    FG,
    KONVENTIONELL,
    MUTED,
    RAMPE,
    SPRINT,
    matplotlib_stil,
)

warnings.filterwarnings("ignore")

OUT = Path(__file__).parent / "out"
OUT.mkdir(exist_ok=True)

SEASONS = range(2018, 2027)
GEOMETRIE_SAISON = 2024            # saison mit telemetrie im cache

# sequenzielle rampe fuer hoehenmeter. ein farbton von dunkel bis hell.
# dieselbe rampe wie in f1lab.design aber als colormap-objekt statt liste.
# matplotlib braucht die interpolation fuer scatter/colorbar.
HOEHE = LinearSegmentedColormap.from_list("hoehe", RAMPE)

plt.rcParams.update(matplotlib_stil())


def kalender_grid(ax, dim):
    """ein feld je rennwochenende. zeile = saison, spalte = runde."""
    seasons = sorted(dim["season"].unique())
    y_of = {s: i for i, s in enumerate(seasons)}
    pad, size = 0.06, 0.88          # abstand zwischen den feldern

    for row in dim.itertuples():
        ax.add_patch(FancyBboxPatch(
            (row.round - size / 2, y_of[row.season] - size / 2), size, size,
            boxstyle="round,pad=0,rounding_size=0.22",
            linewidth=0, facecolor=SPRINT if row.is_sprint else KONVENTIONELL))

    for s in seasons:                # anzahl rennen rechts neben der zeile
        n = int((dim["season"] == s).sum())
        ax.text(dim["round"].max() + 1.1, y_of[s], str(n),
                va="center", ha="left", color=MUTED, fontsize=10)

    ax.set_xlim(0.2, dim["round"].max() + 2.4)
    ax.set_ylim(len(seasons) - 0.5 + pad, -0.5 - pad)   # 2018 oben in der grafik
    ax.set_yticks(range(len(seasons)), [str(s) for s in seasons])
    ax.set_xticks([1, 5, 10, 15, 20, 24])
    ax.set_xlabel("Runde")
    ax.tick_params(length=0)
    for side in ax.spines.values():
        side.set_visible(False)
    ax.set_title("Rennkalender 2018-2026", loc="left", color=FG,
                 fontsize=13, pad=12)
    ax.text(dim["round"].max() + 1.1, -1.15, "Rennen", color=MUTED,
            fontsize=9, ha="left", va="center")

    ax.scatter([], [], marker="s", s=70, color=KONVENTIONELL,
               label="konventionell")
    ax.scatter([], [], marker="s", s=70, color=SPRINT, label="Sprint")
    ax.legend(loc="upper left", bbox_to_anchor=(0, -0.13), ncol=2,
              frameon=False, labelcolor=FG, handletextpad=0.4, columnspacing=1.4)


def strecken_profil(ax, geo):
    """streckenlaenge gegen kurvenzahl, eingefaerbt nach hoehenspanne.
    hoehe ist eine groesse und keine kategorie. deshalb eine sequenzielle
    rampe statt weiterer kategoriefarben."""
    d = geo.dropna(subset=["length_m", "corners"]).copy()
    if d.empty:
        ax.text(0.5, 0.5, "keine Telemetrie im Cache", ha="center",
                va="center", color=MUTED, transform=ax.transAxes)
        ax.set_axis_off()
        return

    d["km"] = d["length_m"] / 1000
    span = d["elev_span_m"].astype(float)
    norm = Normalize(vmin=float(span.min()), vmax=float(span.max()))
    ax.scatter(d["km"], d["corners"], s=190, c=span, cmap=HOEHE,
               norm=norm, edgecolor=BG, linewidth=2, zorder=3)

    # nur die extreme in laenge, kurvenzahl und hoehe beschriften.
    # eine zahl an jedem punkt waere rauschen.
    zeigen = (set(d.nlargest(2, "km").index) | set(d.nsmallest(2, "km").index)
              | set(d.nlargest(1, "elev_span_m").index)
              | set(d.nlargest(1, "corners").index)
              | set(d.nsmallest(1, "corners").index))

    x_mitte = (d["km"].min() + d["km"].max()) / 2
    for i in zeigen:
        r = d.loc[i]
        # rechte haelfte nach links beschriften. sonst laeuft der text in die colorbar.
        rechts = r["km"] > x_mitte
        ax.annotate(
            r["circuit"], (r["km"], r["corners"]),
            textcoords="offset points",
            xytext=(-12, 0) if rechts else (12, 0),
            ha="right" if rechts else "left", va="center",
            color=FG, fontsize=9)

    ax.set_xlabel("Streckenlaenge [km]  -  gefahrene Linie")
    ax.set_ylabel("Kurven")
    ax.set_yticks(range(int(d["corners"].min()), int(d["corners"].max()) + 1, 2))
    ax.margins(x=0.12, y=0.10)
    ax.grid(axis="both", alpha=0.35, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_title(f"Streckenprofil {GEOMETRIE_SAISON}", loc="left", color=FG,
                 fontsize=13, pad=12)

    cb = ax.figure.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=HOEHE), ax=ax, pad=0.02)
    cb.set_label("Hoehenspanne [m]", color=MUTED, fontsize=10)
    cb.ax.tick_params(colors=MUTED, length=0)
    cb.outline.set_visible(False)


def main():
    f1lab.enable_cache()

    print("[1/3] Event-Dimension ...")
    dim = f1lab.event_dimension(SEASONS)
    dim.to_parquet(OUT / "dim_events.parquet", index=False)
    print(f"      {len(dim)} Wochenenden, {int(dim['is_sprint'].sum())} Sprints")

    print(f"[2/3] Circuit-Dimension {GEOMETRIE_SAISON} (braucht Telemetrie) ...")
    events = [(GEOMETRIE_SAISON, r) for r in
              dim.loc[dim["season"] == GEOMETRIE_SAISON, "round"]]
    geo = f1lab.circuit_dimension(events)
    geo.to_parquet(OUT / "dim_circuits.parquet", index=False)
    fertig = int(geo["length_m"].notna().sum())
    print(f"      {fertig} von {len(geo)} Strecken vermessen")

    print("[3/3] Grafik ...")
    fig, ax = plt.subplots(2, 1, figsize=(12, 12.5),
                           gridspec_kw={"height_ratios": [1, 1.15], "hspace": 0.42})
    kalender_grid(ax[0], dim)
    strecken_profil(ax[1], geo)
    fig.suptitle("Saison-Kalender und Streckengeometrie", x=0.125, ha="left",
                 fontsize=16, color=FG, y=0.965)
    path = OUT / "kalender_und_strecken.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"      -> {path}")

    print("\nLaengste und kuerzeste Strecken:")
    spalten = ["circuit", "corners", "length_m", "elev_gain_m", "elev_span_m"]
    ok = geo.dropna(subset=["length_m"])
    if not ok.empty:
        print(ok.head(3)[spalten].to_string(index=False))
        print("   ...")
        print(ok.tail(3)[spalten].to_string(index=False))


if __name__ == "__main__":
    main()
