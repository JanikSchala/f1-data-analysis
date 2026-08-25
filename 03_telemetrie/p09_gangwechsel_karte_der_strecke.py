"""malt die ideallinie als 2x2-karte eingefaerbt nach gang speed throttle und drs"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")                      # schreibt nur dateien ohne fenster

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import (
    BoundaryNorm,
    LinearSegmentedColormap,
    ListedColormap,
    Normalize,
)

import f1lab
from f1lab.design import FG, MUTED, POSITIV, RAMPE, SERIEN, matplotlib_stil

warnings.filterwarnings("ignore")

OUT = Path(__file__).parent / "out"
OUT.mkdir(exist_ok=True)

SEASON, EVENT, IDENT = 2024, "Monza", "Q"

plt.rcParams.update(matplotlib_stil())


def zeichne_karte(ax, x: np.ndarray, y: np.ndarray, werte: np.ndarray, *,
                  cmap, norm, titel: str, cbar_label: str,
                  cbar_ticks=None, boundaries=None,
                  cbar_ticklabels=None) -> None:
    """zeichnet eine nach werten eingefaerbte streckenkarte. wird fuer alle vier kanaele wiederverwendet."""
    punkte = np.column_stack([x, y]).reshape(-1, 1, 2)
    segmente = np.concatenate([punkte[:-1], punkte[1:]], axis=1)
    lc = LineCollection(list(segmente), cmap=cmap, norm=norm, linewidths=4)
    lc.set_array(werte[:-1])
    ax.add_collection(lc)

    pad = 0.05 * max(x.max() - x.min(), y.max() - y.min())
    ax.set_xlim(x.min() - pad, x.max() + pad)
    ax.set_ylim(y.min() - pad, y.max() + pad)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(titel, loc="left", color=FG, fontsize=13, pad=8)

    cb = ax.figure.colorbar(lc, ax=ax, boundaries=boundaries, fraction=0.045,
                            pad=0.02)
    if cbar_ticks is not None:
        cb.set_ticks(cbar_ticks)
    if cbar_ticklabels is not None:
        cb.set_ticklabels(cbar_ticklabels)
    cb.set_label(cbar_label, color=MUTED, fontsize=9)
    cb.ax.tick_params(colors=MUTED, length=0)
    cb.outline.set_visible(False)


def main():
    f1lab.enable_cache()

    print(f"[1/2] {EVENT} {SEASON} {IDENT} laden (mit Telemetrie) ...")
    ses = f1lab.load(SEASON, EVENT, IDENT, telemetry=True)
    lap = ses.laps.pick_fastest()
    tel = lap.get_telemetry()
    print(f"      Schnellste Runde: {lap['Driver']}, {lap['LapTime']}")

    x = tel["X"].to_numpy(dtype=float)
    y = tel["Y"].to_numpy(dtype=float)
    gear = tel["nGear"].to_numpy(dtype=float)
    speed = tel["Speed"].to_numpy(dtype=float)
    throttle = tel["Throttle"].to_numpy(dtype=float)
    drs = f1lab.drs_state(tel["DRS"].to_numpy())

    offen_pct = 100 * (drs == 2).mean()
    print(f"      DRS offen auf {offen_pct:.0f}% der Runde "
         f"(erkannt/im Fenster: {100 * (drs == 1).mean():.0f}%)")

    print("[2/2] Grafik ...")
    fig, ax = plt.subplots(2, 2, figsize=(14, 13))

    n_gaenge = int(gear.max() - gear.min() + 1)
    gang_cmap = plt.get_cmap("viridis", n_gaenge)
    zeichne_karte(
        ax[0, 0], x, y, gear, cmap=gang_cmap,
        norm=BoundaryNorm(np.arange(gear.min() - 0.5, gear.max() + 1.5), n_gaenge),
        titel="Gang", cbar_label="Gang",
        cbar_ticks=np.arange(gear.min(), gear.max() + 1))

    rampe_cmap = LinearSegmentedColormap.from_list("rampe", RAMPE)
    zeichne_karte(
        ax[0, 1], x, y, speed, cmap=rampe_cmap,
        norm=Normalize(speed.min(), speed.max()),
        titel="Speed", cbar_label="km/h")

    zeichne_karte(
        ax[1, 0], x, y, throttle, cmap=rampe_cmap,
        norm=Normalize(0, 100),
        titel="Throttle", cbar_label="%")

    drs_cmap = ListedColormap([MUTED, SERIEN[1], POSITIV])
    zeichne_karte(
        ax[1, 1], x, y, drs.astype(float), cmap=drs_cmap,
        norm=BoundaryNorm([-0.5, 0.5, 1.5, 2.5], drs_cmap.N),
        titel="DRS", cbar_label="", cbar_ticks=[0, 1, 2],
        cbar_ticklabels=["zu", "erkannt", "offen"])

    fig.suptitle(f"{ses.event['EventName']} {SEASON} {IDENT} - "
                f"{lap['Driver']}, schnellste Runde", x=0.08, ha="left",
                fontsize=16, color=FG, y=0.99)
    plt.tight_layout()
    path = OUT / "gangwechsel_karte.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"\n      -> {path}")


if __name__ == "__main__":
    main()
