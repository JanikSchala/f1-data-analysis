"""vergleicht drs-zeitanteil und topspeed-gewinn je fahrer und die drs-zonenlage einer strecke ueber zwei jahre"""
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

SEASON, EVENT, IDENT = 2024, "Italy", "Q"          # monza hat zwei drs-zonen
ZONEN_STRECKE = "Azerbaijan"
ZONEN_JAHRE = (2018, 2024)
MIN_ZONE_M = 100.0

plt.rcParams.update(matplotlib_stil())


def zeichne_ranking(ax, df: pd.DataFrame, spalte: str, xlabel: str,
                    titel: str) -> None:
    d = df.dropna(subset=[spalte]).sort_values(spalte)
    ax.barh(d["driver"], d[spalte], color=SERIEN[0], height=0.65)
    ax.set_xlabel(xlabel)
    ax.set_title(titel, loc="left", color=FG, fontsize=13, pad=10)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="x", alpha=0.35, linewidth=0.8, color=GRID)
    ax.set_axisbelow(True)


def zeichne_zonenvergleich(ax, zonen_alt: pd.DataFrame, zonen_neu: pd.DataFrame,
                           strecke_m: float, jahr_alt: int, jahr_neu: int) -> None:
    """zeigt die zonenlage zweier jahre auf einer gemeinsamen distanzachse."""
    for jahr, zonen, y, farbe in ((jahr_alt, zonen_alt, 1, SERIEN[0]),
                                  (jahr_neu, zonen_neu, 0, SERIEN[1])):
        for _, z in zonen.iterrows():
            ax.barh(y, z["length_m"], left=z["start_m"], height=0.55,
                   color=farbe, edgecolor="none")
            ax.text(z["start_m"], y + 0.34, f"{z['start_m']:.0f} m",
                   ha="left", va="bottom", color=MUTED, fontsize=8)
        ax.text(-40, y, str(jahr), ha="right", va="center", color=FG,
               fontsize=11, fontweight="bold")

    ax.set_xlim(-450, strecke_m * 1.02)
    ax.set_ylim(-0.7, 2.0)
    ax.set_yticks([])
    ax.set_xlabel("Distanz [m]")
    ax.set_title(f"DRS-Zonen {ZONEN_STRECKE}: {jahr_alt} gegen {jahr_neu}",
                loc="left", color=FG, fontsize=13, pad=10)
    for side in ax.spines.values():
        side.set_visible(False)
    ax.grid(axis="x", alpha=0.3, linewidth=0.8, color=GRID)
    ax.set_axisbelow(True)


def main():
    f1lab.enable_cache()

    print(f"[1/3] {EVENT} {SEASON} {IDENT} laden (mit Telemetrie) ...")
    ses = f1lab.load(SEASON, EVENT, IDENT, telemetry=True)
    nutzung = f1lab.drs_usage(ses)
    print(nutzung.round(1).to_string(index=False))

    lap = ses.laps.pick_fastest()
    zonen_monza = f1lab.drs_zones(ses, str(lap["Driver"]), min_length_m=MIN_ZONE_M)
    print(f"\nDRS-Zonen {EVENT} {SEASON} (schnellste Runde {lap['Driver']}):")
    print(zonen_monza.round(0).to_string(index=False))

    print(f"\n[2/3] DRS-Zonen {ZONEN_STRECKE} {ZONEN_JAHRE[0]} vs "
         f"{ZONEN_JAHRE[1]} ...")
    sessions = {jahr: f1lab.load(jahr, ZONEN_STRECKE, "Q", telemetry=True)
               for jahr in ZONEN_JAHRE}
    zonen = {}
    laengen = {}
    for jahr, s in sessions.items():
        beste = s.laps.pick_fastest()
        car = beste.get_car_data().add_distance()
        zonen[jahr] = f1lab.drs_zones(s, str(beste["Driver"]),
                                      min_length_m=MIN_ZONE_M)
        laengen[jahr] = car["Distance"].max()
        print(f"      {jahr}: {len(zonen[jahr])} Zonen, "
             f"Streckenlaenge {laengen[jahr]:.0f} m")
    verschiebung = (zonen[ZONEN_JAHRE[1]]["start_m"].to_numpy()
                   - zonen[ZONEN_JAHRE[0]]["start_m"].to_numpy())
    print(f"      Verschiebung Zonenbeginn: "
         f"{', '.join(f'{v:+.0f} m' for v in verschiebung)}")

    print("\n[3/3] Grafik ...")
    fig = plt.figure(figsize=(15, 11))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.2, 0.8], hspace=0.4, wspace=0.3)
    zeichne_ranking(fig.add_subplot(gs[0, 0]), nutzung, "drs_pct",
                    "DRS offen [% der Runde]", f"DRS-Zeitanteil - {EVENT} {SEASON}")
    zeichne_ranking(fig.add_subplot(gs[0, 1]), nutzung, "gewinn_kmh",
                    "Topspeed-Gewinn durch DRS [km/h]",
                    f"DRS-Wirkung - {EVENT} {SEASON}")
    zeichne_zonenvergleich(fig.add_subplot(gs[1, :]), zonen[ZONEN_JAHRE[0]],
                           zonen[ZONEN_JAHRE[1]],
                           max(laengen.values()), *ZONEN_JAHRE)

    fig.suptitle("DRS-Nutzung, Topspeed-Wirkung und Zonenverschiebung",
                x=0.09, ha="left", fontsize=16, color=FG, y=0.995)
    plt.tight_layout()
    path = OUT / "drs_nutzung.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"\n      -> {path}")


if __name__ == "__main__":
    main()
