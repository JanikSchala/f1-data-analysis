"""legt speed, throttle, brake, gang und drs zweier fahrer ueber die schnellste runde uebereinander und markiert bremspunkte"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")                      # schreibt nur dateien ohne fenster

import fastf1.plotting as f1plt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from fastf1.utils import delta_time

import f1lab
from f1lab.design import FG, GRID, MUTED, matplotlib_stil

warnings.filterwarnings("ignore")

OUT = Path(__file__).parent / "out"
OUT.mkdir(exist_ok=True)

SEASON, EVENT, IDENT = 2024, "Japan", "Q"
D1, D2 = "VER", "NOR"
ZONEN_TOLERANZ_M = 150.0

plt.rcParams.update(matplotlib_stil())
f1plt.setup_mpl(mpl_timedelta_support=False, color_scheme="fastf1")


def sektor_check(lap1, lap2, ref: pd.DataFrame, delta: np.ndarray) -> pd.DataFrame:
    """verifiziert delta_time() an den sektorgrenzen gegen die echten sektorzeiten.
    fastf1s doku empfiehlt genau diese gegenprobe."""
    s1 = [lap1[c].total_seconds() for c in
         ("Sector1Time", "Sector2Time", "Sector3Time")]
    s2 = [lap2[c].total_seconds() for c in
         ("Sector1Time", "Sector2Time", "Sector3Time")]
    ref_sorted = ref.sort_values("SessionTime")

    zeilen = []
    kumuliert = 0.0
    for i, (a, b) in enumerate(zip(s1[:2], s2[:2], strict=True), start=1):
        kumuliert += b - a
        grenze_zeit = lap1["LapStartTime"] + sum(
            (lap1[c] for c in ("Sector1Time", "Sector2Time")[:i]), pd.Timedelta(0))
        idx = (ref_sorted["SessionTime"] - grenze_zeit).abs().idxmin()
        distanz = ref_sorted.loc[idx, "Distance"]
        geschaetzt = float(np.interp(distanz, ref["Distance"], delta))
        zeilen.append({"sektorgrenze": i, "distanz_m": round(float(distanz), 1),
                       "delta_sektorzeit_s": round(kumuliert, 3),
                       "delta_time_s": round(geschaetzt, 3),
                       "abweichung_s": round(geschaetzt - kumuliert, 3)})
    return pd.DataFrame(zeilen)


def main():
    f1lab.enable_cache()

    print(f"[1/4] {EVENT} {SEASON} {IDENT} laden (mit Telemetrie) ...")
    ses = f1lab.load(SEASON, EVENT, IDENT, telemetry=True)
    lap1 = ses.laps.pick_drivers(D1).pick_fastest()
    lap2 = ses.laps.pick_drivers(D2).pick_fastest()
    print(f"      {D1} {lap1['LapTime']}  |  {D2} {lap2['LapTime']}")

    t1 = lap1.get_car_data().add_distance()
    t2 = lap2.get_car_data().add_distance()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        delta, ref, _cmp = delta_time(lap1, lap2)

    print("[2/4] Bremszonen und Sektor-Gegenprobe ...")
    z1 = f1lab.driver_braking_zones(ses, D1)
    z2 = f1lab.driver_braking_zones(ses, D2)
    zonen = f1lab.compare_braking_zones(z1, z2, tolerance_m=ZONEN_TOLERANZ_M)
    zonen["spaeter"] = np.where(zonen["delta_m"] > 0, D2, D1)
    print(f"      {D1}: {len(z1)} Zonen, {D2}: {len(z2)} Zonen, "
         f"{len(zonen)} gepaart")
    print(zonen.round(1).to_string(index=False))

    check = sektor_check(lap1, lap2, ref, delta.to_numpy())
    print("\nGegenprobe delta_time() vs. Sektorzeiten:")
    print(check.to_string(index=False))

    print("\n[3/4] ZWEITE AUSBAUSTUFE: Telemetry-Source-Qualitaet ...")
    for drv, lap in ((D1, lap1), (D2, lap2)):
        q = f1lab.telemetry_source_quality(lap.get_telemetry()["Source"])
        print(f"      {drv}: n={q['n']}  car={q['car']:.1%}  "
             f"pos={q['pos']:.1%}  interpolation={q['interpolation']:.1%}")

    print("\n[4/4] Grafik ...")
    s1 = f1plt.get_driver_style(D1, ["color", "linestyle"], session=ses)
    s2 = f1plt.get_driver_style(D2, ["color", "linestyle"], session=ses)

    fig, ax = plt.subplots(5, 1, figsize=(13, 13), sharex=True,
                           gridspec_kw={"height_ratios": [3, 1.6, 1.6, 1, 1.2],
                                       "hspace": 0.12})

    ax[0].plot(t1["Distance"], t1["Speed"], label=D1, lw=1.8, **s1)
    ax[0].plot(t2["Distance"], t2["Speed"], label=D2, lw=1.8, **s2)
    for _, z in z1.iterrows():
        ax[0].axvline(z["start_m"], color=s1["color"], lw=1, ls=":", alpha=0.6)
    for _, z in z2.iterrows():
        ax[0].axvline(z["start_m"], color=s2["color"], lw=1, ls=":", alpha=0.6)
    for _, z in zonen.iterrows():
        if abs(z["delta_m"]) < 3:
            continue
        x = max(z["start_m_a"], z["start_m_b"])
        ax[0].annotate(f"{z['spaeter']} +{abs(z['delta_m']):.0f} m",
                       xy=(x, 0.03), xycoords=("data", "axes fraction"),
                       rotation=90, va="bottom", ha="right", color=MUTED,
                       fontsize=7.5)
    ax[0].set_ylabel("Speed [km/h]")
    ax[0].legend(loc="lower center", ncol=2, frameon=False, labelcolor=FG)
    ax[0].set_title(f"{ses.event['EventName']} {SEASON} {IDENT} - {D1} vs "
                    f"{D2}  (gepunktet: Bremspunkte)", loc="left", color=FG,
                    fontsize=13, pad=10)

    ax[1].plot(ref["Distance"], delta, color=FG, lw=1.4)
    ax[1].axhline(0, color=MUTED, lw=0.8)
    ax[1].scatter(check["distanz_m"], check["delta_sektorzeit_s"], marker="D",
                  s=36, color="none", edgecolor=FG, linewidth=1.3, zorder=3,
                  label="Sektorzeit (Gegenprobe)")
    ax[1].set_ylabel(f"Delta {D2}-{D1} [s]")
    ax[1].legend(loc="upper left", frameon=False, labelcolor=FG, fontsize=8.5)

    ax[2].plot(t1["Distance"], t1["Throttle"], lw=1.4, **s1)
    ax[2].plot(t2["Distance"], t2["Throttle"], lw=1.4, **s2)
    ax[2].set_ylabel("Throttle [%]")

    ax[3].plot(t1["Distance"], t1["Brake"].astype(int), lw=1.4, **s1)
    ax[3].plot(t2["Distance"], t2["Brake"].astype(int), lw=1.4, **s2)
    ax[3].set_ylabel("Brake")
    ax[3].set_yticks([0, 1])

    ax[4].plot(t1["Distance"], t1["nGear"], lw=1.4, **s1)
    ax[4].plot(t2["Distance"], t2["nGear"], lw=1.4, **s2)
    ax[4].set_ylabel("Gang")
    ax[4].set_xlabel("Distanz [m]")

    for a in ax:
        a.grid(alpha=0.3, linewidth=0.8, color=GRID)
        a.set_axisbelow(True)
        for side in ("top", "right"):
            a.spines[side].set_visible(False)

    path = OUT / "telemetrie_overlay.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"\n      -> {path}")


if __name__ == "__main__":
    main()
