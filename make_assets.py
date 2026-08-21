#!/usr/bin/env python3
"""
erzeugt die Grafiken und Kennzahlen fuer das README.

    python make_assets.py

alle Auswertungen laufen ueber f1lab. das ist dasselbe Modul wie in den
Tests. eine fruehere eigene Kopie der Filterlogik in diesem Skript hatte
nach einem Bugfix in f1lab zu abweichenden Ergebnissen gefuehrt. das soll
nicht wieder passieren.

schreibt PNGs nach assets/ und die berechneten Werte nach
assets/kennzahlen.json. der erste Lauf dauert einige Minuten weil die
Sessions heruntergeladen werden. danach kommt alles aus dem Cache.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")                      # kein fenster, nur dateien

import fastf1.plotting as f1plt
import matplotlib.pyplot as plt
import numpy as np
from fastf1.utils import delta_time
from matplotlib.collections import LineCollection

import f1lab

warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent
ASSETS = ROOT / "assets"
ASSETS.mkdir(exist_ok=True)

f1lab.enable_cache()
f1plt.setup_mpl(mpl_timedelta_support=False, color_scheme="fastf1")

DPI = 130
BG = "#15151e"
FG = "#f0f0f0"

plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG,
    "savefig.facecolor": BG, "text.color": FG,
    "axes.labelcolor": FG, "xtick.color": FG, "ytick.color": FG,
    "axes.edgecolor": "#3a3a48", "grid.color": "#2a2a38",
    "font.size": 11,
})

KPI: dict = {}


def save(fig, name: str) -> None:
    path = ASSETS / name
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"    -> {path.relative_to(ROOT)}  ({path.stat().st_size // 1024} KB)")


# ---------------------------------------------------------------- 1
def gear_map(year=2024, gp="Belgium"):
    print(f"[1/5] Gangwechsel-Karte  {gp} {year}")
    ses = f1lab.load(year, gp, "Q", telemetry=True)
    lap = ses.laps.pick_fastest()
    tel = lap.get_telemetry()

    x = tel["X"].to_numpy(float)
    y = tel["Y"].to_numpy(float)
    gear = tel["nGear"].to_numpy(float)

    pts = np.array([x, y]).T.reshape(-1, 1, 2)
    seg = np.concatenate([pts[:-1], pts[1:]], axis=1)

    lc = LineCollection(seg, norm=plt.Normalize(1, 8),
                        cmap=plt.get_cmap("viridis", 8), linewidth=4.5)
    lc.set_array(gear[:-1])

    fig, ax = plt.subplots(figsize=(9, 8))
    ax.add_collection(lc)
    ax.axis("equal")
    ax.axis("off")
    ax.set_title(f"{ses.event['EventName']} {year} - Gangwechsel auf der "
                 f"schnellsten Runde\n{lap['Driver']} - "
                 f"{lap['LapTime'].total_seconds():.3f} s",
                 color=FG, fontsize=13, pad=16)
    cbar = fig.colorbar(lc, ax=ax, boundaries=np.arange(0.5, 9.5), shrink=0.7)
    cbar.set_ticks(np.arange(1, 9))
    cbar.set_label("Gang", color=FG)
    cbar.ax.yaxis.set_tick_params(color=FG)
    plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color=FG)
    save(fig, "gangwechsel.png")

    KPI["gangkarte"] = {
        "event": str(ses.event["EventName"]), "jahr": year,
        "fahrer": str(lap["Driver"]),
        "rundenzeit_s": round(lap["LapTime"].total_seconds(), 3),
        "punkte": int(len(tel)),
        "streckenlaenge_m": round(float(tel.add_distance()["Distance"].max())),
    }


# ---------------------------------------------------------------- 2
def telemetry_overlay(year=2024, gp="Japan", d1="VER", d2="NOR"):
    print(f"[2/5] Telemetrie-Overlay  {gp} {year}  {d1} vs {d2}")
    ses = f1lab.load(year, gp, "Q", telemetry=True)
    lap1 = ses.laps.pick_drivers(d1).pick_fastest()
    lap2 = ses.laps.pick_drivers(d2).pick_fastest()
    t1 = lap1.get_car_data().add_distance()
    t2 = lap2.get_car_data().add_distance()
    dlt, ref, _ = delta_time(lap1, lap2)

    s1 = f1plt.get_driver_style(d1, ["color", "linestyle"], session=ses)
    s2 = f1plt.get_driver_style(d2, ["color", "linestyle"], session=ses)

    fig, ax = plt.subplots(4, 1, figsize=(12, 9), sharex=True,
                           gridspec_kw={"height_ratios": [3, 2, 2, 2]})
    ax[0].plot(t1["Distance"], t1["Speed"], label=d1, lw=1.6, **s1)
    ax[0].plot(t2["Distance"], t2["Speed"], label=d2, lw=1.6, **s2)
    ax[0].set_ylabel("Speed [km/h]")
    ax[0].legend(loc="lower right")

    ax[1].plot(ref["Distance"], dlt, color=FG, lw=1.4)
    ax[1].axhline(0, color="#666", lw=0.8)
    ax[1].fill_between(ref["Distance"], dlt, 0, where=(dlt > 0),
                       color=s1.get("color", "#e10600"), alpha=0.25)
    ax[1].fill_between(ref["Distance"], dlt, 0, where=(dlt < 0),
                       color=s2.get("color", "#00d2be"), alpha=0.25)
    ax[1].set_ylabel(f"Delta {d2}\nzu {d1} [s]")

    ax[2].plot(t1["Distance"], t1["Throttle"], lw=1.2, **s1)
    ax[2].plot(t2["Distance"], t2["Throttle"], lw=1.2, **s2)
    ax[2].set_ylabel("Gas [%]")

    ax[3].plot(t1["Distance"], t1["Brake"].astype(int), lw=1.2, **s1)
    ax[3].plot(t2["Distance"], t2["Brake"].astype(int), lw=1.2, **s2)
    ax[3].set_ylabel("Bremse")
    ax[3].set_yticks([0, 1])
    ax[3].set_xlabel("Distanz [m]")

    for a in ax:
        a.grid(alpha=0.2)
    fig.suptitle(f"{ses.event['EventName']} {year} - Qualifying, "
                 f"schnellste Runde", color=FG, fontsize=13)
    plt.tight_layout()
    save(fig, "telemetrie_overlay.png")

    KPI["overlay"] = {
        "event": str(ses.event["EventName"]), "jahr": year,
        d1: round(lap1["LapTime"].total_seconds(), 3),
        d2: round(lap2["LapTime"].total_seconds(), 3),
        "delta_s": round(lap2["LapTime"].total_seconds()
                         - lap1["LapTime"].total_seconds(), 3),
        "max_delta_s": round(float(np.nanmax(np.abs(dlt))), 3),
        "vmax_kmh": {d1: round(float(t1["Speed"].max()), 1),
                     d2: round(float(t2["Speed"].max()), 1)},
    }


# ---------------------------------------------------------------- 3
def race_pace(year=2024, gp="Spain"):
    print(f"[3/5] Race-Pace-Ranking  {gp} {year}")
    ses = f1lab.load(year, gp, "R")

    # dieselbe funktion wie in den tests
    pace = f1lab.pace_table(ses)
    laps_clean = f1lab.clean_laps(ses)

    fig, ax = plt.subplots(figsize=(10, 8))
    colors = [f1plt.get_team_color(t, session=ses) for t in pace["team"]]
    err = [pace["delta_s"] - pace["ci_lo"], pace["ci_hi"] - pace["delta_s"]]
    ax.barh(pace["driver"], pace["delta_s"], xerr=err, color=colors,
            ecolor="#888", capsize=2.5, height=0.72)
    ax.invert_yaxis()
    ax.set_xlabel("Delta zur besten Race Pace [s pro Runde]")
    ax.grid(axis="x", alpha=0.2)
    ax.set_title(f"{ses.event['EventName']} {year} - bereinigte Race Pace\n"
                 f"Median mit 95%-Bootstrap-Intervall, nur Gruenphasen ohne "
                 f"Boxenrunden", color=FG, fontsize=12)
    plt.tight_layout()
    save(fig, "race_pace.png")

    total = int(len(ses.laps))
    kept = int(len(laps_clean))
    KPI["racepace"] = {
        "event": str(ses.event["EventName"]), "jahr": year,
        "runden_gesamt": total,
        "runden_nach_filter": kept,
        "anteil_verworfen_pct": round(100 * (1 - kept / total), 1),
        "schnellster": str(pace["driver"].iloc[0]),
        "top5": [{"fahrer": r["driver"], "team": r["team"],
                  "delta_s": float(r["delta_s"]), "runden": int(r["laps"]),
                  "ci_breite_s": float(r["ci_width"])}
                 for _, r in pace.head(5).iterrows()],
    }


# ---------------------------------------------------------------- 4
def strategy(year=2024, gp="Hungary"):
    print(f"[4/5] Strategieuebersicht  {gp} {year}")
    ses = f1lab.load(year, gp, "R")
    st = f1lab.stints(ses)
    order = [d for d in ses.results.sort_values("Position")["Abbreviation"]
             if d in st["Driver"].values]

    fig, ax = plt.subplots(figsize=(11, 8.5))
    for drv in order:
        prev = 0
        for _, s in st[st["Driver"] == drv].sort_values("start").iterrows():
            c = f1plt.get_compound_color(s["Compound"], session=ses)
            ax.barh(drv, s["laps"], left=prev, color=c,
                    edgecolor=BG, linewidth=1.2, height=0.72)
            if s["laps"] > 5:
                ax.text(prev + s["laps"] / 2, drv, str(s["Compound"])[0],
                        ha="center", va="center", fontsize=8.5,
                        fontweight="bold", color="#111")
            prev += s["laps"]
    ax.invert_yaxis()
    ax.set_xlabel("Runde")
    ax.grid(axis="x", alpha=0.2)
    ax.set_title(f"{ses.event['EventName']} {year} - Reifenstrategien\n"
                 f"sortiert nach Endposition", color=FG, fontsize=12)
    plt.tight_layout()
    save(fig, "strategie.png")

    KPI["strategie"] = {
        "event": str(ses.event["EventName"]), "jahr": year,
        "stints_gesamt": int(len(st)),
        "stints_pro_fahrer": round(len(st) / max(len(order), 1), 2),
        "laenge_je_compound": {
            str(k): round(float(v), 1) for k, v in
            st.groupby("Compound")["laps"].mean().items()},
    }


# ---------------------------------------------------------------- 5
def degradation(year=2024, gp="Bahrain"):
    print(f"[5/5] Reifendegradation  {gp} {year}")
    ses = f1lab.load(year, gp, "R")

    deg = f1lab.degradation(ses)
    agg = f1lab.degradation_by_compound(ses)

    laps = f1lab.clean_laps(ses, threshold=1.10).copy()
    laps["sec"] = laps["LapTime"].dt.total_seconds()
    laps["corrected"] = f1lab.fuel_correct(
        laps["sec"], laps["LapNumber"], ses.total_laps)

    fig, ax = plt.subplots(1, 2, figsize=(13, 5.5),
                           gridspec_kw={"width_ratios": [3, 2]})

    for (_drv, _st), g in laps.groupby(["Driver", "Stint"]):
        if len(g) < 6:
            continue
        c = f1plt.get_compound_color(g["Compound"].iloc[0], session=ses)
        ax[0].plot(g["TyreLife"], g["corrected"], marker=".", ms=3.5, lw=0.9,
                   alpha=0.55, color=c)
    ax[0].set_xlabel("Reifenalter [Runden]")
    ax[0].set_ylabel("Rundenzeit, fuel-korrigiert [s]")
    ax[0].grid(alpha=0.2)
    ax[0].set_title("Jeder Stint einzeln", color=FG, fontsize=11)

    cols = [f1plt.get_compound_color(c, session=ses) for c in agg.index]
    ax[1].bar(agg.index, agg["mean"], yerr=agg["std"], color=cols,
              ecolor="#888", capsize=4)
    ax[1].set_ylabel("Degradation [s pro Runde]")
    ax[1].grid(axis="y", alpha=0.2)
    ax[1].set_title("Mittelwert je Mischung\n(nur belastbare Fits)",
                    color=FG, fontsize=11)

    fig.suptitle(f"{ses.event['EventName']} {year} - Reifendegradation "
                 f"(Fuel-Effekt herausgerechnet)", color=FG, fontsize=13)
    plt.tight_layout()
    save(fig, "degradation.png")

    KPI["degradation"] = {
        "event": str(ses.event["EventName"]), "jahr": year,
        "stints_gesamt": int(len(deg)),
        "stints_belastbar": int(deg["reliable"].sum()),
        "je_compound": {
            str(c): {"mittel_s_pro_runde": round(float(r["mean"]), 4),
                     "std": round(float(r["std"]), 4),
                     "stints": int(r["stints"])}
            for c, r in agg.iterrows()},
        "annahme_fuel": "1.8 kg pro Runde, 0.03 s pro kg",
    }


# ----------------------------------------------------------------
if __name__ == "__main__":
    print(f"\nErzeuge README-Grafiken mit f1lab {f1lab.__version__}.")
    print("Der erste Lauf dauert einige Minuten.\n")

    for fn in (gear_map, telemetry_overlay, race_pace, strategy, degradation):
        try:
            fn()
        except Exception as exc:
            print(f"    FEHLER in {fn.__name__}: {type(exc).__name__}: {exc}")
            KPI[fn.__name__] = {"fehler": f"{type(exc).__name__}: {exc}"}

    KPI["_meta"] = {
        "erzeugt_mit": f"f1lab {f1lab.__version__}",
        "hinweis": "Alle Werte stammen aus f1lab - denselben Funktionen, "
                   "die von tests/ geprueft werden.",
    }

    out = ASSETS / "kennzahlen.json"
    out.write_text(json.dumps(KPI, indent=2, ensure_ascii=False),
                   encoding="utf-8")
    print(f"\nKennzahlen: {out.relative_to(ROOT)}")
    print(f"Grafiken:   {len(list(ASSETS.glob('*.png')))} PNGs in assets/\n")
