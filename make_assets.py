#!/usr/bin/env python3
"""
Erzeugt die Grafiken und Kennzahlen fuer das README.

    python make_assets.py

Schreibt PNGs nach assets/ und die berechneten Werte nach
assets/kennzahlen.json. Der erste Lauf dauert einige Minuten, weil die
Sessions heruntergeladen werden - danach kommt alles aus dem Cache.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")                      # kein Fenster, nur Dateien

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection

import fastf1
import fastf1.plotting as f1plt
from fastf1.utils import delta_time

warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent
ASSETS = ROOT / "assets"
ASSETS.mkdir(exist_ok=True)

fastf1.Cache.enable_cache(str(Path.home() / "f1_cache"))
f1plt.setup_mpl(mpl_timedelta_support=False, color_scheme="fastf1")

DPI = 130
BG = "#15151e"
FG = "#f0f0f0"
ACCENT = "#e10600"

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


def load(year, gp, ident, **kw):
    ses = fastf1.get_session(year, gp, ident)
    ses.load(**kw)
    return ses


def clean(session, threshold=1.07):
    return (session.laps.pick_wo_box().pick_accurate()
            .pick_track_status("1").pick_not_deleted()
            .pick_quicklaps(threshold=threshold))


# ---------------------------------------------------------------- 1
def gear_map(year=2024, gp="Belgium"):
    print(f"[1/5] Gangwechsel-Karte  {gp} {year}")
    ses = load(year, gp, "Q")
    lap = ses.laps.pick_fastest()
    tel = lap.get_telemetry()

    x = tel["X"].to_numpy(float)
    y = tel["Y"].to_numpy(float)
    gear = tel["nGear"].to_numpy(float)

    pts = np.array([x, y]).T.reshape(-1, 1, 2)
    seg = np.concatenate([pts[:-1], pts[1:]], axis=1)

    cmap = plt.get_cmap("viridis", 8)
    lc = LineCollection(seg, norm=plt.Normalize(1, 8), cmap=cmap, linewidth=4.5)
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
    ses = load(year, gp, "Q")
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
    ax[0].set_ylabel("Speed [km/h]"); ax[0].legend(loc="lower right")

    ax[1].plot(ref["Distance"], dlt, color=FG, lw=1.4)
    ax[1].axhline(0, color="#666", lw=0.8)
    ax[1].fill_between(ref["Distance"], dlt, 0, where=(dlt > 0),
                       color=s1.get("color", ACCENT), alpha=0.25)
    ax[1].fill_between(ref["Distance"], dlt, 0, where=(dlt < 0),
                       color=s2.get("color", "#00d2be"), alpha=0.25)
    ax[1].set_ylabel(f"Delta {d2}\nzu {d1} [s]")

    ax[2].plot(t1["Distance"], t1["Throttle"], lw=1.2, **s1)
    ax[2].plot(t2["Distance"], t2["Throttle"], lw=1.2, **s2)
    ax[2].set_ylabel("Gas [%]")

    ax[3].plot(t1["Distance"], t1["Brake"].astype(int), lw=1.2, **s1)
    ax[3].plot(t2["Distance"], t2["Brake"].astype(int), lw=1.2, **s2)
    ax[3].set_ylabel("Bremse"); ax[3].set_yticks([0, 1])
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
    ses = load(year, gp, "R", telemetry=False)
    laps = clean(ses).copy()
    laps["Sec"] = laps["LapTime"].dt.total_seconds()

    rng = np.random.default_rng(42)
    rows = []
    for drv, g in laps.groupby("Driver"):
        v = g["Sec"].to_numpy()
        if len(v) < 8:
            continue
        boot = [np.median(rng.choice(v, len(v), replace=True)) for _ in range(1000)]
        rows.append({"Driver": drv, "Team": g["Team"].iloc[0], "Laps": len(v),
                     "Median": float(np.median(v)),
                     "Lo": float(np.percentile(boot, 2.5)),
                     "Hi": float(np.percentile(boot, 97.5))})

    pace = pd.DataFrame(rows).sort_values("Median").reset_index(drop=True)
    best = pace["Median"].iloc[0]
    for c in ("Median", "Lo", "Hi"):
        pace[c + "D"] = pace[c] - best

    fig, ax = plt.subplots(figsize=(10, 8))
    colors = [f1plt.get_team_color(t, session=ses) for t in pace["Team"]]
    err = [pace["MedianD"] - pace["LoD"], pace["HiD"] - pace["MedianD"]]
    ax.barh(pace["Driver"], pace["MedianD"], xerr=err, color=colors,
            ecolor="#888", capsize=2.5, height=0.72)
    ax.invert_yaxis()
    ax.set_xlabel("Delta zur besten Race Pace [s pro Runde]")
    ax.grid(axis="x", alpha=0.2)
    ax.set_title(f"{ses.event['EventName']} {year} - bereinigte Race Pace\n"
                 f"Median mit 95%-Bootstrap-Intervall, nur Gruenphasen ohne "
                 f"Boxenrunden", color=FG, fontsize=12)
    plt.tight_layout()
    save(fig, "race_pace.png")

    KPI["racepace"] = {
        "event": str(ses.event["EventName"]), "jahr": year,
        "runden_gesamt": int(len(ses.laps)),
        "runden_nach_filter": int(len(laps)),
        "anteil_verworfen_pct": round(100 * (1 - len(laps) / len(ses.laps)), 1),
        "schnellster": str(pace["Driver"].iloc[0]),
        "top5": [{"fahrer": r["Driver"], "team": r["Team"],
                  "delta_s": round(r["MedianD"], 3), "runden": int(r["Laps"])}
                 for _, r in pace.head(5).iterrows()],
    }


# ---------------------------------------------------------------- 4
def strategy(year=2024, gp="Hungary"):
    print(f"[4/5] Strategieuebersicht  {gp} {year}")
    ses = load(year, gp, "R", telemetry=False)
    stints = (ses.laps.groupby(["Driver", "Stint", "Compound"])["LapNumber"]
              .agg(["min", "max", "count"]).reset_index()
              .rename(columns={"count": "Laenge"}))
    order = [d for d in ses.results.sort_values("Position")["Abbreviation"]
             if d in stints["Driver"].values]

    fig, ax = plt.subplots(figsize=(11, 8.5))
    for drv in order:
        prev = 0
        for _, s in stints[stints["Driver"] == drv].sort_values("Stint").iterrows():
            c = f1plt.get_compound_color(s["Compound"], session=ses)
            ax.barh(drv, s["Laenge"], left=prev, color=c,
                    edgecolor=BG, linewidth=1.2, height=0.72)
            if s["Laenge"] > 5:
                ax.text(prev + s["Laenge"] / 2, drv, s["Compound"][0],
                        ha="center", va="center", fontsize=8.5,
                        fontweight="bold", color="#111")
            prev += s["Laenge"]
    ax.invert_yaxis()
    ax.set_xlabel("Runde")
    ax.grid(axis="x", alpha=0.2)
    ax.set_title(f"{ses.event['EventName']} {year} - Reifenstrategien\n"
                 f"sortiert nach Endposition", color=FG, fontsize=12)
    plt.tight_layout()
    save(fig, "strategie.png")

    KPI["strategie"] = {
        "event": str(ses.event["EventName"]), "jahr": year,
        "stints_gesamt": int(len(stints)),
        "stints_pro_fahrer": round(len(stints) / max(len(order), 1), 2),
        "laenge_je_compound": {
            k: round(float(v), 1) for k, v in
            stints.groupby("Compound")["Laenge"].mean().items()},
    }


# ---------------------------------------------------------------- 5
def degradation(year=2024, gp="Bahrain"):
    print(f"[5/5] Reifendegradation  {gp} {year}")
    ses = load(year, gp, "R", telemetry=False)
    laps = clean(ses, threshold=1.10).copy()
    laps["Sec"] = laps["LapTime"].dt.total_seconds()
    total = ses.total_laps
    laps["Corr"] = laps["Sec"] - (total - laps["LapNumber"]) * 1.8 * 0.03

    rows = []
    for (drv, st), g in laps.groupby(["Driver", "Stint"]):
        g = g.sort_values("TyreLife")
        if len(g) < 6:
            continue
        slope, inter = np.polyfit(g["TyreLife"], g["Corr"], 1)
        rows.append({"Driver": drv, "Team": g["Team"].iloc[0],
                     "Compound": g["Compound"].iloc[0], "Laps": len(g),
                     "Deg": float(slope), "Basis": float(inter)})
    deg = pd.DataFrame(rows)

    fig, ax = plt.subplots(1, 2, figsize=(13, 5.5),
                           gridspec_kw={"width_ratios": [3, 2]})

    for (drv, st), g in laps.groupby(["Driver", "Stint"]):
        if len(g) < 6:
            continue
        c = f1plt.get_compound_color(g["Compound"].iloc[0], session=ses)
        ax[0].plot(g["TyreLife"], g["Corr"], marker=".", ms=3.5, lw=0.9,
                   alpha=0.55, color=c)
    ax[0].set_xlabel("Reifenalter [Runden]")
    ax[0].set_ylabel("Rundenzeit, fuel-korrigiert [s]")
    ax[0].grid(alpha=0.2)
    ax[0].set_title("Jeder Stint einzeln", color=FG, fontsize=11)

    agg = deg.groupby("Compound")["Deg"].agg(["mean", "std", "count"])
    agg = agg.sort_values("mean")
    cols = [f1plt.get_compound_color(c, session=ses) for c in agg.index]
    ax[1].bar(agg.index, agg["mean"], yerr=agg["std"], color=cols,
              ecolor="#888", capsize=4)
    ax[1].set_ylabel("Degradation [s pro Runde]")
    ax[1].grid(axis="y", alpha=0.2)
    ax[1].set_title("Mittelwert je Mischung", color=FG, fontsize=11)

    fig.suptitle(f"{ses.event['EventName']} {year} - Reifendegradation "
                 f"(Fuel-Effekt herausgerechnet)", color=FG, fontsize=13)
    plt.tight_layout()
    save(fig, "degradation.png")

    KPI["degradation"] = {
        "event": str(ses.event["EventName"]), "jahr": year,
        "stints_ausgewertet": int(len(deg)),
        "je_compound": {
            str(c): {"mittel_s_pro_runde": round(float(r["mean"]), 4),
                     "std": round(float(r["std"]), 4),
                     "stints": int(r["count"])}
            for c, r in agg.iterrows()},
        "annahme_fuel": "1.8 kg pro Runde, 0.03 s pro kg",
    }


# ----------------------------------------------------------------
if __name__ == "__main__":
    print("\nErzeuge README-Grafiken. Der erste Lauf dauert einige Minuten.\n")
    for fn in (gear_map, telemetry_overlay, race_pace, strategy, degradation):
        try:
            fn()
        except Exception as exc:
            print(f"    FEHLER in {fn.__name__}: {exc}")
            KPI[fn.__name__] = {"fehler": str(exc)}

    out = ASSETS / "kennzahlen.json"
    out.write_text(json.dumps(KPI, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nKennzahlen: {out.relative_to(ROOT)}")
    print(f"Grafiken:   {len(list(ASSETS.glob('*.png')))} PNGs in assets/\n")
