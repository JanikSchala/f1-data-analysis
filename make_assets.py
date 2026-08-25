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

import importlib.util
import json
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")                      # kein fenster, nur dateien

import duckdb
import fastf1
import fastf1.plotting as f1plt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from fastf1.ergast import Ergast
from fastf1.utils import delta_time
from matplotlib.collections import LineCollection
from scipy.stats import binomtest
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import f1lab
from f1lab.design import BG, FG, GRID, MUTED, SERIEN, matplotlib_stil

warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent
ASSETS = ROOT / "assets"
ASSETS.mkdir(exist_ok=True)


def _skript_importieren(rel_pfad: str):
    """importiert ein Analyseskript als Modul, um seine Funktionen wieder-
    zuverwenden statt sie hier zu kopieren (siehe P24/P30 unten)."""
    pfad = ROOT / rel_pfad
    spec = importlib.util.spec_from_file_location(pfad.stem, pfad)
    assert spec is not None and spec.loader is not None
    modul = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modul)
    return modul

f1lab.enable_cache()
f1plt.setup_mpl(mpl_timedelta_support=False, color_scheme="fastf1")

DPI = 130

plt.rcParams.update(matplotlib_stil())

KPI: dict = {}


def save(fig, name: str) -> None:
    path = ASSETS / name
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"    -> {path.relative_to(ROOT)}  ({path.stat().st_size // 1024} KB)")


# ---------------------------------------------------------------- 1
def gear_map(year=2024, gp="Belgium"):
    print(f"[1/16] Gangwechsel-Karte  {gp} {year}")
    ses = f1lab.load(year, gp, "Q", telemetry=True)
    lap = ses.laps.pick_fastest()
    tel = lap.get_telemetry()

    x = tel["X"].to_numpy(float)
    y = tel["Y"].to_numpy(float)
    gear = tel["nGear"].to_numpy(float)

    pts = np.array([x, y]).T.reshape(-1, 1, 2)
    seg = np.concatenate([pts[:-1], pts[1:]], axis=1)

    lc = LineCollection(list(seg), norm=plt.Normalize(1, 8),
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
    print(f"[2/16] Telemetrie-Overlay  {gp} {year}  {d1} vs {d2}")
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
    print(f"[3/16] Race-Pace-Ranking  {gp} {year}")
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
    print(f"[4/16] Strategieuebersicht  {gp} {year}")
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
    print(f"[5/16] Reifendegradation  {gp} {year}")
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


# ---------------------------------------------------------------- 6
def undercut(year=2024):
    print(f"[6/16] Undercut-Erfolgsquote  Saison {year}")
    schedule = f1lab.event_dimension([year])
    rows = []
    for _, row in schedule.iterrows():
        try:
            ses = f1lab.load(year, int(row["round"]), "R", telemetry=False)
        except Exception:
            continue
        d = f1lab.undercut_duels(ses)
        if not d.empty:
            rows.append(d)
    duelle = pd.concat(rows, ignore_index=True)
    n = len(duelle)
    erfolge = int(duelle["erfolg"].sum())
    test = binomtest(erfolge, n, 0.5)
    ci = test.proportion_ci(confidence_level=0.95)
    rate = erfolge / n * 100

    fig, ax = plt.subplots(figsize=(9, 3.2))
    ax.barh([0], [rate], color=SERIEN[1], height=0.5)
    ax.errorbar([rate], [0],
               xerr=[[rate - ci.low * 100], [ci.high * 100 - rate]],
               color=FG, capsize=6, lw=1.5, fmt="none")
    ax.axvline(50, color=MUTED, lw=1.5, ls="--", label="50%-Erwartung")
    ax.set_xlim(0, 60)
    ax.set_yticks([])
    ax.set_xlabel("Undercut-Erfolgsquote [%], 95%-Konfidenzintervall")
    ax.set_title(f"Saison {year}: {erfolge}/{n} echte Undercut-Duelle "
                 f"erfolgreich ({rate:.1f}%)", loc="left", color=FG,
                 fontsize=13, pad=10)
    ax.legend(loc="lower right", frameon=False, labelcolor=FG, fontsize=9)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="x", alpha=0.3, linewidth=0.8, color=GRID)
    ax.set_axisbelow(True)
    plt.tight_layout()
    save(fig, "undercut.png")

    KPI["undercut"] = {
        "saison": year, "duelle": n, "erfolge": erfolge,
        "quote_pct": round(rate, 1),
        "ki_95_pct": [round(ci.low * 100, 1), round(ci.high * 100, 1)],
        "p_wert_gegen_50pct": test.pvalue,
    }


# ---------------------------------------------------------------- 7
def safety_car(year=2024, gp="Canada"):
    print(f"[7/16] Safety-Car-Kompaktierung  {gp} {year}")
    ses = f1lab.load(year, gp, "R", telemetry=False)
    phasen = f1lab.track_status_phases(ses)
    neutral = phasen[phasen["label"].isin(["safety car", "vsc"])]
    spread = f1lab.field_spread(ses)
    komp = f1lab.sc_compaction(neutral, spread)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(spread.index, spread.to_numpy(), color=MUTED, lw=1.4,
            label="Feldstreckung [s]")
    for p in neutral.itertuples():
        ax.axvspan(p.lap_start, p.lap_end, color=SERIEN[1], alpha=0.18)
    for _, k in komp.iterrows():
        ax.annotate(f"-{k['kompaktierung_pct']:.0f}%",
                    xy=(k["ende"], k["minimum_s"]),
                    xytext=(0, -14), textcoords="offset points",
                    ha="center", color=SERIEN[1], fontsize=9,
                    fontweight="bold")
    ax.set_xlabel("Runde")
    ax.set_ylabel("Sekunden zwischen erstem und letztem Fahrer")
    ax.set_title(f"{ses.event['EventName']} {year} - Safety-Car/VSC-Phasen "
                 f"(rot) stauchen das Feld zusammen", loc="left", color=FG,
                 fontsize=13, pad=10)
    ax.legend(loc="upper right", frameon=False, labelcolor=FG, fontsize=9)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(alpha=0.3, linewidth=0.8, color=GRID)
    ax.set_axisbelow(True)
    plt.tight_layout()
    save(fig, "safety_car.png")

    KPI["safety_car"] = {
        "event": str(ses.event["EventName"]), "jahr": year,
        "phasen": int(len(neutral)),
        "kompaktierung": [
            {"start": int(r["start"]), "ende": int(r["ende"]),
             "baseline_s": round(float(r["baseline_s"]), 1),
             "minimum_s": round(float(r["minimum_s"]), 1),
             "kompaktierung_pct": round(float(r["kompaktierung_pct"]), 1)}
            for _, r in komp.iterrows()],
    }


# ---------------------------------------------------------------- 8
def lap_simulation(ref=(2024, "Bahrain", "Q")):
    print(f"[8/16] Rundenzeit-Simulation  {ref[1]} {ref[0]} {ref[2]}")
    ses_ref = f1lab.load(*ref, telemetry=True)
    dist, kappa, speed_real = f1lab.lap_speed_profile(ses_ref)
    t_real = float(ses_ref.laps.pick_fastest()["LapTime"].total_seconds())

    params = f1lab.calibrate_lap_model(dist, kappa, speed_real)
    v_sim, t_sim = f1lab.simulate_lap(dist, kappa, params["mu_g"],
                                      params["a_accel"], params["a_brake"],
                                      params["v_top"])

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(dist, speed_real * 3.6, color=MUTED, lw=1.8,
            label="Echte Telemetrie")
    ax.plot(dist, v_sim * 3.6, color=SERIEN[0], lw=1.8, ls="--",
            label="Simulation (an diese Runde kalibriert)")
    ax.set_xlabel("Distanz [m]")
    ax.set_ylabel("Speed [km/h]")
    ax.set_title(f"{ref[1]} {ref[0]} {ref[2]} - physikalisches "
                 f"Punktmassenmodell: real {t_real:.2f}s, simuliert "
                 f"{t_sim:.2f}s ({100 * (t_sim - t_real) / t_real:+.1f}%)",
                 loc="left", color=FG, fontsize=13, pad=10)
    ax.legend(loc="upper right", frameon=False, labelcolor=FG, fontsize=9)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(alpha=0.3, linewidth=0.8, color=GRID)
    ax.set_axisbelow(True)
    plt.tight_layout()
    save(fig, "rundenzeit_simulation.png")

    KPI["lap_simulation"] = {
        "event": ref[1], "jahr": ref[0], "session": ref[2],
        "rundenzeit_real_s": round(t_real, 3),
        "rundenzeit_simuliert_s": round(float(t_sim), 3),
        "abweichung_pct": round(100 * (t_sim - t_real) / t_real, 2),
        "parameter": {k: round(float(v), 3) for k, v in params.items()},
    }


# ---------------------------------------------------------------- 9
def historic_lap_times():
    print("[9/16] 75 Jahre F1: Rundenzeit-Entwicklung dreier Strecken")
    erg = Ergast(result_type="pandas", auto_cast=True)
    strecken = {"monza": "Monza", "spa": "Spa-Francorchamps",
                "silverstone": "Silverstone"}
    rows = []
    for circuit_id, name in strecken.items():
        for year in range(1950, 2025, 4):
            try:
                res = erg.get_race_results(season=year, circuit=circuit_id)
            except Exception:
                continue
            if not res.content or res.content[0].empty:
                continue
            df = res.content[0]
            sieger = df[df["position"] == 1]
            if sieger.empty:
                continue
            sieger = sieger.iloc[0]
            if pd.isna(sieger["totalRaceTime"]) or not sieger["laps"]:
                continue
            sekunden = sieger["totalRaceTime"].total_seconds() / sieger["laps"]
            if sekunden < 30:      # bekannte kaputte Ergast-Werte fuer sehr alte Rennen
                continue
            rows.append({"strecke": name, "season": year,
                         "rundenzeit_s": sekunden})
    rz = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(11, 5.5))
    for i, (strecke, g) in enumerate(rz.groupby("strecke")):
        g = g.sort_values("season")
        ax.plot(g["season"], g["rundenzeit_s"], marker="o", ms=4, lw=1.6,
               color=SERIEN[i % len(SERIEN)], label=strecke)
    ax.axvline(1979, color=MUTED, lw=1, ls=":")
    ax.text(1979, ax.get_ylim()[1], " Spa fehlt 1971-82,\n kehrt verkuerzt zurueck",
           fontsize=8, color=MUTED, va="top")
    ax.set_ylabel("Rundenzeit des Siegers [s]")
    ax.set_xlabel("Saison")
    ax.set_title("75 Jahre F1: dieselbe Strecke ist selten dieselbe Strecke",
                loc="left", color=FG, fontsize=13, pad=10)
    ax.legend(loc="upper right", frameon=False, labelcolor=FG, fontsize=9)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(alpha=0.3, linewidth=0.8, color=GRID)
    ax.set_axisbelow(True)
    plt.tight_layout()
    save(fig, "historische_trends.png")

    KPI["historie"] = {
        "strecken": {
            strecke: {
                "erste_saison": int(g["season"].min()),
                "letzte_saison": int(g["season"].max()),
                "rundenzeit_erste_s": round(float(
                    g.sort_values("season")["rundenzeit_s"].iloc[0]), 1),
                "rundenzeit_letzte_s": round(float(
                    g.sort_values("season")["rundenzeit_s"].iloc[-1]), 1),
            }
            for strecke, g in rz.groupby("strecke")},
    }


# ---------------------------------------------------------------- 10
def track_geometry(season=2024):
    print(f"[10/16] Streckenprofil  Saison {season}")
    dim = f1lab.event_dimension([season])
    events = [(season, r) for r in dim["round"]]
    geo = f1lab.circuit_dimension(events).dropna(subset=["length_m", "corners"])

    span = geo["elev_span_m"].astype(float)
    norm = plt.Normalize(float(span.min()), float(span.max()))
    fig, ax = plt.subplots(figsize=(10, 6.5))
    sc = ax.scatter(geo["length_m"] / 1000, geo["corners"], s=190, c=span,
                    cmap="viridis", norm=norm, edgecolor=BG, linewidth=2,
                    zorder=3)
    # nur die extreme beschriften (laenge, kurvenzahl, hoehenspanne) - alle
    # 24 Strecken zu labeln ueberlappt sich zu unlesbarem Kauderwelsch bei
    # dicht beieinanderliegenden Punkten (siehe P02s strecken_profil()).
    zeigen = (set(geo.nlargest(2, "length_m").index)
             | set(geo.nsmallest(2, "length_m").index)
             | set(geo.nlargest(1, "elev_span_m").index)
             | set(geo.nlargest(1, "corners").index)
             | set(geo.nsmallest(1, "corners").index))
    x_mitte = (geo["length_m"].min() + geo["length_m"].max()) / 2000
    for i in zeigen:
        r = geo.loc[i]
        rechts = r["length_m"] / 1000 > x_mitte
        ax.annotate(r["circuit"], (r["length_m"] / 1000, r["corners"]),
                    textcoords="offset points",
                    xytext=(-10, 0) if rechts else (10, 0),
                    ha="right" if rechts else "left", va="center",
                    color=FG, fontsize=9)
    ax.set_xlabel("Streckenlaenge [km] - gefahrene Ideallinie")
    ax.set_ylabel("Kurven")
    ax.set_title(f"Streckengeometrie {season}: Laenge, Kurvenzahl und "
                 f"Hoehenspanne", loc="left", color=FG, fontsize=13, pad=10)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(alpha=0.3, linewidth=0.8, color=GRID)
    ax.set_axisbelow(True)
    cb = fig.colorbar(sc, ax=ax, pad=0.02)
    cb.set_label("Hoehenspanne [m]", color=MUTED, fontsize=10)
    cb.ax.tick_params(colors=MUTED, length=0)
    plt.tight_layout()
    save(fig, "streckenprofil.png")

    KPI["streckenprofil"] = {
        "saison": season, "strecken_vermessen": int(len(geo)),
        "laengste": str(geo.loc[geo["length_m"].idxmax(), "circuit"]),
        "kuerzeste": str(geo.loc[geo["length_m"].idxmin(), "circuit"]),
        "groesste_hoehenspanne": str(geo.loc[span.idxmax(), "circuit"]),
        "hoehenspanne_m": round(float(span.max()), 1),
    }


# ---------------------------------------------------------------- 11
def weather_effect(event=("Japan", 2024, "R")):
    print(f"[11/16] Streckentemperatur-Effekt  {event[0]} {event[1]}")
    ses = f1lab.load(event[1], event[0], event[2], telemetry=False, weather=True)
    merged = f1lab.weather_join(ses)
    erg = f1lab.temperature_effect(merged)

    d = erg["dry"]
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.scatter(d["TrackTemp"], d["partial"], s=10, color=MUTED, alpha=0.4,
              edgecolors="none")
    xs = np.linspace(d["TrackTemp"].min(), d["TrackTemp"].max(), 50)
    ax.plot(xs, erg["coef_temp"] * xs + erg["intercept"], color=SERIEN[1], lw=2.2)
    ax.set_xlabel("Streckentemperatur [°C]")
    ax.set_ylabel("Rundenzeit ggue. Fahrer-Median,\num Reifenalter bereinigt [s]")
    ax.set_title(f"{ses.event['EventName']} {event[1]} - "
                 f"+{erg['coef_temp']:.3f} s/°C bei Reifenalter-Kontrolle "
                 f"(R² {erg['r2_tyre_only']:.2f}→{erg['r2_voll']:.2f})",
                 loc="left", color=FG, fontsize=12, pad=10)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(alpha=0.3, linewidth=0.8, color=GRID)
    ax.set_axisbelow(True)
    plt.tight_layout()
    save(fig, "wetter_effekt.png")

    KPI["wetter"] = {
        "event": str(ses.event["EventName"]), "jahr": event[1],
        "n_runden": int(erg["n"]),
        "naive_regression": {"steigung_s_pro_grad": round(float(erg["naiv_slope"]), 4),
                             "r2": round(float(erg["naiv_r2"]), 3)},
        "kontrolliert": {
            "r2_nur_reifenalter": round(float(erg["r2_tyre_only"]), 3),
            "r2_plus_temperatur": round(float(erg["r2_voll"]), 3),
            "koeffizient_s_pro_grad": round(float(erg["coef_temp"]), 4),
        },
    }


# ---------------------------------------------------------------- 12
def driving_style_clusters():
    print("[12/16] Fahrstil-Clustering  Saison 2024")
    p24 = _skript_importieren(
        "09_machine_learning/p24_fahrstil_clustering_wer_faehrt_wie.py")

    df = p24.sammle_features()
    agg = df.groupby("driver").mean(numeric_only=True)
    X = StandardScaler().fit_transform(agg)
    pcs = PCA(n_components=2).fit_transform(X)
    bestes_k, scores = p24.beste_clusteranzahl(X)
    labels = KMeans(n_clusters=bestes_k, n_init=20, random_state=0).fit_predict(X)

    fig, ax = plt.subplots(figsize=(9, 7))
    p24.zeichne_pca(ax, pcs, agg, labels)
    plt.tight_layout()
    save(fig, "fahrstil_cluster.png")

    KPI["fahrstil_cluster"] = {
        "saison": p24.SEASON, "strecken": list(p24.STRECKEN),
        "fahrer": int(len(agg)), "gewaehltes_k": int(bestes_k),
        "silhouette_scores": {int(k): round(float(v), 3) for k, v in scores.items()},
    }


# ---------------------------------------------------------------- 13
def warehouse_pace():
    print("[13/16] Data-Warehouse-Query  Saison-Pace-Ranking (DuckDB)")
    con = duckdb.connect(str(ROOT / "10_data_engineering/f1_warehouse/f1.duckdb"),
                         read_only=True)
    rang = con.execute("""
        SELECT Driver, avg(rel_pace) AS mittel_rel_pace, count(*) AS events
        FROM mart_driver_pace
        GROUP BY Driver
        HAVING count(*) >= 5
        ORDER BY mittel_rel_pace ASC
        LIMIT 15
    """).df()
    con.close()

    # als delta-vom-schnellsten statt absolutem rel_pace plotten - die werte
    # clustern alle nahe 1.000, eine bei 0 startende achse macht jeden
    # balken optisch gleich lang und das ranking damit unsichtbar.
    delta_pct = (rang["mittel_rel_pace"] - 1.0) * 100
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(rang["Driver"], delta_pct, color=SERIEN[0], height=0.65)
    ax.invert_yaxis()
    ax.set_xlabel("Mittlere relative Race Pace ggue. Event-Schnellster [%]")
    ax.set_title("Saison-Pace-Ranking direkt aus dem DuckDB-Warehouse",
                 loc="left", color=FG, fontsize=13, pad=10)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="x", alpha=0.3, linewidth=0.8, color=GRID)
    ax.set_axisbelow(True)
    plt.tight_layout()
    save(fig, "warehouse_pace.png")

    KPI["warehouse_pace"] = {
        "top5": [{"fahrer": r["Driver"],
                  "mittel_rel_pace": round(float(r["mittel_rel_pace"]), 4),
                  "events": int(r["events"])}
                 for _, r in rang.head(5).iterrows()],
        "quelle": "10_data_engineering/f1_warehouse/f1.duckdb, view mart_driver_pace",
    }


# ---------------------------------------------------------------- 14
def position_chart(year=2024, gp="Hungary"):
    print(f"[14/16] Positionsverlauf  {gp} {year}")
    ses = f1lab.load(year, gp, "R", telemetry=False)
    pos = f1lab.position_progression(ses)
    order = ses.results.sort_values("Position")["Abbreviation"].tolist()
    top5 = [d for d in order if d in pos.columns][:5]

    fig, ax = plt.subplots(figsize=(10, 6.5))
    for drv in pos.columns:
        style = f1plt.get_driver_style(drv, ["color", "linestyle"], session=ses)
        lw, alpha = (2.4, 1.0) if drv in top5 else (1.0, 0.35)
        ax.plot(pos.index, pos[drv], lw=lw, alpha=alpha,
                color=style.get("color", MUTED),
                label=drv if drv in top5 else None)
    ax.invert_yaxis()
    ax.set_xlabel("Runde")
    ax.set_ylabel("Position")
    ax.set_title(f"{ses.event['EventName']} {year} - Positionsverlauf "
                 f"(genau die Grafik, die auch der automatische PDF-"
                 f"Rennbericht zeigt)", loc="left", color=FG, fontsize=12,
                 pad=10)
    ax.legend(loc="upper right", frameon=False, labelcolor=FG, fontsize=9,
              ncol=5)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(alpha=0.3, linewidth=0.8, color=GRID)
    ax.set_axisbelow(True)
    plt.tight_layout()
    save(fig, "positionsverlauf.png")

    KPI["positionsverlauf"] = {
        "event": str(ses.event["EventName"]), "jahr": year,
        "top5_endstand": top5,
    }


# ---------------------------------------------------------------- 15
def live_timing_board():
    print("[15/16] Live-Timing-Board  Bahrain 2024 R (Replay)")
    p30 = _skript_importieren(
        "12_live_timing/p30_live_timing_aufzeichnen_und_in_echtzeit_auswerte.py")

    ses = f1lab.load(2024, "Bahrain", "R", telemetry=False)
    con = p30.zeitreihe_anlegen(":memory:")
    p30.replay_als_livefeed(ses, con)
    n = con.execute("SELECT count(*) FROM pace_snapshots").fetchone()[0]

    out_png = ASSETS / "live_board.png"
    p30.dashboard_rendern(con, ses, out_png)
    con.close()
    print(f"    -> {out_png.relative_to(ROOT)}  "
         f"({out_png.stat().st_size // 1024} KB)")

    KPI["live_timing"] = {
        "event": str(ses.event["EventName"]), "jahr": 2024,
        "datenpunkte": int(n),
        "hinweis": "Replay derselben Session in ihrer tatsaechlichen "
                   "Rundenreihenfolge, ersetzt einen echten Live-Feed "
                   "(siehe P30-Docstring).",
    }


# ---------------------------------------------------------------- 16
def constructors_championship():
    print("[16/16] Konstrukteurs-WM-Titelchance  laufende Saison")
    p45 = _skript_importieren(
        "08_historie/p45_konstrukteurs_wm_simulator_wer_gewinnt_das_team.py")

    erg = Ergast(result_type="pandas", auto_cast=True)
    standings = p45._mit_wiederholung(
        erg.get_constructor_standings, season=p45.YEAR).content[0]
    remaining = fastf1.get_events_remaining(include_testing=False)
    sprint_flags = remaining["EventFormat"].str.contains(
        "sprint", case=False).tolist()
    gefahrene_runden = 23 - len(remaining)

    races, sprints = p45.saison_verlauf(erg, p45.YEAR, gefahrene_runden)
    base_points = standings.set_index("constructorId")["points"].to_dict()
    name_map = standings.set_index("constructorId")["constructorName"].to_dict()
    teams = list(base_points.keys())

    pos_hist_alle = (races.groupby("constructorId")["position"]
                     .apply(lambda s: s.dropna().to_numpy()).to_dict())
    sprint_hist = (sprints.groupby("constructorId")["position"]
                  .apply(lambda s: s.dropna().to_numpy()).to_dict()
                  if not sprints.empty else {})

    rng = np.random.default_rng(7)
    sim = p45.monte_carlo_team(base_points, teams, pos_hist_alle, sprint_hist,
                               sprint_flags, None, rng)
    chance = pd.Series(
        {name_map[t]: (sim.idxmax(axis=1) == t).mean() * 100
         for t in teams}).sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    p45.zeichne_titelchance(ax, chance)
    plt.tight_layout()
    save(fig, "konstrukteurs_titelchance.png")

    KPI["konstrukteurs_wm"] = {
        "saison": p45.YEAR,
        "fuehrend": str(chance.index[0]),
        "titelchance_pct": round(float(chance.iloc[0]), 1),
        "standings_top3": [
            {"team": r["constructorName"], "punkte": float(r["points"])}
            for _, r in standings.head(3).iterrows()],
    }


# ----------------------------------------------------------------
if __name__ == "__main__":
    print(f"\nErzeuge README-Grafiken mit f1lab {f1lab.__version__}.")
    print("Der erste Lauf dauert einige Minuten.\n")

    for fn in (gear_map, telemetry_overlay, race_pace, strategy, degradation,
               undercut, safety_car, lap_simulation, historic_lap_times,
               track_geometry, weather_effect, driving_style_clusters,
               warehouse_pace, position_chart, live_timing_board,
               constructors_championship):
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
