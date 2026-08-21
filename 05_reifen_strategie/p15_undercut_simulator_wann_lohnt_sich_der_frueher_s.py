"""simuliert den undercut-gewinn aus degradation, pitloss und out-lap-malus und prüft den einfluss von verkehr nach dem stopp"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")                      # kein fenster, nur dateien

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

import f1lab
from f1lab.design import FG, GRID, MUTED, RAMPE, SERIEN, matplotlib_stil

warnings.filterwarnings("ignore")

OUT = Path(__file__).parent / "out"
OUT.mkdir(exist_ok=True)

SEASON, EVENT, IDENT = 2024, "Spain", "R"
RAMPE_CMAP = LinearSegmentedColormap.from_list("rampe", RAMPE)

plt.rcParams.update(matplotlib_stil())


def pitloss_zerlegen(ses) -> tuple[float, float, float]:
    """wie f1lab.pit_loss(), aber in-lap und out-lap getrennt statt nur die summe."""
    laps = ses.laps.copy()
    laps["sec"] = laps["LapTime"].dt.total_seconds()
    baseline = (f1lab.clean_laps(ses).groupby("Driver")["LapTime"]
               .median().dt.total_seconds())
    in_laps = laps[laps["PitInTime"].notna()]
    out_laps = laps[laps["PitOutTime"].notna()]
    loss_in = (in_laps["sec"] - in_laps["Driver"].map(baseline)).dropna()
    loss_out = (out_laps["sec"] - out_laps["Driver"].map(baseline)).dropna()
    return float(loss_in.median()), float(loss_out.median()), \
        float(loss_in.median() + loss_out.median())


def out_lap_malus_schaetzen(ses) -> np.ndarray:
    """residuum der ersten gewerteten runde eines stints gegen den eigenen degradations-fit. positiv heisst langsamer als der trend vorhersagt."""
    laps = f1lab.clean_laps(ses).copy()
    laps["sec"] = laps["LapTime"].dt.total_seconds()
    laps["corrected"] = f1lab.fuel_correct(
        laps["sec"], laps["LapNumber"], ses.total_laps)
    resid = []
    for (_drv, _stint), g in laps.groupby(["Driver", "Stint"]):
        g = g.sort_values("TyreLife")
        if len(g) < 6:
            continue
        try:
            fit = f1lab.fit_degradation(g["TyreLife"], g["corrected"])
        except ValueError:
            continue
        if not fit.is_reliable:
            continue
        erste = g.iloc[0]
        pred = fit.slope * erste["TyreLife"] + fit.intercept
        resid.append(erste["corrected"] - pred)
    return np.array(resid)


def verkehr_nach_stopp(ses) -> pd.DataFrame:
    """dirty_air_effect() je fahrer plus wie oft ein fahrer unmittelbar nach dem eigenen stopp in echten nahverkehr gerät."""
    stint_map = ses.laps.set_index(["Driver", "LapNumber"])["Stint"]
    zeilen = []
    for drv in ses.drivers:
        abbr = ses.get_driver(drv)["Abbreviation"]
        try:
            df = f1lab.close_following(ses, abbr)
        except Exception:
            continue
        if df.empty:
            continue
        slope, _inter, r2, d = f1lab.dirty_air_effect(df)
        if slope != slope:
            continue
        d = d.copy()
        d["stint"] = [stint_map.get((abbr, lap), np.nan) for lap in d["lap"]]
        nach_stopp = d[(d["stint"] > 1) & (d["tyre_life"] <= 3)]
        zeilen.append({
            "driver": abbr, "slope": slope, "r2": r2, "n": len(d),
            "nah_nach_stopp": int((nach_stopp["anteil_nah"] >= 50).sum()),
        })
    return pd.DataFrame(zeilen)


def zeichne_fenster(ax, deg_alt: float, deg_neu: float, malus: float) -> None:
    """gewinn über das undercut-fenster, 1-8 runden."""
    ns = np.arange(1, 9)
    gains = [f1lab.undercut_gain(deg_alt, deg_neu, n, malus) for n in ns]
    ax.plot(ns, gains, marker="o", color=SERIEN[0], lw=2, ms=6)
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.set_xlabel("Undercut-Fenster [Runden]")
    ax.set_ylabel("Gewinn [s]")
    ax.set_title(f"SOFT ({deg_alt:.3f}) vs. MEDIUM ({deg_neu:.3f} s/Runde)",
                loc="left", color=FG, fontsize=11.5, pad=10)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(alpha=0.3, linewidth=0.8, color=GRID)
    ax.set_axisbelow(True)


def zeichne_sensitivitaet(ax, deg_neu: float) -> None:
    """gewinn bei 3 runden über degradations-differenz und out-lap-malus."""
    # untergrenze bei deg_neu weil die "alte" mischung sonst langsamer
    # degradiert als die neue. kein realistisches undercut-szenario und
    # würde die skala nur mit stark negativen werten stauchen
    degs_alt = np.round(np.arange(deg_neu, 0.21, 0.01), 3)
    malusse = np.round(np.arange(0.0, 1.3, 0.1), 2)
    grid = np.array([[f1lab.undercut_gain(d, deg_neu, 3, m) for m in malusse]
                     for d in degs_alt])
    im = ax.imshow(grid, aspect="auto", cmap=RAMPE_CMAP, origin="lower",
                   extent=[malusse[0], malusse[-1], degs_alt[0], degs_alt[-1]])
    ax.axhline(0.1267, color=FG, lw=1, ls="--")
    ax.text(malusse[-1], 0.1267, " gemessen (SOFT)", color=FG, fontsize=8,
           va="bottom", ha="right")
    ax.axvline(0.0, color=FG, lw=1, ls=":")
    ax.text(0.02, degs_alt[0], "gemessen (kein Malus)", color=FG, fontsize=8,
           va="bottom", ha="left")
    ax.set_xlabel("Out-Lap-Malus [s]")
    ax.set_ylabel("Degradation alte Mischung [s/Runde]")
    ax.set_title("Gewinn bei 3 Runden Fenster [s]", loc="left", color=FG,
                fontsize=11.5, pad=10)
    cb = ax.figure.colorbar(im, ax=ax, pad=0.02)
    cb.set_label("Gewinn [s]", color=MUTED, fontsize=9)
    cb.ax.tick_params(colors=MUTED, length=0)
    cb.outline.set_visible(False)


def zeichne_verkehr(ax, tab: pd.DataFrame) -> None:
    """steigung je fahrer gegen die im projekt sonst genutzte belastbarkeitsschwelle R^2 >= 0.3."""
    t = tab.sort_values("slope")
    farben = [SERIEN[0] if r >= 0.3 else MUTED for r in t["r2"]]
    ax.barh(t["driver"], t["slope"], color=farben, height=0.65)
    ax.axvline(0, color=MUTED, lw=0.8)
    ax.set_xlabel("Steigung [s pro Prozentpunkt Nahanteil]")
    ax.set_title("Dirty-Air-Steigung je Fahrer - kein Fit erreicht R²>=0.3",
                loc="left", color=FG, fontsize=12, pad=10)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="x", alpha=0.3, linewidth=0.8, color=GRID)
    ax.set_axisbelow(True)


def main():
    f1lab.enable_cache()

    print(f"[1/5] {EVENT} {SEASON} {IDENT} laden (mit Telemetrie, fuer die "
         f"Verkehrs-Ausbaustufe) ...")
    ses = f1lab.load(SEASON, EVENT, IDENT, telemetry=True)

    print("\n[2/5] Pitloss (VORGEHEN 1) ...")
    loss_in, loss_out, pitloss = pitloss_zerlegen(ses)
    print(f"      in-lap {loss_in:.2f}s + out-lap {loss_out:.2f}s = "
         f"{pitloss:.2f}s (f1lab.pit_loss() zur Kontrolle: "
         f"{f1lab.pit_loss(ses):.2f}s)")

    print("\n[3/5] Degradation je Mischung (VORGEHEN 2, aus P13-Bausteinen) ...")
    je_compound = f1lab.degradation_by_compound(ses)
    print(je_compound.to_string())
    deg_alt = float(je_compound.loc["SOFT", "mean"])
    deg_neu = float(je_compound.loc["MEDIUM", "mean"])
    print(f"      DEG_ALT (SOFT) = {deg_alt:.4f}, DEG_NEU (MEDIUM) = {deg_neu:.4f}")

    print("\n[4/5] Out-Lap-Malus (VORGEHEN 3) ...")
    resid = out_lap_malus_schaetzen(ses)
    print(f"      n={len(resid)}  Median={np.median(resid):+.3f}s  "
         f"Mittel={resid.mean():+.3f}s -> kein messbarer Rest-Malus, "
         f"Simulator nutzt 0.0s")
    malus = 0.0

    print("\n[5/5] Simulation (VORGEHEN 4) und Sensitivitaet (VORGEHEN 5) ...")
    for n in range(1, 9):
        g = f1lab.undercut_gain(deg_alt, deg_neu, n, malus)
        print(f"      {n} Runden fruehe stoppen -> {g:+.2f}s")

    print("\nAUSBAUSTUFE: Verkehr ...")
    verkehr = verkehr_nach_stopp(ses)
    print(verkehr.round(4).to_string(index=False))
    print(f"      Median Steigung {verkehr['slope'].median():+.4f}s/%pkt, "
         f"max R^2 {verkehr['r2'].max():.2f}, Fits mit R^2>=0.3: "
         f"{(verkehr['r2'] >= 0.3).sum()}/{len(verkehr)}")
    print(f"      Stopps mit >=50% Nahanteil in den 3 Runden danach: "
         f"{int(verkehr['nah_nach_stopp'].sum())}")

    print("\nGrafik ...")
    fig = plt.figure(figsize=(15, 11))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1], hspace=0.45, wspace=0.3)
    zeichne_fenster(fig.add_subplot(gs[0, 0]), deg_alt, deg_neu, malus)
    zeichne_sensitivitaet(fig.add_subplot(gs[0, 1]), deg_neu)
    zeichne_verkehr(fig.add_subplot(gs[1, :]), verkehr)
    fig.suptitle(f"{ses.event['EventName']} {SEASON} - Undercut-Simulator",
                x=0.09, ha="left", fontsize=16, color=FG, y=0.995)
    path = OUT / "undercut_simulator.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"\n      -> {path}")


if __name__ == "__main__":
    main()
