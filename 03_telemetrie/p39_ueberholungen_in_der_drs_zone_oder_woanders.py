"""lokalisiert ueberholorte in der telemetrie und prueft sie gegen drs-zonen und bremszonen ueber strecken und saisons"""
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
from scipy.stats import chi2_contingency, mannwhitneyu

import f1lab
from f1lab.design import FG, GRID, MUTED, POSITIV, SERIEN, matplotlib_stil

warnings.filterwarnings("ignore")

OUT = Path(__file__).parent / "out"
OUT.mkdir(exist_ok=True)

REFERENZ = (2024, "Italy", "R")          # monza hat die laengsten drs-zonen im cache

plt.rcParams.update(matplotlib_stil())


def zeichne_streckenkarte(ax, ref_xy: np.ndarray, dist: np.ndarray,
                          zonen: pd.DataFrame, orte: pd.DataFrame) -> None:
    """zeichnet ueberholorte auf der streckenkarte mit markierten drs-zonen."""
    ax.plot(ref_xy[:, 0], ref_xy[:, 1], color=GRID, lw=6, zorder=1,
           solid_capstyle="round")
    for _, z in zonen.iterrows():
        maske = (dist >= z["start_m"]) & (dist <= z["end_m"])
        ax.plot(ref_xy[maske, 0], ref_xy[maske, 1], color=SERIEN[0], lw=6,
               zorder=2, solid_capstyle="butt")

    for in_zone, gruppe in orte.groupby("in_drs_zone"):
        idx = [int(np.argmin(np.abs(dist - d))) for d in gruppe["distance_m"]]
        ax.scatter(ref_xy[idx, 0], ref_xy[idx, 1],
                  color=POSITIV if in_zone else SERIEN[1], s=45, zorder=3,
                  edgecolor=FG, linewidth=0.5,
                  label="in DRS-Zone" if in_zone else "ausserhalb")

    ax.set_aspect("equal")
    ax.axis("off")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=2,
             frameon=False, labelcolor=FG, fontsize=10)
    ax.set_title(f"{REFERENZ[1]} {REFERENZ[0]} {REFERENZ[2]}: Ueberholorte "
                "gegen DRS-Zonen (blau)", loc="left", color=FG, fontsize=13,
                pad=10)


def zeichne_saisonvergleich(ax, ergebnisse: pd.DataFrame) -> None:
    e = ergebnisse.sort_values("drs_pct")
    ax.barh(e["gp"], e["drs_pct"], color=SERIEN[0], height=0.6)
    ax.set_xlabel("Anteil Ueberholungen in der DRS-Zone [%]")
    ax.set_title("AUSBAUSTUFE: DRS-Anteil je Strecke, Saison 2024", loc="left",
                color=FG, fontsize=13, pad=10)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="x", alpha=0.3, linewidth=0.8, color=GRID)
    ax.set_axisbelow(True)


def drs_saison_scan(saison: int) -> pd.DataFrame:
    """berechnet den drs-anteil je telemetriefaehigem rennen einer saison. wiederverwendbar fuer mehrere saisons."""
    inv = f1lab.cached_sessions()
    tel_rennen = sorted(inv[(inv["season"] == saison) & (inv["ident"] == "R")
                            & inv["telemetry"]]["event"].unique())
    zeilen = []
    for gp in tel_rennen:
        s = f1lab.load(saison, gp, "R", telemetry=True)
        ev = f1lab.overtake_events(s)
        if ev.empty:
            continue
        o = f1lab.overtake_locations(s)
        if o.empty:
            continue
        zeilen.append({
            "gp": gp, "ueberholungen": len(ev), "lokalisiert": len(o),
            "abdeckung_pct": round(100 * len(o) / len(ev), 1),
            "drs_pct": round(100 * o["in_drs_zone"].mean(), 1),
        })
    return pd.DataFrame(zeilen)


def zeichne_drs_ueber_zeit(ax, erg_2018: pd.DataFrame, erg_2024: pd.DataFrame) -> None:
    daten = [erg_2018["drs_pct"], erg_2024["drs_pct"]]
    bp = ax.boxplot(daten, tick_labels=["2018", "2024"], patch_artist=True,
                    widths=0.5, showfliers=False)
    for patch, farbe in zip(bp["boxes"], SERIEN):
        patch.set_facecolor(farbe)
        patch.set_alpha(0.6)
    for element in ("whiskers", "caps", "medians"):
        for line in bp[element]:
            line.set_color(FG)
    for i, erg in enumerate((erg_2018, erg_2024), start=1):
        jitter = np.random.default_rng(0).uniform(-0.08, 0.08, len(erg))
        ax.scatter(np.full(len(erg), i) + jitter, erg["drs_pct"],
                  color=MUTED, s=18, zorder=3)
    ax.set_ylabel("DRS-Anteil je Strecke [%]")
    ax.set_title("VIERTE AUSBAUSTUFE: DRS-Anteil vor (2018) und nach (2024) "
                "dem Ground-Effect-Umbruch", loc="left", color=FG,
                fontsize=13, pad=10)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="y", alpha=0.3, linewidth=0.8, color=GRID)
    ax.set_axisbelow(True)


def zeichne_bremszonen_abstand(ax, vorlauf_echt: np.ndarray,
                               vorlauf_zufall: np.ndarray) -> None:
    bins = np.linspace(0, max(vorlauf_echt.max(), 1500), 25)
    ax.hist(vorlauf_zufall, bins=bins, density=True, color=GRID, alpha=0.7,
           label="Zufalls-Baseline (gleichverteilt)")
    ax.hist(vorlauf_echt, bins=bins, density=True, color=SERIEN[0], alpha=0.7,
           label="Echte Ueberholorte")
    ax.set_xlabel("Abstand zur naechsten Bremszone [m]")
    ax.set_ylabel("Dichte")
    ax.set_title("DRITTE AUSBAUSTUFE: Bremszonen-Naehe, echt gegen Zufall",
                loc="left", color=FG, fontsize=13, pad=10)
    ax.legend(loc="upper right", frameon=False, labelcolor=FG, fontsize=10)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(alpha=0.3, linewidth=0.8, color=GRID)
    ax.set_axisbelow(True)


def main():
    f1lab.enable_cache()

    print(f"[1/3] {REFERENZ[1]} {REFERENZ[0]} {REFERENZ[2]} laden, "
         "Ueberholorte bestimmen (VORGEHEN 1-2) ...")
    ses = f1lab.load(*REFERENZ, telemetry=True)
    events = f1lab.overtake_events(ses)
    orte = f1lab.overtake_locations(ses)
    print(f"      {len(events)} Ueberholungen, {len(orte)} lokalisiert "
         f"({100 * len(orte) / len(events):.1f}%)")

    print("\n[2/3] DRS-Zonen und Abgleich (VORGEHEN 3-4) ...")
    q = f1lab.load(REFERENZ[0], int(ses.event["RoundNumber"]), "Q",
                   telemetry=True)
    lap = q.laps.pick_fastest()
    zonen = f1lab.drs_zones(q, str(lap["Driver"]))
    print(zonen.round(0).to_string(index=False))
    anteil = 100 * orte["in_drs_zone"].mean()
    print(f"\n      {int(orte['in_drs_zone'].sum())}/{len(orte)} lokalisierte "
         f"Ueberholungen in einer DRS-Zone ({anteil:.1f}%)")

    ref_lap = ses.laps.pick_fastest()
    tel = ref_lap.get_telemetry().add_distance()
    ref_xy = tel[["X", "Y"]].to_numpy(dtype=float) / 10
    dist = tel["Distance"].to_numpy()

    print("\n[3/3] Saison-Scan ueber alle telemetriefaehigen Rennen 2024 "
         "(AUSBAUSTUFE) ...")
    ergebnisse = drs_saison_scan(2024)
    for row in ergebnisse.itertuples(index=False):
        print(f"      {row.gp:28s} {dict(row._asdict())}")
    print(f"\n      DRS-Anteil ueber {len(ergebnisse)} Strecken: "
         f"{ergebnisse['drs_pct'].min():.1f}-{ergebnisse['drs_pct'].max():.1f}%, "
         f"Median {ergebnisse['drs_pct'].median():.1f}%")

    print("\nDRITTE AUSBAUSTUFE: Abstand zur naechsten Bremszone, echt gegen "
         "Zufall ...")
    ref_bz = f1lab.driver_braking_zones(ses, str(ref_lap["Driver"]))
    strecke_m = float(dist.max())
    vorlauf_echt = f1lab.lead_distance_to_zone(
        orte["distance_m"], ref_bz["start_m"], strecke_m)
    rng = np.random.default_rng(42)
    zufall = rng.uniform(0, strecke_m, 20_000)
    vorlauf_zufall = f1lab.lead_distance_to_zone(
        zufall, ref_bz["start_m"], strecke_m)
    print(f"      Median Abstand: echt {np.median(vorlauf_echt):.0f}m gegen "
         f"Zufall {np.median(vorlauf_zufall):.0f}m")
    print(f"      Anteil < 200m:  echt {100 * (vorlauf_echt < 200).mean():.1f}% "
         f"gegen Zufall {100 * (vorlauf_zufall < 200).mean():.1f}%")
    print(f"      Anteil < 800m:  echt {100 * (vorlauf_echt < 800).mean():.1f}% "
         f"gegen Zufall {100 * (vorlauf_zufall < 800).mean():.1f}%")

    print("\nVIERTE AUSBAUSTUFE: DRS-Anteil 2018 (vor Ground-Effect) gegen "
         "2024 (danach) ...")
    ergebnisse_2018 = drs_saison_scan(2018)
    for row in ergebnisse_2018.itertuples(index=False):
        print(f"      {row.gp:28s} {dict(row._asdict())}")
    u, p_mw = mannwhitneyu(ergebnisse_2018["drs_pct"], ergebnisse["drs_pct"])
    print(f"\n      2018: n={len(ergebnisse_2018)} Strecken, Median "
         f"{ergebnisse_2018['drs_pct'].median():.1f}%")
    print(f"      2024: n={len(ergebnisse)} Strecken, Median "
         f"{ergebnisse['drs_pct'].median():.1f}%")
    print(f"      Mann-Whitney (je Strecke): p={p_mw:.3f}")
    drs_n_2018 = (ergebnisse_2018["lokalisiert"]
                 * ergebnisse_2018["drs_pct"] / 100).round().astype(int)
    drs_n_2024 = (ergebnisse["lokalisiert"]
                 * ergebnisse["drs_pct"] / 100).round().astype(int)
    tabelle_2x2 = [[drs_n_2018.sum(),
                    ergebnisse_2018["lokalisiert"].sum() - drs_n_2018.sum()],
                   [drs_n_2024.sum(),
                    ergebnisse["lokalisiert"].sum() - drs_n_2024.sum()]]
    _, p_chi, _, _ = chi2_contingency(tabelle_2x2)
    print(f"      Gepoolt: 2018 {drs_n_2018.sum()}/"
         f"{ergebnisse_2018['lokalisiert'].sum()} = "
         f"{100 * drs_n_2018.sum() / ergebnisse_2018['lokalisiert'].sum():.1f}%, "
         f"2024 {drs_n_2024.sum()}/{ergebnisse['lokalisiert'].sum()} = "
         f"{100 * drs_n_2024.sum() / ergebnisse['lokalisiert'].sum():.1f}%, "
         f"Chi-Quadrat p={p_chi:.2e}")

    print("\nGrafik ...")
    fig = plt.figure(figsize=(15, 21))
    gs = fig.add_gridspec(4, 1, height_ratios=[1.3, 1, 1, 1], hspace=0.5)
    zeichne_streckenkarte(fig.add_subplot(gs[0]), ref_xy, dist, zonen, orte)
    zeichne_saisonvergleich(fig.add_subplot(gs[1]), ergebnisse)
    zeichne_bremszonen_abstand(fig.add_subplot(gs[2]), vorlauf_echt,
                               vorlauf_zufall)
    zeichne_drs_ueber_zeit(fig.add_subplot(gs[3]), ergebnisse_2018, ergebnisse)
    fig.suptitle("Ueberholungen: in der DRS-Zone oder woanders?", x=0.09,
                ha="left", fontsize=16, color=FG, y=0.995)
    plt.tight_layout()
    path = OUT / "ueberholungen_drs_zone.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"\n      -> {path}")


if __name__ == "__main__":
    main()
