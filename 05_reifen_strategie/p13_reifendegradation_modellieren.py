"""schätzt die reifendegradation je stint per regression und sucht nach cliff-effekten im abbau"""
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

import f1lab
from f1lab.design import COMPOUND, FG, GRID, MUTED, SERIEN, matplotlib_stil

warnings.filterwarnings("ignore")

OUT = Path(__file__).parent / "out"
OUT.mkdir(exist_ok=True)

SEASON, EVENT, IDENT = 2024, "Bahrain", "R"
SCHWELLE, MIN_RUNDEN = 1.10, 6
MIN_RUNDEN_CLIFF = 10

plt.rcParams.update(matplotlib_stil())


def cliffs_suchen(laps: pd.DataFrame) -> list[dict]:
    """sucht cliffs per f1lab.find_cliff() je stint mit genug runden."""
    treffer = []
    for (drv, stint), g in laps.groupby(["Driver", "Stint"]):
        g = g.sort_values("TyreLife")
        if len(g) < MIN_RUNDEN_CLIFF:
            continue
        knick, links, rechts = f1lab.find_cliff(g["TyreLife"], g["corrected"])
        if knick is not None:
            assert rechts is not None  # find_cliff() setzt beide nur gemeinsam
            treffer.append({
                "driver": str(drv), "stint": int(stint),
                "compound": str(g["Compound"].iloc[0]), "n": len(g),
                "knick_tyrelife": knick, "slope_vorher": links.slope,
                "slope_danach": rechts.slope, "laps": g,
            })
    return treffer


def zeichne_kurven(ax, laps: pd.DataFrame) -> None:
    """zeichnet alle stints fuel-korrigiert, eingefärbt nach mischung."""
    for (_, _), g in laps.groupby(["Driver", "Stint"]):
        if len(g) < MIN_RUNDEN:
            continue
        g = g.sort_values("TyreLife")
        farbe = COMPOUND.get(str(g["Compound"].iloc[0]).upper(), MUTED)
        ax.plot(g["TyreLife"], g["corrected"], marker=".", ms=3, lw=0.8,
               alpha=0.55, color=farbe)
    vorhanden = {str(c).upper() for c in laps["Compound"].unique()}
    for mischung in ("SOFT", "MEDIUM", "HARD"):
        if mischung in vorhanden:
            ax.plot([], [], color=COMPOUND[mischung], lw=2, label=mischung.title())

    pb = laps[laps["IsPersonalBest"]]
    ax.scatter(pb["TyreLife"], pb["corrected"], marker="*", s=90,
              facecolors="none", edgecolors=FG, linewidths=1.1, zorder=3,
              label="Personal Best")

    ax.legend(loc="upper left", frameon=False, labelcolor=FG, fontsize=9)
    ax.set_xlabel("Reifenalter [Runden]")
    ax.set_ylabel("Fuel-korrigierte Rundenzeit [s]")
    ax.set_title(f"{EVENT} {SEASON} - Degradationskurven je Stint", loc="left",
                color=FG, fontsize=13, pad=10)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(alpha=0.3, linewidth=0.8, color=GRID)
    ax.set_axisbelow(True)


def zeichne_teams(ax, deg: pd.DataFrame) -> None:
    """zeigt die mittlere degradation je team nur für belastbare fits. die drei teams mit der stärksten degradation sind hervorgehoben."""
    rel = deg[deg["reliable"]]
    je_team = rel.groupby("team")["deg_s_per_lap"].mean().sort_values()
    farben = [SERIEN[1] if i >= len(je_team) - 3 else MUTED
             for i in range(len(je_team))]
    ax.barh(je_team.index, je_team.to_numpy(), color=farben, height=0.65)
    ax.set_xlabel("Mittlere Degradation [s/Runde]")
    ax.set_title("Team-Vergleich (nur belastbare Fits)", loc="left", color=FG,
                fontsize=13, pad=10)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="x", alpha=0.3, linewidth=0.8, color=GRID)
    ax.set_axisbelow(True)


def zeichne_cliff(ax, beispiel: dict) -> None:
    """zeichnet das stärkste cliff-beispiel als zwei geraden über die echten runden."""
    g = beispiel["laps"]
    ax.scatter(g["TyreLife"], g["corrected"], s=26, color=MUTED, zorder=2)
    knick = beispiel["knick_tyrelife"]
    vor = g[g["TyreLife"] <= knick]
    nach = g[g["TyreLife"] >= knick]
    xv = np.array([vor["TyreLife"].min(), knick])
    xn = np.array([knick, nach["TyreLife"].max()])
    b_vor = vor["corrected"].mean() - beispiel["slope_vorher"] * vor["TyreLife"].mean()
    b_nach = nach["corrected"].mean() - beispiel["slope_danach"] * nach["TyreLife"].mean()
    ax.plot(xv, beispiel["slope_vorher"] * xv + b_vor, color=SERIEN[0], lw=2.2,
           label=f"vor Knick: {beispiel['slope_vorher']:+.2f} s/Runde")
    ax.plot(xn, beispiel["slope_danach"] * xn + b_nach, color=SERIEN[1], lw=2.2,
           label=f"nach Knick: {beispiel['slope_danach']:+.2f} s/Runde")
    ax.axvline(knick, color=MUTED, lw=1, ls="--")
    ax.legend(loc="upper left", frameon=False, labelcolor=FG, fontsize=9)
    ax.set_xlabel("Reifenalter [Runden]")
    ax.set_ylabel("Fuel-korrigierte Rundenzeit [s]")
    ax.set_title(f"{beispiel['driver']} Stint {beispiel['stint']} "
                f"({beispiel['compound'].title()}) - deutlichster Cliff",
                loc="left", color=FG, fontsize=13, pad=10)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(alpha=0.3, linewidth=0.8, color=GRID)
    ax.set_axisbelow(True)


def zeichne_frischevergleich(ax, deg: pd.DataFrame) -> None:
    """vergleicht degradation frischer gegen wiederverwendeter reifen je mischung getrennt weil compound sonst als konfundierer wirkt. nur belastbare fits und nur mischungen mit fällen in beiden gruppen."""
    rel = deg[deg["reliable"]]
    tab = rel.groupby(["compound", "fresh"])["deg_s_per_lap"].agg(
        ["mean", "count"])
    mischungen = [c for c in tab.index.get_level_values(0).unique()
                 if {True, False} <= set(tab.loc[c].index)]
    if not mischungen:
        ax.text(0.5, 0.5, "keine Mischung mit beiden Gruppen", ha="center",
               va="center", color=MUTED)
        ax.axis("off")
        return
    x = np.arange(len(mischungen))
    w = 0.35
    for i, (fresh, label, farbe) in enumerate(
            ((False, "wiederverwendet", MUTED), (True, "frisch", SERIEN[0]))):
        werte = [tab.loc[(c, fresh), "mean"] if (c, fresh) in tab.index
                else np.nan for c in mischungen]
        ns = [tab.loc[(c, fresh), "count"] if (c, fresh) in tab.index
             else 0 for c in mischungen]
        ax.bar(x + (i - 0.5) * w, werte, width=w, color=farbe, label=label)
        for xi, n, v in zip(x + (i - 0.5) * w, ns, werte, strict=True):
            if n:
                ax.text(xi, v, f"n={n}", ha="center", va="bottom",
                       color=MUTED, fontsize=8)
    ax.set_xticks(x, [m.title() for m in mischungen])
    ax.set_ylabel("Mittlere Degradation [s/Runde]")
    ax.set_title("ZWEITE AUSBAUSTUFE: frisch vs. wiederverwendet",
                loc="left", color=FG, fontsize=13, pad=10)
    ax.legend(loc="upper left", frameon=False, labelcolor=FG, fontsize=9)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="y", alpha=0.3, linewidth=0.8, color=GRID)
    ax.set_axisbelow(True)


def main():
    f1lab.enable_cache()

    print(f"[1/5] {EVENT} {SEASON} {IDENT} laden ...")
    ses = f1lab.load(SEASON, EVENT, IDENT, telemetry=False)

    print("[2/5] Runden filtern, Fuel-Korrektur (VORGEHEN 1-2) ...")
    laps = f1lab.clean_laps(ses, threshold=SCHWELLE).copy()
    laps["sec"] = laps["LapTime"].dt.total_seconds()
    laps["corrected"] = f1lab.fuel_correct(
        laps["sec"], laps["LapNumber"], ses.total_laps)
    print(f"      {len(laps)} saubere Runden, {laps.groupby(['Driver', 'Stint']).ngroups} Stints")

    print("\n[3/5] Degradation je Stint (VORGEHEN 3) und Aggregation "
         "(VORGEHEN 4) ...")
    deg = f1lab.degradation(ses, threshold=SCHWELLE, min_laps=MIN_RUNDEN)
    print(f"      {len(deg)} Stints mit >= {MIN_RUNDEN} Runden, "
         f"{deg['reliable'].sum()} davon belastbar")
    je_compound = f1lab.degradation_by_compound(ses, threshold=SCHWELLE,
                                                min_laps=MIN_RUNDEN)
    print("\n      Mittel/Median je Mischung:")
    print(je_compound.to_string())
    print(deg[deg["reliable"]].groupby("compound")["deg_s_per_lap"]
         .median().round(4).rename("median").to_string())
    print("\n      Team-Mittel (belastbar):")
    print(deg[deg["reliable"]].groupby("team")["deg_s_per_lap"]
         .mean().round(4).sort_values().to_string())

    print(f"\n[4/5] AUSBAUSTUFE: Cliff-Suche (mind. {MIN_RUNDEN_CLIFF} Runden) ...")
    cliffs = cliffs_suchen(laps)
    kandidaten = [(d, s) for (d, s), g in laps.groupby(["Driver", "Stint"])
                 if len(g) >= MIN_RUNDEN_CLIFF]
    print(f"      {len(cliffs)}/{len(kandidaten)} Stints mit erkanntem Knick "
         f"({100 * len(cliffs) / len(kandidaten):.0f} %)")
    beispiel = max(cliffs, key=lambda c: c["slope_danach"] - c["slope_vorher"])
    print(f"      Deutlichster Fall: {beispiel['driver']} Stint "
         f"{beispiel['stint']} ({beispiel['compound']}) - Knick bei "
         f"Reifenalter {beispiel['knick_tyrelife']}, "
         f"{beispiel['slope_vorher']:+.3f} -> {beispiel['slope_danach']:+.3f} s/Runde")

    print("\n[5/5] ZWEITE AUSBAUSTUFE: FreshTyre und IsPersonalBest ...")
    rel = deg[deg["reliable"]]
    print(rel.groupby(["compound", "fresh"])["deg_s_per_lap"]
         .agg(["mean", "count"]).round(4).to_string())
    pb = laps[laps["IsPersonalBest"]]
    print(f"\n      {len(pb)} Personal-Best-Runden, Reifenalter dabei: "
         f"Median {pb['TyreLife'].median():.0f}, Mittel "
         f"{pb['TyreLife'].mean():.1f}, Max {pb['TyreLife'].max():.0f}")

    print("\nGrafik ...")
    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.2, 1], hspace=0.4, wspace=0.28)
    zeichne_kurven(fig.add_subplot(gs[0, :]), laps)
    zeichne_teams(fig.add_subplot(gs[1, 0]), deg)
    zeichne_cliff(fig.add_subplot(gs[1, 1]), beispiel)
    fig.suptitle(f"{ses.event['EventName']} {SEASON} - Reifendegradation",
                x=0.09, ha="left", fontsize=16, color=FG, y=0.995)
    path = OUT / "reifendegradation.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"\n      -> {path}")

    print("\nGrafik ZWEITE AUSBAUSTUFE ...")
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    zeichne_frischevergleich(ax2, deg)
    plt.tight_layout()
    path2 = OUT / "reifendegradation_frische.png"
    fig2.savefig(path2, dpi=130, bbox_inches="tight")
    plt.close(fig2)
    print(f"      -> {path2}")


if __name__ == "__main__":
    main()
