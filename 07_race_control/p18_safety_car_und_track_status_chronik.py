"""baut safety-car- und track-status-phasen als rundenintervalle und misst ihre auswirkung auf die feldstreckung"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")                      # kein fenster, nur dateien

import fastf1
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import f1lab
from f1lab.design import FG, GRID, MUTED, PHASE, SERIEN, matplotlib_stil

warnings.filterwarnings("ignore")
fastf1.set_log_level("ERROR")   # saison-scan lädt 24 sessions, das wäre sonst sehr geschwätzig

OUT = Path(__file__).parent / "out"
OUT.mkdir(exist_ok=True)

SEASON, EVENT, IDENT = 2024, "Canada", "R"
SAISON_RENNEN = [
    "Bahrain", "Saudi Arabia", "Australia", "Japan", "China", "Miami",
    "Emilia Romagna", "Monaco", "Canada", "Spain", "Austria", "Britain",
    "Hungary", "Belgium", "Netherlands", "Italy", "Azerbaijan", "Singapore",
    "United States", "Mexico", "Brazil", "Las Vegas", "Qatar", "Abu Dhabi",
]
FENSTER_DAVOR = 3     # runden vor SC-beginn, die noch als "kurz davor" zählen

plt.rcParams.update(matplotlib_stil())


def _position_bei(pos: pd.Series, drv: str, lap: float) -> float:
    try:
        v = pos.loc[(drv, lap)]
        return float(v.iloc[0]) if isinstance(v, pd.Series) else float(v)
    except KeyError:
        return float("nan")


def saison_scan() -> pd.DataFrame:
    """positionswechsel von der runde vor dem eigenen stopp bis zum phasenende, getrennt nach stopp-zeitpunkt relativ zur neutralisation."""
    zeilen = []
    for gp in SAISON_RENNEN:
        try:
            ses = f1lab.load(SEASON, gp, "R", telemetry=False)
            phasen = f1lab.track_status_phases(ses)
        except Exception:
            continue
        neutral = phasen[phasen["label"].isin(["safety car", "vsc"])]
        if neutral.empty:
            continue
        laps = ses.laps
        pos = laps.set_index(["Driver", "LapNumber"])["Position"]

        for p in neutral.itertuples():
            for drv in laps["Driver"].unique():
                davor = laps[(laps["Driver"] == drv)
                            & (laps["LapNumber"] >= p.lap_start - FENSTER_DAVOR)
                            & (laps["LapNumber"] < p.lap_start)
                            & (laps["PitInTime"].notna())]
                waehrend = laps[(laps["Driver"] == drv)
                                & (laps["LapNumber"] >= p.lap_start)
                                & (laps["LapNumber"] <= p.lap_end)
                                & (laps["PitInTime"].notna())]
                if len(davor) > 0:
                    klasse = "kurz davor"
                    stopp_runde = int(davor.iloc[0]["LapNumber"])
                elif len(waehrend) > 0:
                    klasse = "waehrend SC/VSC"
                    stopp_runde = int(waehrend.iloc[0]["LapNumber"])
                else:
                    continue
                pv = _position_bei(pos, drv, stopp_runde - 1)
                pn = _position_bei(pos, drv, p.lap_end)
                if np.isnan(pv) or np.isnan(pn):
                    continue
                zeilen.append({"gp": gp, "driver": drv, "klasse": klasse,
                               "delta_pos": pv - pn})
    return pd.DataFrame(zeilen)


def zeichne_chronik(ax, phasen: pd.DataFrame, spread: pd.Series) -> None:
    """track-status-bänder über der feldstreckung."""
    for p in phasen.itertuples():
        if p.label == "gruen":
            continue
        ax.axvspan(p.lap_start, max(p.lap_end, p.lap_start + 0.3),
                  color=PHASE.get(p.label, MUTED), alpha=0.35, lw=0)
    ax.plot(spread.index, spread.to_numpy(), color=FG, lw=1.2)
    for lbl in ("gelb", "safety car"):
        ax.plot([], [], color=PHASE[lbl], lw=8, alpha=0.35, label=lbl)
    ax.legend(loc="upper right", frameon=False, labelcolor=FG, fontsize=9)
    ax.set_xlabel("Runde")
    ax.set_ylabel("Feldstreckung P1-Letzter [s]")
    ax.set_title(f"{EVENT} {SEASON} - Track-Status und Feldstreckung",
                loc="left", color=FG, fontsize=12, pad=10)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(alpha=0.3, linewidth=0.8, color=GRID)
    ax.set_axisbelow(True)


def zeichne_kompaktierung(ax, komp: pd.DataFrame) -> None:
    x = np.arange(len(komp))
    w = 0.35
    ax.bar(x - w / 2, komp["baseline_s"], width=w, color=MUTED,
          label="Baseline (3 gruene Runden davor)")
    ax.bar(x + w / 2, komp["minimum_s"], width=w, color=SERIEN[1],
          label="Minimum waehrend SC")
    ax.set_xticks(x, [f"Runde {int(r.start)}-{int(r.ende)}"
                      for r in komp.itertuples()])
    for i, r in enumerate(komp.itertuples()):
        ax.text(i, max(r.baseline_s, r.minimum_s) + 5,
               f"-{r.kompaktierung_pct:.0f}%", ha="center", color=FG, fontsize=9)
    ax.set_ylabel("Feldstreckung [s]")
    ax.set_title("Kompaktierung: Minimum gegen Baseline", loc="left",
                color=FG, fontsize=12, pad=10)
    ax.legend(loc="upper right", frameon=False, labelcolor=FG, fontsize=9)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="y", alpha=0.3, linewidth=0.8, color=GRID)
    ax.set_axisbelow(True)


def zeichne_deployment_sektoren(ax, sek: pd.DataFrame) -> None:
    """feldverteilung über die drei timing-sektoren im moment jeder SC-deployment-meldung."""
    tab = (sek.dropna(subset=["sector"])
          .groupby(["time", "sector"]).size().unstack(fill_value=0))
    tab.columns = [f"Sektor {int(c)}" for c in tab.columns]
    x = np.arange(len(tab))
    w = 0.25
    farben = [SERIEN[0], SERIEN[1], SERIEN[2]]
    for i, col in enumerate(tab.columns):
        ax.bar(x + (i - 1) * w, tab[col], width=w, color=farben[i % 3],
              label=col)
    ax.set_xticks(x, [f"Deployment {i + 1}\n({str(t).split('days ')[-1][:8]})"
                      for i, t in enumerate(tab.index)])
    ax.set_ylabel("Fahrer")
    ax.set_title("ZWEITE AUSBAUSTUFE: Feld-Sektorverteilung bei SC-Deployment",
                loc="left", color=FG, fontsize=12, pad=10)
    ax.legend(loc="upper right", frameon=False, labelcolor=FG, fontsize=9)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="y", alpha=0.3, linewidth=0.8, color=GRID)
    ax.set_axisbelow(True)


def zeichne_ausbaustufe(ax, scan: pd.DataFrame) -> None:
    g = scan.groupby("klasse")["delta_pos"].agg(["mean", "count"])
    g = g.reindex(["kurz davor", "waehrend SC/VSC"])
    farben = [SERIEN[1], SERIEN[0]]
    ax.bar(g.index, g["mean"], color=farben, width=0.5)
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.set_ylim(g["mean"].min() * 1.2, 0.22)
    for i, (_idx, row) in enumerate(g.iterrows()):
        ax.text(i, 0.03, f"n={int(row['count'])}", ha="center",
               va="bottom", color=MUTED, fontsize=9)
    ax.set_ylabel("mittlere Positionsaenderung\n(Runde vor Stopp -> SC-Ende)")
    ax.set_title(f"Saison-Scan {SEASON}: Stopp-Zeitpunkt relativ zu SC/VSC",
                loc="left", color=FG, fontsize=12, pad=10)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="y", alpha=0.3, linewidth=0.8, color=GRID)
    ax.set_axisbelow(True)


def main():
    f1lab.enable_cache()

    print(f"[1/4] {EVENT} {SEASON} {IDENT} laden (VORGEHEN 1-4) ...")
    ses = f1lab.load(SEASON, EVENT, IDENT, telemetry=False)
    phasen = f1lab.track_status_phases(ses)
    neutral = phasen[phasen["label"].isin(["safety car", "vsc"])]
    print(phasen[phasen["label"] != "gruen"]
         [["label", "lap_start", "lap_end", "duration_s"]]
         .round(1).to_string(index=False))

    print("\n[2/4] Feldstreckung: Baseline gegen Kompaktierungs-Minimum ...")
    spread = f1lab.field_spread(ses)
    komp = f1lab.sc_compaction(neutral, spread)
    print(komp.round(1).to_string(index=False))
    gruene_runden = phasen.loc[phasen["label"] == "gruen", "lap_start"]
    referenz = spread.reindex(gruene_runden).dropna().median()
    print(f"      Gruenphasen-Referenz (Median ueber Runden am Beginn "
         f"jeder gruenen Phase): {referenz:.1f} s")

    print("\n[3/4] ZWEITE AUSBAUSTUFE: Sector*SessionTime bei SC-Deployment ...")
    sek = f1lab.sc_deployment_sectors(ses)
    if sek.empty:
        print("      keine Safety-Car-Deployment-Meldungen in dieser Session")
    else:
        print(sek.groupby(["time", "sector"]).size().unstack(fill_value=0)
             .rename(columns=lambda c: f"Sektor {int(c)}").to_string())

    print(f"\n[4/4] AUSBAUSTUFE: Saison-Scan {SEASON} ({len(SAISON_RENNEN)} "
         f"Rennen) ...")
    scan = saison_scan()
    print(f"      {scan['gp'].nunique()} Rennen mit SC/VSC, "
         f"{len(scan)} auswertbare Faelle")
    print(scan.groupby("klasse")["delta_pos"].agg(["mean", "median", "count"])
         .round(2).to_string())

    print("\nGrafik ...")
    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1], hspace=0.4, wspace=0.28)
    zeichne_chronik(fig.add_subplot(gs[0, :]), phasen, spread)
    zeichne_kompaktierung(fig.add_subplot(gs[1, 0]), komp)
    zeichne_ausbaustufe(fig.add_subplot(gs[1, 1]), scan)
    fig.suptitle(f"{ses.event['EventName']} {SEASON} - Safety-Car-Chronik",
                x=0.09, ha="left", fontsize=16, color=FG, y=0.995)
    path = OUT / "safety_car_chronik.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"\n      -> {path}")

    if not sek.empty:
        print("\nGrafik ZWEITE AUSBAUSTUFE ...")
        fig2, ax2 = plt.subplots(figsize=(9, 6))
        zeichne_deployment_sektoren(ax2, sek)
        plt.tight_layout()
        path2 = OUT / "safety_car_deployment_sektoren.png"
        fig2.savefig(path2, dpi=130, bbox_inches="tight")
        plt.close(fig2)
        print(f"      -> {path2}")


if __name__ == "__main__":
    main()
