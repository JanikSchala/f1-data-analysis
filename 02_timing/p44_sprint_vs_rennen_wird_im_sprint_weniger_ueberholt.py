"""vergleicht ueberholraten zwischen sprint und rennen desselben
wochenendes und wann im rennverlauf sie passieren"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")                      # nur dateien statt fenster

import fastf1
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

import f1lab
from f1lab.design import FG, GRID, MUTED, SERIEN, matplotlib_stil

warnings.filterwarnings("ignore")

OUT = Path(__file__).parent / "out"
OUT.mkdir(exist_ok=True)

SAISONS = (2023, 2024)

plt.rcParams.update(matplotlib_stil())


def ist_nass(session) -> bool:
    """regen ist ein eigener, viel groesserer zeit-/chaos-effekt als das
    rennformat selbst."""
    return session.laps["Compound"].isin(["INTERMEDIATE", "WET"]).any()


def sprint_namen_im_schedule(saison: int) -> list[str]:
    """findet sprint-wochenenden ueber EventFormat. das steht nur im rohen
    fastf1-schedule, nicht in f1lab.event_dimension()."""
    sched_roh = fastf1.get_event_schedule(saison, include_testing=False)
    return sched_roh[sched_roh["EventFormat"]
                     .str.contains("sprint", case=False,
                                  na=False)]["EventName"].tolist()


def sprint_wochenenden(saisons) -> pd.DataFrame:
    """sammelt alle trockenen sprint-wochenenden ueber mehrere saisons."""
    zeilen, nass = [], []
    for saison in saisons:
        for gp in sprint_namen_im_schedule(saison):
            try:
                ses_s = f1lab.load(saison, gp, "S", telemetry=False)
                ses_r = f1lab.load(saison, gp, "R", telemetry=False)
            except Exception:
                continue
            if ist_nass(ses_s) or ist_nass(ses_r):
                nass.append(f"{saison} {gp}")
                continue
            n_s = int(f1lab.overtakes_matrix(ses_s).values.sum())
            n_r = int(f1lab.overtakes_matrix(ses_r).values.sum())
            zeilen.append({
                "saison": saison, "gp": gp,
                "rate_sprint": n_s / ses_s.total_laps,
                "rate_rennen": n_r / ses_r.total_laps,
                "n_sprint": n_s, "n_rennen": n_r,
                "runden_sprint": ses_s.total_laps,
                "runden_rennen": ses_r.total_laps})
    return pd.DataFrame(zeilen), nass


def ueberholzeitpunkte(saisons) -> tuple[pd.Series, pd.Series]:
    """anteil der renndistanz zu dem ueberholungen passieren. gepoolt ueber
    alle trockenen sprint-wochenenden."""
    anteile_s, anteile_r = [], []
    for saison in saisons:
        for gp in sprint_namen_im_schedule(saison):
            try:
                ses_s = f1lab.load(saison, gp, "S", telemetry=False)
                ses_r = f1lab.load(saison, gp, "R", telemetry=False)
            except Exception:
                continue
            if ist_nass(ses_s) or ist_nass(ses_r):
                continue
            ev_s = f1lab.overtake_events(ses_s)
            ev_r = f1lab.overtake_events(ses_r)
            if not ev_s.empty:
                anteile_s.append(ev_s["lap"] / ses_s.total_laps)
            if not ev_r.empty:
                anteile_r.append(ev_r["lap"] / ses_r.total_laps)
    return (pd.concat(anteile_s, ignore_index=True),
            pd.concat(anteile_r, ignore_index=True))


def zeichne_vergleich(ax, df: pd.DataFrame) -> None:
    """gepaarter vergleich je wochenende, sortiert nach faktor."""
    df = df.copy()
    df["faktor"] = df["rate_rennen"] / df["rate_sprint"]
    df = df.sort_values("faktor")
    labels = [f"{row.gp.replace(' Grand Prix', '')} {row.saison}"
             for row in df.itertuples()]
    y = np.arange(len(df))
    ax.barh(y - 0.19, df["rate_sprint"], height=0.36, color=SERIEN[0],
           label="Sprint")
    ax.barh(y + 0.19, df["rate_rennen"], height=0.36, color=SERIEN[1],
           label="Rennen")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    x_max = df[["rate_sprint", "rate_rennen"]].max(axis=1).max()
    for yi, faktor in zip(y, df["faktor"]):
        ax.text(x_max * 1.08, yi, f"{faktor:.2f}x", va="center", fontsize=8,
               color=MUTED)
    ax.set_xlim(0, x_max * 1.22)
    ax.set_xlabel("Ueberholungen pro Runde (nur gruene Flagge)")
    ax.set_title("Sprint gegen Rennen, je trockenes Sprint-Wochenende "
                "2023+2024", loc="left", color=FG, fontsize=13, pad=10)
    ax.legend(loc="lower right", bbox_to_anchor=(1.0, 1.01), ncol=2,
             frameon=False, labelcolor=FG, fontsize=9)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="x", alpha=0.3, linewidth=0.8, color=GRID)
    ax.set_axisbelow(True)


def zeichne_zeitpunkte(ax, anteile_s: pd.Series, anteile_r: pd.Series) -> None:
    """zeigt wann im rennverlauf die ueberholungen passieren."""
    bins = np.linspace(0, 1, 11)
    ax.hist(anteile_s, bins=bins, density=True, alpha=0.6, color=SERIEN[0],
           label=f"Sprint (n={len(anteile_s)})")
    ax.hist(anteile_r, bins=bins, density=True, alpha=0.6, color=SERIEN[1],
           label=f"Rennen (n={len(anteile_r)})")
    ax.set_xlabel("Renndistanz, zu der die Ueberholung passiert")
    ax.set_ylabel("Dichte")
    ax.set_title("AUSBAUSTUFE: wann im Rennverlauf wird ueberholt?",
                loc="left", color=FG, fontsize=13, pad=10)
    ax.legend(loc="upper center", frameon=False, labelcolor=FG, fontsize=9)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="y", alpha=0.3, linewidth=0.8, color=GRID)
    ax.set_axisbelow(True)


def main():
    f1lab.enable_cache()

    print(f"[1-3/5] Sprint-Wochenenden {SAISONS}, nasse raus "
         "(VORGEHEN 1-3) ...")
    df, nass = sprint_wochenenden(SAISONS)
    print(f"      {len(df)} trockene Sprint-Wochenenden, "
         f"{len(nass)} wegen Regen ausgeschlossen: {nass}")
    for row in df.itertuples():
        faktor = row.rate_rennen / row.rate_sprint
        print(f"      {row.saison} {row.gp}: Sprint {row.rate_sprint:.2f}/"
             f"Runde, Rennen {row.rate_rennen:.2f}/Runde ({faktor:.2f}x)")

    print("\n[4/5] Gepaarter Test (VORGEHEN 4) ...")
    diff = df["rate_rennen"] - df["rate_sprint"]
    test = wilcoxon(diff)
    print(f"      Median Sprint: {df['rate_sprint'].median():.2f}/Runde")
    print(f"      Median Rennen: {df['rate_rennen'].median():.2f}/Runde")
    print(f"      Wilcoxon p={test.pvalue:.4f}")

    print("\n[5/5] Je-Wochenende-Konsistenz (VORGEHEN 5) ...")
    hoeher = (df["rate_rennen"] > df["rate_sprint"]).sum()
    print(f"      Rennen > Sprint in {hoeher}/{len(df)} Wochenenden")

    print("\nAUSBAUSTUFE: wann im Rennverlauf wird ueberholt? ...")
    anteile_s, anteile_r = ueberholzeitpunkte(SAISONS)
    for name, a in (("Sprint", anteile_s), ("Rennen", anteile_r)):
        print(f"      {name}: erstes Drittel={( a < 1 / 3).mean():.1%}, "
             f"mittleres={((a >= 1 / 3) & (a < 2 / 3)).mean():.1%}, "
             f"letztes={(a >= 2 / 3).mean():.1%}")

    print("\nGrafiken speichern ...")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 11),
                                   gridspec_kw={"height_ratios": [1.3, 1]})
    fig.patch.set_facecolor(plt.rcParams["figure.facecolor"])
    zeichne_vergleich(ax1, df)
    zeichne_zeitpunkte(ax2, anteile_s, anteile_r)
    fig.tight_layout()
    fig.savefig(OUT / "p44_sprint_vs_rennen.png", dpi=140)
    plt.close(fig)
    print(f"      {OUT / 'p44_sprint_vs_rennen.png'}")


if __name__ == "__main__":
    main()
