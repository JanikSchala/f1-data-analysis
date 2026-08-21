"""ermittelt die undercut-erfolgsquote aus echten paarweisen rivalen-duellen über eine ganze saison"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")                      # kein fenster, nur dateien

import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import binomtest

import f1lab
from f1lab.design import FG, GRID, MUTED, SERIEN, matplotlib_stil

warnings.filterwarnings("ignore")

OUT = Path(__file__).parent / "out"
OUT.mkdir(exist_ok=True)

SAISON = 2024
SANITY_EVENT = (2024, "Spain", "R")
FENSTER_SENSITIVITAET = [(2, 2), (3, 2), (5, 2), (3, 1), (3, 3)]

plt.rcParams.update(matplotlib_stil())


def saison_scan(saison: int) -> pd.DataFrame:
    """sammelt alle undercut-duelle einer saison."""
    schedule = f1lab.event_dimension([saison])
    alle = []
    for _, row in schedule.iterrows():
        try:
            ses = f1lab.load(saison, int(row["round"]), "R", telemetry=False)
        except Exception:
            continue
        duelle = f1lab.undercut_duels(ses)
        if not duelle.empty:
            duelle = duelle.copy()
            duelle["gp"] = row["event_name"]
            alle.append(duelle)
    return pd.concat(alle, ignore_index=True) if alle else pd.DataFrame()


def zeichne_verteilung(ax, je_rennen: pd.DataFrame) -> None:
    """erfolgsquote je rennen, sortiert."""
    e = je_rennen.sort_values("rate")
    farben = [SERIEN[1] if n >= 5 else MUTED for n in e["n"]]
    ax.barh(e["gp"], e["rate"] * 100, color=farben, height=0.65)
    ax.axvline(50, color=MUTED, lw=1, ls="--")
    for y, (n, rate) in enumerate(zip(e["n"], e["rate"])):
        ax.text(rate * 100 + 1.5, y, f"n={n}", va="center", fontsize=7,
                color=MUTED)
    ax.set_xlabel("Undercut-Erfolgsquote [%] (gestrichelt = 50%-Erwartung)")
    ax.set_title(f"Undercut-Erfolgsquote je Rennen, Saison {SAISON} "
                f"(rot = n>=5 Duelle)", loc="left", color=FG, fontsize=13,
                pad=10)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="x", alpha=0.3, linewidth=0.8, color=GRID)
    ax.set_axisbelow(True)


def zeichne_gesamt(ax, n: int, erfolge: int, ci) -> None:
    """gesamtquote mit konfidenzintervall gegen 50%."""
    rate = erfolge / n * 100
    ax.barh([0], [rate], color=SERIEN[1], height=0.5)
    ax.errorbar([rate], [0], xerr=[[rate - ci.low * 100], [ci.high * 100 - rate]],
               color=FG, capsize=6, lw=1.5, fmt="none")
    ax.axvline(50, color=MUTED, lw=1.5, ls="--", label="50%-Erwartung")
    ax.set_xlim(0, 60)
    ax.set_yticks([])
    ax.set_xlabel("Undercut-Erfolgsquote [%], 95%-KI")
    ax.set_title(f"Saison {SAISON} gesamt: {erfolge}/{n} Duelle erfolgreich "
                f"({rate:.1f}%)", loc="left", color=FG, fontsize=13, pad=10)
    ax.legend(loc="lower right", frameon=False, labelcolor=FG, fontsize=9)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="x", alpha=0.3, linewidth=0.8, color=GRID)
    ax.set_axisbelow(True)


def main():
    f1lab.enable_cache()

    print(f"[1/4] Sanity-Check: {SANITY_EVENT[1]} {SANITY_EVENT[0]} "
         f"{SANITY_EVENT[2]} (VORGEHEN 1) ...")
    ses = f1lab.load(*SANITY_EVENT, telemetry=False)
    einzel = f1lab.undercut_duels(ses)
    print(einzel.to_string(index=False))
    print(f"      n={len(einzel)}, erfolge={einzel['erfolg'].sum()}, "
         f"Quote={einzel['erfolg'].mean():.1%}")

    print(f"\n[2/4] Saison-Scan {SAISON}, alle Rennen (VORGEHEN 2) ...")
    duelle = saison_scan(SAISON)
    n = len(duelle)
    erfolge = int(duelle["erfolg"].sum())
    print(f"      {n} Duelle ueber {duelle['gp'].nunique()} Rennen gesammelt")

    print("\n[3/4] Statistik (VORGEHEN 3) ...")
    test = binomtest(erfolge, n, 0.5)
    ci = test.proportion_ci(confidence_level=0.95)
    print(f"      Erfolgsquote: {erfolge}/{n} = {erfolge / n:.1%}")
    print(f"      95%-KI: [{ci.low:.1%}, {ci.high:.1%}]")
    print(f"      Binomialtest gegen 50%: p={test.pvalue:.2e}")

    print("\n[4/4] Robustheit gegen fenster/nachlauf (AUSBAUSTUFE) ...")
    for fenster, nachlauf in FENSTER_SENSITIVITAET:
        rows = []
        for _, row in f1lab.event_dimension([SAISON]).iterrows():
            try:
                s = f1lab.load(SAISON, int(row["round"]), "R", telemetry=False)
            except Exception:
                continue
            d = f1lab.undercut_duels(s, fenster=fenster, nachlauf=nachlauf)
            if not d.empty:
                rows.append(d)
        voll = pd.concat(rows, ignore_index=True)
        n_s, e_s = len(voll), int(voll["erfolg"].sum())
        p_s = binomtest(e_s, n_s, 0.5).pvalue
        print(f"      fenster={fenster} nachlauf={nachlauf}: n={n_s} "
             f"Quote={e_s / n_s:.1%} p={p_s:.1e}")

    print("\nGrafiken speichern ...")
    je_rennen = (duelle.groupby("gp")
                       .agg(n=("erfolg", "size"), erfolge=("erfolg", "sum"))
                       .reset_index())
    je_rennen["rate"] = je_rennen["erfolge"] / je_rennen["n"]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 12),
                                   gridspec_kw={"height_ratios": [1, 3]})
    fig.patch.set_facecolor(plt.rcParams["figure.facecolor"])
    zeichne_gesamt(ax1, n, erfolge, ci)
    zeichne_verteilung(ax2, je_rennen)
    fig.tight_layout()
    fig.savefig(OUT / "p42_undercut_erfolgsquote.png", dpi=140)
    plt.close(fig)
    print(f"      {OUT / 'p42_undercut_erfolgsquote.png'}")


if __name__ == "__main__":
    main()
