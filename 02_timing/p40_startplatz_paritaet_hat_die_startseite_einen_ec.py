"""testet je strecke ob ungerade oder gerade startplaetze einen echten
positionsvorteil nach runde 1 haben"""
from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")                      # nur dateien statt fenster

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ttest_ind

import f1lab
from f1lab.design import FG, GRID, MUTED, SERIEN, matplotlib_stil

warnings.filterwarnings("ignore")

OUT = Path(__file__).parent / "out"
OUT.mkdir(exist_ok=True)

MIN_SAISONS = 4
KONSISTENZ_STRECKEN = ["Saudi Arabian Grand Prix", "Australian Grand Prix",
                       "Belgian Grand Prix", "Dutch Grand Prix",
                       "Mexico City Grand Prix"]

plt.rcParams.update(matplotlib_stil())


def zeichne_strecken(ax, ergebnisse: pd.DataFrame) -> None:
    """paritaets-effekt je strecke. signifikante strecken hervorgehoben."""
    e = ergebnisse.sort_values("diff")
    farben = [SERIEN[1] if p < 0.05 else MUTED for p in e["p"]]
    ax.barh(e["gp"], e["diff"], color=farben, height=0.65)
    ax.axvline(0, color=MUTED, lw=1)
    ax.set_xlabel("Ungerade minus gerade Startplaetze [Positionen nach Runde 1]")
    ax.set_title(f"Paritaets-Effekt je Strecke ({len(e)} Strecken, "
                f"mind. {MIN_SAISONS} Saisons; rot = p<0.05)", loc="left",
                color=FG, fontsize=13, pad=10)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="x", alpha=0.3, linewidth=0.8, color=GRID)
    ax.set_axisbelow(True)


def zeichne_konsistenz(ax, saison_diffs: pd.DataFrame) -> None:
    """zeigt ob die richtung des effekts in jeder einzelnen saison haelt."""
    strecken = KONSISTENZ_STRECKEN
    breite = 0.8 / max(len(saison_diffs["season"].unique()), 1)
    for i, s in enumerate(sorted(saison_diffs["season"].unique())):
        werte = saison_diffs[saison_diffs["season"] == s].set_index("gp")
        werte = werte.reindex(strecken)
        x = np.arange(len(strecken)) + i * breite
        ax.bar(x, werte["diff"], width=breite, color=SERIEN[i % len(SERIEN)],
              label=str(s))
    ax.axhline(0, color=MUTED, lw=1)
    ax.set_xticks(np.arange(len(strecken)) + 0.35)
    ax.set_xticklabels([s.replace(" Grand Prix", "") for s in strecken])
    ax.set_ylabel("Ungerade minus gerade [Positionen]")
    ax.set_title("AUSBAUSTUFE: haelt die Richtung in jeder einzelnen Saison?",
                loc="left", color=FG, fontsize=13, pad=10)
    ax.legend(loc="upper right", frameon=False, labelcolor=FG, fontsize=8,
             ncol=3)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="y", alpha=0.3, linewidth=0.8, color=GRID)
    ax.set_axisbelow(True)


def main():
    f1lab.enable_cache()

    print("[1/3] Startplatz gegen Runde-1-Position, ganzer Cache (VORGEHEN "
         "1-2, keine Telemetrie noetig) ...")
    inv = f1lab.cached_sessions()
    rennen = inv[inv["ident"] == "R"][["season", "event"]].drop_duplicates()

    alle = []
    saison_diffs = []
    for _, r in rennen.iterrows():
        saison, gp = int(r["season"]), r["event"]
        try:
            ses = f1lab.load(saison, gp, "R", telemetry=False)
            m = f1lab.grid_lap1_positions(ses)
        except Exception:
            continue
        time.sleep(0.2)      # schont ergast/jolpica gegen das rate-limit
        if m.empty:
            continue
        m["parity"] = np.where(m["grid"].astype(int) % 2 == 0,
                               "gerade", "ungerade")
        m["gp"] = gp
        m["season"] = saison
        alle.append(m)
        if gp in KONSISTENZ_STRECKEN:
            ung = m.loc[m["parity"] == "ungerade", "gewinn"]
            ger = m.loc[m["parity"] == "gerade", "gewinn"]
            saison_diffs.append({"gp": gp, "season": saison,
                                 "diff": ung.mean() - ger.mean()})

    df = pd.concat(alle, ignore_index=True)
    print(f"      {len(df)} Starts (ohne Boxenstarts) ueber "
         f"{df[['season', 'gp']].drop_duplicates().shape[0]} Rennen, "
         f"{df['gp'].nunique()} Strecken")

    print(f"\n[2/3] t-Test je Strecke mit mindestens {MIN_SAISONS} Saisons "
         "(VORGEHEN 3) ...")
    zeilen = []
    for gp, g in df.groupby("gp"):
        if g["season"].nunique() < MIN_SAISONS:
            continue
        ung = g.loc[g["parity"] == "ungerade", "gewinn"]
        ger = g.loc[g["parity"] == "gerade", "gewinn"]
        t, p = ttest_ind(ung, ger)
        zeilen.append({"gp": gp, "saisons": g["season"].nunique(),
                       "n": len(g), "diff": round(ung.mean() - ger.mean(), 2),
                       "p": round(p, 4)})
    ergebnisse = pd.DataFrame(zeilen).sort_values("p")
    print(ergebnisse.to_string(index=False))

    signifikant = ergebnisse[ergebnisse["p"] < 0.05]
    bonferroni = 0.05 / len(ergebnisse)
    print(f"\n      {len(signifikant)}/{len(ergebnisse)} Strecken mit p<0.05 "
         f"(Zufallserwartung: {0.05 * len(ergebnisse):.1f})")
    print(f"      Bonferroni-Schwelle bei {len(ergebnisse)} Tests: "
         f"{bonferroni:.4f} - ueberlebt: "
         f"{(ergebnisse['p'] < bonferroni).sum()}/{len(ergebnisse)}")

    print("\n[3/3] Saison-Konsistenz der fuenf auffaelligsten Strecken "
         "(AUSBAUSTUFE) ...")
    sd = pd.DataFrame(saison_diffs)
    for gp in KONSISTENZ_STRECKEN:
        werte = sd.loc[sd["gp"] == gp, "diff"]
        gleiche_richtung = (werte > 0).all() or (werte < 0).all()
        print(f"      {gp:28s} {len(werte)} Saisons, gleiche Richtung: "
             f"{gleiche_richtung}  ({[round(v, 2) for v in werte]})")

    print("\nGrafik ...")
    fig, ax = plt.subplots(2, 1, figsize=(13, 13))
    zeichne_strecken(ax[0], ergebnisse)
    zeichne_konsistenz(ax[1], sd)
    fig.suptitle("Startplatz-Paritaet: hat die Startseite einen echten "
                "Effekt?", x=0.09, ha="left", fontsize=16, color=FG, y=0.995)
    plt.tight_layout()
    path = OUT / "startplatz_paritaet.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"\n      -> {path}")


if __name__ == "__main__":
    main()
