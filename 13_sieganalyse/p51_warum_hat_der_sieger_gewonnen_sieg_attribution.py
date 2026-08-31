"""warum hat der sieger gewonnen: startvorteil, strategie, ausfall des rivalen, safety car oder eine echte ueberholung - je rennen"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")                      # kein Fenster, nur Dateien

import fastf1
import matplotlib.pyplot as plt
import pandas as pd

import f1lab
from f1lab.design import FG, GRID, MUTED, SERIEN, matplotlib_stil

warnings.filterwarnings("ignore")
fastf1.set_log_level("ERROR")

OUT = Path(__file__).parent / "out"
OUT.mkdir(exist_ok=True)

SAISON = 2024
BEISPIEL_RENNEN = "São Paulo Grand Prix"   # VER von P17 zum Sieg, Safety-Car-Wende

GRUENDE_REIHENFOLGE = ["Start-Vorteil", "Erkaempft auf der Strecke",
                       "Strategie/Boxenstopp", "Safety-Car-Wende",
                       "Ausfall des Rivalen", "Einbruch des Rivalen",
                       "Ungeklaert"]

ERAS = [
    (2018, 2021, "V6-Hybrid-Turbo"),
    (2022, 2026, "Ground Effect"),
]


def saison_scan(saison: int) -> pd.DataFrame:
    """f1lab.sieg_attribution() je rennen einer saison, komplett aus dem
    lokalen cache (keine der beteiligten f1lab-funktionen braucht
    telemetrie oder netzwerk)."""
    inv = f1lab.cached_sessions()
    rennen = sorted(inv[(inv["season"] == saison) & (inv["ident"] == "R")]
                    ["event"].unique())
    zeilen = []
    for gp in rennen:
        try:
            ses = f1lab.load(saison, gp, "R", telemetry=False)
            r = f1lab.sieg_attribution(ses)
        except Exception:
            continue
        r["event"] = gp
        r["season"] = saison
        zeilen.append(r)
    return pd.DataFrame(zeilen)


def zeichne_verteilung(ax, daten: pd.DataFrame) -> None:
    counts = (daten["grund"].value_counts()
             .reindex(GRUENDE_REIHENFOLGE).dropna().astype(int))
    farben = [SERIEN[0] if g == counts.idxmax() else MUTED for g in counts.index]
    ax.barh(counts.index[::-1], counts.to_numpy()[::-1],
           color=farben[::-1], height=0.6)
    for i, v in enumerate(counts.to_numpy()[::-1]):
        ax.text(v + 0.15, i, str(v), va="center", color=FG, fontsize=9)
    ax.set_xlabel("Anzahl Rennen")
    ax.set_title(f"Saison {SAISON}: warum hat der Sieger gewonnen?",
                loc="left", color=FG, fontsize=12, pad=10)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="x", alpha=0.3, linewidth=0.8, color=GRID)
    ax.set_axisbelow(True)


def zeichne_beispiel(ax, session, attribution: dict) -> None:
    pos = f1lab.position_progression(session)
    sieger = attribution["sieger"]
    phasen = f1lab.track_status_phases(session)
    neutral = phasen[phasen["label"].isin(["safety car", "vsc"])]
    for p in neutral.itertuples():
        ax.axvspan(p.lap_start, p.lap_end, color=SERIEN[1], alpha=0.15)

    for drv in pos.columns:
        if drv == sieger:
            continue
        ax.plot(pos.index, pos[drv], color=MUTED, lw=0.9, alpha=0.4)
    ax.plot(pos.index, pos[sieger], color=SERIEN[0], lw=2.2, label=sieger)

    runde = attribution["entscheidende_runde"]
    if runde is not None:
        ax.axvline(runde, color=FG, lw=1, ls="--")
        ax.annotate(f"Runde {runde}\n{attribution['grund']}",
                   xy=(runde, pos.loc[runde, sieger]),
                   xytext=(runde + 2, 6), color=FG, fontsize=9,
                   arrowprops={"arrowstyle": "-", "color": MUTED, "lw": 0.8})

    ax.invert_yaxis()
    ax.set_xlabel("Runde")
    ax.set_ylabel("Position")
    ax.set_title(f"Beispiel: {BEISPIEL_RENNEN} {SAISON} - {sieger} von "
                f"P{attribution['startplatz']}", loc="left", color=FG,
                fontsize=12, pad=10)
    ax.legend(loc="upper right", frameon=False, labelcolor=FG, fontsize=9)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(alpha=0.3, linewidth=0.8, color=GRID)
    ax.set_axisbelow(True)


def main():
    f1lab.enable_cache()
    plt.rcParams.update(matplotlib_stil())

    print(f"[1/3] Sieg-Attribution fuer alle Rennen der Saison {SAISON} "
         "sammeln (VORGEHEN 1) ...")
    daten = saison_scan(SAISON)
    print(f"      {len(daten)} Rennen ausgewertet")
    print(daten[["event", "sieger", "startplatz", "entscheidende_runde",
                "alter_fuehrender", "grund"]].to_string(index=False))

    print("\n[2/3] Verteilung der Gruende (VORGEHEN 2) ...")
    verteilung = daten["grund"].value_counts()
    print(verteilung.to_string())

    print(f"\n[3/3] Beispielrennen im Detail: {BEISPIEL_RENNEN} (VORGEHEN 3) ...")
    ses_beispiel = f1lab.load(SAISON, BEISPIEL_RENNEN, "R", telemetry=False)
    attribution = f1lab.sieg_attribution(ses_beispiel)
    print(f"      {attribution['sieger']} startete P{attribution['startplatz']}, "
         f"uebernahm die Fuehrung endgueltig in Runde "
         f"{attribution['entscheidende_runde']} von {attribution['alter_fuehrender']} "
         f"- Grund: {attribution['grund']}")
    print(f"      Fuehrungsanteil {attribution['fuehrungsanteil']:.1%}, "
         f"Pace-Rang {attribution['pace_rang']}, "
         f"Abstand zu P2 {attribution['abstand_p2_s']}s")

    print("\nGrafik ...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.5),
                                   gridspec_kw={"width_ratios": [2, 3]})
    zeichne_verteilung(ax1, daten)
    zeichne_beispiel(ax2, ses_beispiel, attribution)
    fig.suptitle("Sieg-Attribution: was hat den Ausschlag gegeben?",
                x=0.06, ha="left", fontsize=15, color=FG, y=1.02)
    plt.tight_layout()
    path = OUT / "sieg_attribution.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"      -> {path}")

    print("\nAUSBAUSTUFE: aendert sich die Mischung ueber die Regelaeren? ...")
    alle = []
    for start, ende, _name in ERAS:
        for jahr in range(start, ende + 1):
            teil = saison_scan(jahr)
            if not teil.empty:
                alle.append(teil)
    gesamt = pd.concat(alle, ignore_index=True) if alle else pd.DataFrame()
    if gesamt.empty:
        print("      keine weiteren Saisons im Cache.")
    else:
        def era_von(jahr: int) -> str:
            for start, ende, name in ERAS:
                if start <= jahr <= ende:
                    return name
            return "unbekannt"
        gesamt["era"] = gesamt["season"].apply(era_von)
        mix = (gesamt.groupby("era")["grund"]
              .value_counts(normalize=True).unstack().fillna(0) * 100)
        mix = mix.reindex(columns=GRUENDE_REIHENFOLGE).dropna(axis=1, how="all")
        print(f"      {len(gesamt)} Rennen ueber {gesamt['season'].nunique()} "
             "Saisons, Anteil je Grund und Aera [%]:")
        print(mix.round(1).to_string())


if __name__ == "__main__":
    main()
