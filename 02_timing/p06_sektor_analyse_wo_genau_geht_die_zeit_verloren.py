"""
P06 - Sektor-Analyse: Wo genau geht die Zeit verloren?
======================================================

Sektorzeiten und Speed-Traps zerlegen die Rundenzeit in Aussagen ueber Abtrieb, Topspeed und Traktion.

Kategorie:   Timing & Rundenanalyse
Niveau:      Einsteiger
Aufwand:     2 h
Schwerpunkt: Datenanalyse, Strategie

WARUM DAS LOHNT
Sektor-Deltas sind die schnellste Diagnose fuer Auto-Charakteristik. Perfekt, um analytisches Denken zu demonstrieren, ohne dass es kompliziert wird.

VORGEHEN
  1. Beste Sektorzeit je Fahrer ermitteln
  2. Theoretisch beste Runde je Fahrer bilden (Summe der Bestsektoren)
  3. Delta zur tatsaechlich gefahrenen Bestzeit = ungenutztes Potenzial
  4. Speed-Traps gegen Sektorzeiten plotten

GENUTZTE FASTF1-BAUSTEINE
  - Laps.pick_fastest
  - Sector1Time/2/3
  - SpeedI1/I2/FL/ST
  - Lap.get_telemetry (Mini-Sektoren)
  - matplotlib LineCollection

AUSBAUSTUFE  [umgesetzt]
Mini-Sektor-Analyse: die Runde per Telemetrie-Distanz in 25 gleich lange
Abschnitte teilen, fuer jeden die Zeit interpolieren (nicht die Geschwindigkeit
mitteln - Zeit ist additiv ueber einen Abschnitt, eine gemittelte
Geschwindigkeit waere bei ungleich verteilten Messpunkten verzerrt) und die
Strecke danach einfaerben, wer den Abschnitt gewonnen hat.

Bewusst nur die drei schnellsten Fahrer statt des ganzen Feldes: mehr als
MAX_SERIEN=3 unterscheidbare Farben verbietet f1lab.design, und inhaltlich
ist "wer dominiert welchen Abschnitt" ohnehin nur unter echten Anwaertern
eine Diagnose - Rang 15 gegen Rang 3 in einem Mini-Sektor sagt vor allem
etwas ueber die Ausgangslage, nicht uebers Auto.

Monaco 2024 Q, LEC/PIA/SAI (die drei schnellsten): 9 der 25 Mini-Sektoren
gehen an Piastri, je 8 an Leclerc und Sainz - trotz 0.25s Rueckstand auf
Leclercs Gesamtzeit. Verteilte Staerke schlaegt sich nicht eins zu eins in
der Rundenzeit nieder, weil die Abschnitte unterschiedlich lang sind und
Piastris Vorteile in kuerzeren Abschnitten liegen.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")                      # kein Fenster, nur Dateien

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection

import f1lab
from f1lab.design import FG, GRID, SERIEN, matplotlib_stil

warnings.filterwarnings("ignore")

OUT = Path(__file__).parent / "out"
OUT.mkdir(exist_ok=True)

SEASON, EVENT, IDENT = 2024, "Monaco", "Q"
N_MINISEKTOREN = 25

plt.rcParams.update(matplotlib_stil())


def sektor_tabelle(laps: pd.DataFrame) -> pd.DataFrame:
    """VORGEHEN 1-3: theoretisch beste Runde gegen tatsaechliche Bestzeit."""
    sec = (laps.groupby("Driver")[["Sector1Time", "Sector2Time", "Sector3Time"]]
           .min().apply(lambda s: s.dt.total_seconds()))
    sec["theoretisch"] = sec.sum(axis=1)
    sec["real"] = laps.groupby("Driver")["LapTime"].min().dt.total_seconds()
    sec["ungenutzt"] = sec["real"] - sec["theoretisch"]
    return sec.sort_values("real")


def mini_sektoren(laps: pd.DataFrame, fahrer: list[str], n: int = N_MINISEKTOREN
                  ) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], np.ndarray, np.ndarray]:
    """Zerlegt die Runde in n gleich lange Distanz-Abschnitte und ermittelt
    je Abschnitt, wer dort am wenigsten Zeit gebraucht hat."""
    telemetrie = {}
    for drv in fahrer:
        lap = laps[laps["Driver"] == drv].pick_fastest()
        tel = lap.get_telemetry()
        telemetrie[drv] = pd.DataFrame({
            "Distance": tel["Distance"].to_numpy(dtype=float),
            "sec": tel["Time"].dt.total_seconds().to_numpy(),
            "X": tel["X"].to_numpy(dtype=float),
            "Y": tel["Y"].to_numpy(dtype=float),
        })

    strecke = min(t["Distance"].max() for t in telemetrie.values())
    edges = np.linspace(0, strecke, n + 1)

    dauer = pd.DataFrame({
        drv: np.diff(np.interp(edges, t["Distance"], t["sec"]))
        for drv, t in telemetrie.items()
    })
    gewinner = dauer.idxmin(axis=1).to_numpy()
    return dauer, telemetrie, edges, gewinner


def zeichne_potenzial(ax, sec: pd.DataFrame) -> None:
    p = sec.sort_values("ungenutzt")
    ax.barh(p.index, p["ungenutzt"], color=SERIEN[0], height=0.65)
    ax.invert_yaxis()
    ax.set_xlabel("Ungenutztes Potenzial [s]")
    ax.set_title("Bestzeit minus Summe der Bestsektoren", loc="left",
                color=FG, fontsize=13, pad=12)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="x", alpha=0.35, linewidth=0.8, color=GRID)
    ax.set_axisbelow(True)


def zeichne_speedtraps(ax, laps: pd.DataFrame, sec: pd.DataFrame) -> None:
    """VORGEHEN 4: Speed-Traps gegen Sektorzeiten - Topspeed gegen Rueckstand."""
    speeds = laps.groupby("Driver")[["SpeedI1", "SpeedI2", "SpeedFL", "SpeedST"]].max()
    d = sec.join(speeds)
    rueckstand = d["real"] - d["real"].min()
    ax.scatter(d["SpeedST"], rueckstand, s=42, color=SERIEN[0], zorder=3)

    for drv in {d["real"].idxmin(), d["real"].idxmax(),
               d["SpeedST"].idxmax(), d["SpeedST"].idxmin()}:
        ax.annotate(drv, (d.loc[drv, "SpeedST"], rueckstand.loc[drv]),
                   textcoords="offset points", xytext=(6, 4), color=FG,
                   fontsize=9)

    ax.set_xlabel("Speed Trap Start-Ziel-Gerade [km/h]")
    ax.set_ylabel("Rueckstand auf die Bestzeit [s]")
    ax.set_title("Topspeed gegen Rueckstand", loc="left", color=FG,
                fontsize=13, pad=12)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(alpha=0.35, linewidth=0.8, color=GRID)
    ax.set_axisbelow(True)


def zeichne_streckenkarte(ax, telemetrie: dict[str, pd.DataFrame],
                          edges: np.ndarray, gewinner: np.ndarray,
                          fahrer: list[str]) -> None:
    farbe = dict(zip(fahrer, SERIEN, strict=True))
    referenz = telemetrie[fahrer[0]]
    x = referenz["X"].to_numpy()
    y = referenz["Y"].to_numpy()
    dist = referenz["Distance"].to_numpy()

    punkte = np.column_stack([x, y]).reshape(-1, 1, 2)
    segmente = np.concatenate([punkte[:-1], punkte[1:]], axis=1)
    mitte = (dist[:-1] + dist[1:]) / 2
    bin_idx = np.clip(np.digitize(mitte, edges) - 1, 0, len(gewinner) - 1)
    farben = [farbe[gewinner[i]] for i in bin_idx]

    lc = LineCollection(segmente, colors=farben, linewidths=5)
    ax.add_collection(lc)
    ax.set_xlim(x.min() - 200, x.max() + 200)
    ax.set_ylim(y.min() - 200, y.max() + 200)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(f"Mini-Sektoren (n={len(gewinner)}), schnellster Fahrer je "
                f"Abschnitt - Referenzlinie {fahrer[0]}", loc="left",
                color=FG, fontsize=13, pad=12)

    for drv in fahrer:
        n_gewonnen = int((gewinner == drv).sum())
        ax.scatter([], [], color=farbe[drv], s=80, label=f"{drv} ({n_gewonnen})")
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 0.03), frameon=False,
             labelcolor=FG, fontsize=10, ncol=3)


def main():
    f1lab.enable_cache()

    print(f"[1/4] {EVENT} {SEASON} {IDENT} laden (mit Telemetrie) ...")
    ses = f1lab.load(SEASON, EVENT, IDENT, telemetry=True)
    laps = ses.laps.pick_accurate().pick_wo_box()

    print("[2/4] Sektor-Zerlegung ...")
    sec = sektor_tabelle(laps)
    print(sec[["real", "theoretisch", "ungenutzt"]].round(3).to_string())

    print(f"\n[3/4] Mini-Sektoren (Top 3, n={N_MINISEKTOREN}) ...")
    top3 = sec.index[:3].tolist()
    dauer, telemetrie, edges, gewinner = mini_sektoren(laps, top3)
    for drv in top3:
        print(f"      {drv}: {int((gewinner == drv).sum())} Mini-Sektoren gewonnen")

    arr = np.sort(dauer.to_numpy(), axis=1)
    lueck = arr[:, 1] - arr[:, 0]
    engster, weitester = int(np.argmin(lueck)), int(np.argmax(lueck))
    print(f"      engster Mini-Sektor: #{engster + 1} "
         f"({lueck[engster] * 1000:.0f} ms Unterschied)")
    print(f"      groesster Vorsprung: #{weitester + 1} "
         f"({lueck[weitester] * 1000:.0f} ms Unterschied)")

    print("[4/4] Grafik ...")
    fig = plt.figure(figsize=(15, 10.5))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.3], hspace=0.38, wspace=0.28)
    zeichne_potenzial(fig.add_subplot(gs[0, 0]), sec)
    zeichne_speedtraps(fig.add_subplot(gs[0, 1]), laps, sec)
    zeichne_streckenkarte(fig.add_subplot(gs[1, :]), telemetrie, edges,
                          gewinner, top3)

    fig.suptitle(f"{EVENT} {SEASON} {IDENT} - Sektor-Analyse", x=0.09,
                ha="left", fontsize=16, color=FG, y=0.98)
    path = OUT / "sektor_analyse.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"\n      -> {path}")


if __name__ == "__main__":
    main()
