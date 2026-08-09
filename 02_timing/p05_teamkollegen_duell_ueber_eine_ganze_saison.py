"""
P05 - Teamkollegen-Duell ueber eine ganze Saison
================================================

Der fairste Leistungsvergleich in der F1: gleiches Auto, gleiche Strecke. Quali-Delta und Race-Delta ueber 24 Rennen.

Kategorie:   Timing & Rundenanalyse
Niveau:      Fortgeschritten
Aufwand:     4-5 h
Schwerpunkt: Datenanalyse, Strategie

WARUM DAS LOHNT
Teamintern ist genau das die Kernfrage bei Fahrerbewertung. Zeigt, dass du ueber Einzelrennen hinaus denkst und mit Batch-Verarbeitung umgehen kannst.

VORGEHEN
  1. Alle Rennen einer Saison iterieren (mit try/except pro Event)
  2. Pro Team die zwei Fahrer identifizieren
  3. Quali: bestes Q-Ergebnis vergleichen, Prozent-Delta bilden
  4. Rennen: bereinigte Median-Pace vergleichen (f1lab.pace_table, siehe P04)
  5. Heatmap Team x Rennen

GENUTZTE FASTF1-BAUSTEINE
  - fastf1.get_event_schedule
  - Laps.pick_accurate / pick_wo_box
  - Laps.groupby

AUSBAUSTUFE  [umgesetzt]
Elo-artiges Rating, das nach jedem Rennwochenende aktualisiert wird - ein
Duell je Session-Typ (Quali, Rennen), nicht je Runde: sonst wuerde ein Fahrer,
der 60 Runden lang eine Zehntelsekunde schneller ist, 60 Siege gutgeschrieben
bekommen statt einem. Die Elo-Mechanik selbst (elo_expected/elo_update) steckt
in f1lab.core, mit Tests auf bekannten Referenzwerten (Gleichstand -> 50%,
400 Punkte Vorsprung -> ~91%) - der Saison-Scan und das Verbuchen der Duelle
bleiben Skript, weil sie nur hier gebraucht werden (siehe P01: derselbe
Schnitt zwischen wiederverwendbarer Rechnung und einmaliger Orchestrierung).

Die Duell-Extraktion selbst (Quali-Bestzeit, nicht wegen Streckenlimits
gestrichen, dieselbe not_deleted_mask() wie in P03/P04; Rennen ueber die
bereinigte, treibstoffkorrigierte Race Pace) sitzt seit der App-Integration
in f1lab.teammate_duels() - dieselbe Funktion, die auch die Teamkollegen-
Seite im Dashboard nutzt, statt einer zweiten Kopie hier im Skript.

Saison 2024, 450 Duelle aus 24 Wochenenden: Verstappen fuehrt das Elo-Ranking
mit 1676 klar an (23 von 24 Quali-Duellen gegen Perez gewonnen), gefolgt von
Norris (1633) und Alonso (1623). Die Heatmap zeigt denselben Befund aus einer
zweiten Richtung: Red Bull Racing hat ueber die Saison den groessten
internen Quali-Abstand aller Teams - Dominanz eines Fahrers, keine
Team-Schwaeche.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")                      # kein Fenster, nur Dateien

import fastf1
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

import f1lab
from f1lab.design import FG, GRID, MUTED, RAMPE, SERIEN, matplotlib_stil

warnings.filterwarnings("ignore")
# Der Saison-Scan laedt 48 Sessions am Stueck - FastF1s Hintergrundabgleich
# mit dem Ergast-Spiegel laeuft dabei staendig ins Rate-Limit und faellt auf
# den eigenen Cache zurueck (unschaedlich, aber sehr laut ohne diese Zeile).
fastf1.set_log_level("ERROR")

OUT = Path(__file__).parent / "out"
OUT.mkdir(exist_ok=True)

YEAR = 2024
START_ELO = 1500.0
K = 24.0
RAMPE_CMAP = LinearSegmentedColormap.from_list("rampe", RAMPE)

plt.rcParams.update(matplotlib_stil())


def saison_scannen(year: int) -> pd.DataFrame:
    """Iteriert den ganzen Kalender, ueberspringt fehlende Sessions robust.

    Die Duell-Extraktion je Session steckt in f1lab.teammate_duels() - dort
    entscheidet die Funktion selbst anhand von session.name, ob Quali- oder
    Renn-Logik greift.
    """
    schedule = fastf1.get_event_schedule(year, include_testing=False)
    rows = []
    for _, ev in schedule.iterrows():
        rnd = int(ev["RoundNumber"])
        for ident in ("Q", "R"):
            try:
                ses = f1lab.load(year, rnd, ident)
                duelle = f1lab.teammate_duels(ses)
            except Exception as exc:
                print(f"      R{rnd:>2} {ident} uebersprungen: "
                     f"{type(exc).__name__}")
                continue
            for d in duelle:
                rows.append({"round": rnd, "event": ev["EventName"],
                            "session": ident, **d})
    return pd.DataFrame(rows)


def elo_verlauf(duelle: pd.DataFrame, k: float = K,
                start: float = START_ELO) -> pd.DataFrame:
    """Elo-Rating nach jedem Rennwochenende, ueber Quali- und Rennduelle."""
    rating: dict[str, float] = {}
    verlauf = []
    for rnd, gruppe in duelle.sort_values(["round", "session"]).groupby("round"):
        for _, d in gruppe.iterrows():
            ra = rating.get(d["a"], start)
            rb = rating.get(d["b"], start)
            rating[d["a"]], rating[d["b"]] = f1lab.elo_update(
                ra, rb, d["score_a"], k=k)
        for drv, elo in rating.items():
            verlauf.append({"round": rnd, "driver": drv, "elo": elo})
    return pd.DataFrame(verlauf)


def zeichne_heatmap(ax, quali: pd.DataFrame) -> None:
    """Team x Runde, Farbe = wie deutlich der schnellere Teamkollege vorn lag.

    Sortiert nach mittlerem Rueckstand - oben die Teams mit dem knappsten
    internen Duell, unten die mit dem klarsten. Auf dem 90.-Perzentil
    gekappt, sonst dominiert eine einzelne Ausreisser-Session (Q1-Ausfall
    gegen Pole-Nachbar) die ganze Farbskala.
    """
    pivot = quali.pivot_table(index="team", columns="round",
                              values="delta_pct", aggfunc="first")
    pivot = pivot.reindex(pivot.mean(axis=1).sort_values().index)

    im = ax.imshow(pivot.to_numpy(), aspect="auto", cmap=RAMPE_CMAP,
                   vmin=0, vmax=np.nanpercentile(pivot.to_numpy(), 90))
    ax.set_yticks(range(len(pivot.index)), pivot.index, fontsize=9)
    ax.set_xticks(range(0, len(pivot.columns), 2),
                  pivot.columns[::2], fontsize=8)
    ax.set_xlabel("Rennwoche")
    ax.set_title("Quali-Dominanz je Team und Rennen", loc="left", color=FG,
                fontsize=13, pad=12)
    for side in ax.spines.values():
        side.set_visible(False)

    cb = ax.figure.colorbar(im, ax=ax, pad=0.02)
    cb.set_label("Rueckstand des langsameren Teamkollegen [%]", color=MUTED,
                fontsize=9)
    cb.ax.tick_params(colors=MUTED, length=0)
    cb.outline.set_visible(False)


def zeichne_elo(ax, verlauf: pd.DataFrame, top_n: int = 3) -> None:
    """Elo-Verlauf aller Fahrer, die Top N am Saisonende farbig hervorgehoben.

    Dieselbe Loesung wie in P04 fuer das Problem "mehr Linien als
    unterscheidbare Farben": alle im Hintergrund duenn und grau, nur die
    Story-tragenden Linien in Akzentfarbe - MAX_SERIEN bleibt eingehalten.
    """
    pivot = verlauf.pivot(index="round", columns="driver", values="elo")
    top = pivot.iloc[-1].dropna().sort_values(ascending=False).index[:top_n]

    for drv in pivot.columns:
        if drv not in top:
            ax.plot(pivot.index, pivot[drv], color=GRID, linewidth=1.0,
                   alpha=0.8, zorder=1)

    for i, drv in enumerate(top):
        serie = pivot[drv].dropna()
        ax.plot(serie.index, serie, color=SERIEN[i], linewidth=2.2, zorder=3)
        ax.text(serie.index[-1] + 0.3, serie.iloc[-1], drv, color=SERIEN[i],
               fontsize=10, va="center", fontweight="bold")

    ax.axhline(START_ELO, color=MUTED, linewidth=0.8, linestyle=":")
    ax.set_xlabel("Rennwoche")
    ax.set_ylabel("Elo-Rating")
    ax.set_title(f"Teamkollegen-Elo, Top {top_n} am Saisonende", loc="left",
                color=FG, fontsize=13, pad=12)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(alpha=0.3, linewidth=0.8, color=GRID)
    ax.set_axisbelow(True)


def main():
    f1lab.enable_cache()

    print(f"[1/3] {YEAR} scannen (Quali + Rennen, aus dem Cache) ...")
    duelle = saison_scannen(YEAR)
    print(f"      {len(duelle)} Duelle aus {duelle['round'].nunique()} "
         f"Wochenenden ({(duelle['session'] == 'Q').sum()} Quali, "
         f"{(duelle['session'] == 'R').sum()} Rennen)")

    print("[2/3] Elo-Verlauf ...")
    verlauf = elo_verlauf(duelle)
    endstand = (verlauf.sort_values(["driver", "round"])
               .groupby("driver")["elo"].last().sort_values(ascending=False))
    print("\nEndstand (Top 10):")
    print(endstand.head(10).round(1).to_string())

    quali_siege = (duelle[duelle["session"] == "Q"].groupby("a").size()
                  .rename("Quali-Siege").sort_values(ascending=False))
    print("\nMeiste Quali-Siege gegen den Teamkollegen:")
    print(quali_siege.head(5).to_string())

    print("[3/3] Grafik ...")
    fig, ax = plt.subplots(1, 2, figsize=(15, 8),
                           gridspec_kw={"width_ratios": [1, 1.15]})
    zeichne_heatmap(ax[0], duelle[duelle["session"] == "Q"])
    zeichne_elo(ax[1], verlauf)
    fig.suptitle(f"{YEAR} - Teamkollegen-Duell ueber die Saison", x=0.125,
                ha="left", fontsize=16, color=FG, y=1.02)
    plt.tight_layout()
    path = OUT / "teamkollegen_duell.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"\n      -> {path}")


if __name__ == "__main__":
    main()
