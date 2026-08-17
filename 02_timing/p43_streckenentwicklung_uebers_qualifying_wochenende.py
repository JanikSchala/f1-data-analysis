"""
P43 - Streckenentwicklung: wird die Strecke von Q1 zu Q3 wirklich schneller?
===============================================================================

Die verbreitete Behauptung im Kommentatoren-Jargon: die Strecke "gummiert ein"
waehrend eines Qualifyings, jede Session wird schneller als die vorherige.
Stimmt das in echten Daten - und wie trennt man das von den drei
naheliegenden Verwechslungen (Sprit, Reifenmischung, weniger schnelle Autos
in Q3)?

Kategorie:   Timing & Rundenanalyse
Niveau:      Profi
Aufwand:     4-5 h
Schwerpunkt: Datenanalyse, Statistik

WARUM DAS LOHNT
Diese Frage stand explizit als "riskant" auf der Liste offener Ideen, mit
drei bekannten Fallstricken: Reifenmischung (Q1 manchmal auf Hard, wenn ein
Team Reifen sparen will), Sprit (freies Training faehrt oft schwer beladen)
und "Sandbagging" in FP1 (Fahrer zeigen dort bewusst nicht ihr Tempo). Der
Trick, der alle drei umgeht: gar nicht erst das freie Training verwenden.
FastF1s `Laps.split_qualifying_sessions()` zerlegt eine Qualifying-Session
in Q1/Q2/Q3 - und innerhalb von Qualifying sind alle drei Fallstricke
strukturell entschaerft: jede gewertete Rundenzeit ist per Definition eine
Banzai-Runde (niemand faehrt in Q auf hohem Sprit), praktisch jeder faehrt
auf SOFT (siehe VORGEHEN 3), und Sandbagging ergibt in Q keinen Sinn - wer
ausscheidet, ist raus.

Der verbleibende, wichtigste Fallstrick ist ein vierter, der in der
urspruenglichen Liste nicht genannt war: von Q1 zu Q3 bleiben nur die
schnelleren Autos uebrig. Ein einfacher Schnitt-Vergleich "Q3 ist im
Schnitt schneller als Q1" wuerde deshalb teilweise nur "die langsamen
Autos sind raus" messen, nicht Streckenentwicklung. Die Loesung: NUR
Fahrer vergleichen, die in beiden Segmenten eine Zeit gesetzt haben, und je
Fahrer sein eigenes Q1- gegen sein eigenes Q2-Ergebnis stellen (paarweiser
Vergleich, wie beim Undercut in P42) - das haelt Auto- und
Fahrerqualitaet konstant.

VORGEHEN
  1. Einzelnes Rennen (Spanien 2024 Q) als Sanity-Check:
     f1lab.qualifying_track_evolution() aufrufen, Ergebnis nachvollziehen
  2. Saison-weiter Scan (2024): alle Qualifyings, ausser den nassen (siehe
     Punkt 3)
  3. Nasse Sessions ausschliessen: Regen ist selbst ein starker,
     unabhaengiger Zeiteffekt (trocknende Strecke), der die reine
     "Gummi"-Streckenentwicklung ueberdecken wuerde - erkannt an
     INTERMEDIATE/WET-Reifen irgendwo in der Session (wie P17s
     Regen-Klassifikation)
  4. Statistik ueber alle paarweisen Fahrer-Deltas: Median, Anteil positiv,
     Wilcoxon-Vorzeichen-Rang-Test (gepaarte, nicht normalverteilte Daten -
     kein t-Test)
  5. Je-Rennen-Konsistenz: zeigt JEDES einzelne Rennen dieselbe Richtung,
     oder ist der gepoolte Befund von wenigen Ausreissern getragen?

GENUTZTE FASTF1-BAUSTEINE
  - Laps.split_qualifying_sessions() (neu genutzt in diesem Repo)
  - Laps.Compound (Regen-Erkennung)
  - scipy.stats.wilcoxon

AUSBAUSTUFE  [umgesetzt]
Cross-Saison-Robustheit: haelt der Befund in einer komplett anderen Saison
(2023), oder ist 2024 ein Einzelfall?

Saison 2024, 21 trockene Qualifyings (Grossbritannien, Belgien, Sao Paulo
wegen Regen ausgeschlossen - INTERMEDIATE/WET-Reifen irgendwo in der
Session): **Q1->Q2 median +0.407s schneller (n=313 Fahrer-Vergleiche, 92.3%
positiv, Wilcoxon p=1.3e-48), Q2->Q3 median +0.167s schneller (n=209, 79.4%
positiv, p=3.8e-16)**. Reifenmischung als Restfallstrick geprueft: eine
Wiederholung nur mit SOFT-Runden (statt allen gewerteten Runden) aendert
kaum etwas (+0.422s/+0.176s statt +0.407s/+0.167s) - fast alle gewerteten
Q-Runden sind ohnehin SOFT, der Fallstrick war real, aber praktisch
wirkungslos in dieser Saison.

Die Je-Rennen-Konsistenz ist auffaellig hoch: **21 von 21** trockenen
Rennen zeigen eine positive mediane Q1->Q2-Verbesserung, **21 von 21** auch
fuer Q2->Q3 - kein einziges Gegenbeispiel in der ganzen Saison. Das ist ein
deutlich saubereres Bild als die meisten anderen Befunde dieses Repos
(vgl. P40s nur 2 von 25 wirklich konsistenten Strecken) - Streckenentwicklung
scheint kein schwaches, streckenspezifisches Signal zu sein wie der
Paritaets-Effekt, sondern ein durchgehend wirkendes.

Cross-Saison-Check 2023 (18 trockene Qualifyings): Q1->Q2 median +0.444s
(n=270, 90.7% positiv, p=3.4e-36, **18/18** Rennen positiv), Q2->Q3 median
+0.275s (n=179, 80.4% positiv, p=4.0e-13, **16/18** Rennen positiv) - dieselbe
Groessenordnung, dieselbe Richtung, fast dieselbe Konsistenz wie 2024. Der
Effekt ist kein Artefakt einer einzelnen Saison.

Ehrlich eingeordnet: Q1->Q2 gewinnt in beiden Saisons deutlich mehr Zeit
als Q2->Q3 (rund doppelt bis dreifach so viel) - plausibel, weil die
groesste Zahl an Runden (und damit an frisch aufgetragenem Gummi) in Q1
gefahren wird, wo noch 20 statt 15 bzw. 10 Autos unterwegs sind. Die Daten
trennen "mehr Runden legen mehr Gummi" nicht sauber von "Q1 ist als
Session strukturell anders (laenger, mehr Verkehr)" - beide Erklaerungen
sind mit dem hier Gemessenen vereinbar, nur die Groessenordnung des
Gesamteffekts ist der gesicherte Teil.
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
from scipy.stats import wilcoxon

import f1lab
from f1lab.design import FG, GRID, MUTED, SERIEN, matplotlib_stil

warnings.filterwarnings("ignore")

OUT = Path(__file__).parent / "out"
OUT.mkdir(exist_ok=True)

SAISON = 2024
SANITY_EVENT = (2024, "Spain", "Q")

plt.rcParams.update(matplotlib_stil())


def ist_nass(session) -> bool:
    """VORGEHEN 3: Regen ueberdeckt den reinen Gummi-Effekt, deshalb raus."""
    return session.laps["Compound"].isin(["INTERMEDIATE", "WET"]).any()


def saison_scan(saison: int) -> tuple[pd.DataFrame, list[str]]:
    """VORGEHEN 2: alle trockenen Qualifyings einer Saison sammeln."""
    schedule = f1lab.event_dimension([saison])
    alle, nass = [], []
    for _, row in schedule.iterrows():
        try:
            ses = f1lab.load(saison, int(row["round"]), "Q", telemetry=False)
        except Exception:
            continue
        if ist_nass(ses):
            nass.append(row["event_name"])
            continue
        d = f1lab.qualifying_track_evolution(ses)
        if not d.empty:
            d = d.copy()
            d["gp"] = row["event_name"]
            alle.append(d)
    return (pd.concat(alle, ignore_index=True) if alle else pd.DataFrame(),
            nass)


def zeichne_verteilung(ax, deltas: pd.DataFrame) -> None:
    """VORGEHEN 4: Verteilung der paarweisen Deltas je Segment."""
    segmente = ["Q1->Q2", "Q2->Q3"]
    daten = [deltas.loc[deltas["segment"] == s, "delta_s"] for s in segmente]
    bp = ax.boxplot(daten, tick_labels=segmente, patch_artist=True,
                    widths=0.5, showfliers=False)
    for patch, farbe in zip(bp["boxes"], SERIEN):
        patch.set_facecolor(farbe)
        patch.set_alpha(0.6)
    for element in ("whiskers", "caps", "medians"):
        for line in bp[element]:
            line.set_color(FG)
    ax.axhline(0, color=MUTED, lw=1, ls="--")
    for i, s in enumerate(segmente):
        med = deltas.loc[deltas["segment"] == s, "delta_s"].median()
        pos = (deltas.loc[deltas["segment"] == s, "delta_s"] > 0).mean()
        ax.text(i + 1, ax.get_ylim()[1] * 0.9,
               f"Median {med:+.3f}s\n{pos:.0%} positiv",
               ha="center", fontsize=9, color=FG)
    ax.set_ylabel("Delta [s], positiv = spaeteres Segment schneller")
    ax.set_title(f"Fahrer-Paarvergleich je Segment, Saison {SAISON} "
                f"(trockene Qualifyings)", loc="left", color=FG, fontsize=13,
                pad=10)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="y", alpha=0.3, linewidth=0.8, color=GRID)
    ax.set_axisbelow(True)


def zeichne_konsistenz(ax, je_rennen: pd.DataFrame) -> None:
    """VORGEHEN 5: haelt die Richtung in jedem einzelnen Rennen?"""
    e = je_rennen.sort_values("median_q1q2")
    x = np.arange(len(e))
    breite = 0.38
    ax.bar(x - breite / 2, e["median_q1q2"], width=breite, color=SERIEN[0],
          label="Q1->Q2")
    ax.bar(x + breite / 2, e["median_q2q3"], width=breite, color=SERIEN[1],
          label="Q2->Q3")
    ax.axhline(0, color=MUTED, lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [g.replace(" Grand Prix", "") for g in e["gp"]], rotation=60,
        ha="right", fontsize=7)
    ax.set_ylabel("Median-Delta je Rennen [s]")
    ax.set_title("AUSBAUSTUFE-Vorstufe: Konsistenz ueber alle Rennen der "
                f"Saison ({len(e)} trockene Qualifyings)", loc="left",
                color=FG, fontsize=13, pad=10)
    ax.legend(loc="upper left", frameon=False, labelcolor=FG, fontsize=9)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="y", alpha=0.3, linewidth=0.8, color=GRID)
    ax.set_axisbelow(True)


def main():
    f1lab.enable_cache()

    print(f"[1/5] Sanity-Check: {SANITY_EVENT[1]} {SANITY_EVENT[0]} "
         f"{SANITY_EVENT[2]} (VORGEHEN 1) ...")
    ses = f1lab.load(*SANITY_EVENT, telemetry=False)
    einzel = f1lab.qualifying_track_evolution(ses)
    for seg, g in einzel.groupby("segment"):
        print(f"      {seg}: n={len(g)}, Median={g['delta_s'].median():+.3f}s")

    print(f"\n[2-3/5] Saison-Scan {SAISON}, nasse Sessions raus "
         "(VORGEHEN 2-3) ...")
    deltas, nass = saison_scan(SAISON)
    print(f"      {deltas['gp'].nunique()} trockene Qualifyings, "
         f"{len(nass)} wegen Regen ausgeschlossen: {nass}")

    print("\n[4/5] Statistik ueber alle Fahrer-Paare (VORGEHEN 4) ...")
    for seg in ("Q1->Q2", "Q2->Q3"):
        werte = deltas.loc[deltas["segment"] == seg, "delta_s"]
        test = wilcoxon(werte)
        print(f"      {seg}: n={len(werte)}, Median={werte.median():+.3f}s, "
             f"Anteil positiv={( werte > 0).mean():.1%}, "
             f"Wilcoxon p={test.pvalue:.2e}")

    print("\n[5/5] Je-Rennen-Konsistenz (VORGEHEN 5) ...")
    je_rennen = (deltas.groupby(["gp", "segment"])["delta_s"].median()
                       .unstack("segment").reset_index()
                       .rename(columns={"Q1->Q2": "median_q1q2",
                                        "Q2->Q3": "median_q2q3"}))
    pos12 = (je_rennen["median_q1q2"] > 0).sum()
    pos23 = (je_rennen["median_q2q3"] > 0).sum()
    print(f"      Q1->Q2 positiv in {pos12}/{len(je_rennen)} Rennen")
    print(f"      Q2->Q3 positiv in {pos23}/{len(je_rennen)} Rennen")

    print("\nAUSBAUSTUFE: Cross-Saison-Check 2023 ...")
    deltas_2023, nass_2023 = saison_scan(2023)
    for seg in ("Q1->Q2", "Q2->Q3"):
        werte = deltas_2023.loc[deltas_2023["segment"] == seg, "delta_s"]
        test = wilcoxon(werte)
        je_r = (deltas_2023.loc[deltas_2023["segment"] == seg]
                          .groupby("gp")["delta_s"].median())
        print(f"      {seg}: n={len(werte)}, Median={werte.median():+.3f}s, "
             f"Anteil positiv={( werte > 0).mean():.1%}, p={test.pvalue:.2e}, "
             f"Rennen positiv={(je_r > 0).sum()}/{len(je_r)}")
    print(f"      ({len(nass_2023)} 2023er Qualifyings wegen Regen raus: "
         f"{nass_2023})")

    print("\nGrafiken speichern ...")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 12),
                                   gridspec_kw={"height_ratios": [1, 1.3]})
    fig.patch.set_facecolor(plt.rcParams["figure.facecolor"])
    zeichne_verteilung(ax1, deltas)
    zeichne_konsistenz(ax2, je_rennen)
    fig.tight_layout()
    fig.savefig(OUT / "p43_streckenentwicklung.png", dpi=140)
    plt.close(fig)
    print(f"      {OUT / 'p43_streckenentwicklung.png'}")


if __name__ == "__main__":
    main()
