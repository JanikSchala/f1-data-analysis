"""
P42 - Undercut-Erfolgsquote: echte, paarweise Rivalen-Duelle
================================================================

Wie oft gewinnt ein Undercut wirklich - nicht in der Simulation (P15), nicht
im Modell (P35), sondern gegen den echten Rivalen, den ein Boxenstopp
tatsaechlich angreift?

Kategorie:   Reifen & Strategie
Niveau:      Profi
Aufwand:     4-5 h
Schwerpunkt: Datenanalyse, Statistik

WARUM DAS LOHNT
P15 simuliert den Undercut-Gewinn aus Reifenmodellen, P35 rechnet die
optimale Stoppstrategie gegen die Uhr - beide bewusst ohne Rivalen, siehe
P35s eigener Hinweis auf Verkehr als blinden Fleck (jetzt in P41
aufgegriffen). Diese Frage ist eine andere: nicht "wie viel Zeit gewinnt
ein Undercut modelliert", sondern "wie oft setzt sich ein ECHTER
Undercut-Versuch am Ende auch durch". Das laesst sich direkt aus
Positionsdaten beantworten, ohne Reifenmodell - aber nur, wenn man den
richtigen Rivalen vergleicht.

Ein erster Anlauf dazu wurde bereits einmal verworfen (siehe CLAUDE.md,
Nachtrag zu P38): Position vor einem Boxenstopp gegen Position im ganzen
Feld ein paar Runden danach verglichen, ergab fuer ALLE getesteten Stopps
ein negatives Delta - kein Befund, sondern eine Methodenluecke. Wer stoppt,
verliert gegen das GANZE Feld automatisch Positionen, weil nicht-stoppende
Autos in der Zwischenzeit einfach weiterfahren; das misst die
Boxengassenzeit, nicht den Undercut. Ein Undercut ist per Definition ein
Duell gegen EINEN Rivalen (den davor liegenden Verfolger), nicht gegen 19
andere Autos gleichzeitig.

VORGEHEN
  1. Einzelnes Rennen (Spanien 2024 R) als Sanity-Check: alle Undercut-
     Duelle auflisten, Ergebnis von Hand nachvollziehbar
  2. Saison-weiter Scan (2024, alle 24 Rennen): f1lab.undercut_duels() je
     Rennen aufrufen, alle Duelle sammeln
  3. Erfolgsquote ueber die ganze Saison, 95%-Konfidenzintervall
     (Wilson-Score via scipy.stats.binomtest), Binomialtest gegen die
     naive 50%-Erwartung ("Undercut und Verteidigung sind gleich oft
     erfolgreich")
  4. Verteilung je Rennen zeigen: schwankt die Quote nur um den
     Mittelwert, oder gibt es Rennen mit strukturell anderer Erfolgsquote?

GENUTZTE FASTF1-BAUSTEINE
  - Laps.Position (pivotiert je Runde) fuer die Positionsverlaeufe
  - Laps.PitInTime fuer den Stoppzeitpunkt
  - scipy.stats.binomtest fuer den Signifikanztest samt Konfidenzintervall

Definition (siehe f1lab.session.undercut_duels() Docstring fuer die volle
Herleitung): fuer jeden Boxenstopp (Fahrer A, Runde L) ist der Rivale der
Fahrer, der zu Rundenbeginn genau eine Position vor A lag - nur gezaehlt,
wenn dieser Rivale nicht selbst in derselben Runde stoppt (sonst kein
Undercut-Versuch, sondern ein gemeinsamer Stopp) und innerhalb von
``fenster`` Runden selbst an die Box faehrt (sonst ist die Verfolgung kein
Undercut, sondern zufaellig zeitversetzte Stopps). Erfolg heisst: nach
``nachlauf`` gruenen Runden liegt A vor dem Rivalen.

AUSBAUSTUFE  [umgesetzt]
Robustheit gegen die beiden Fenster-Parameter: haelt der Befund, wenn
``fenster`` (wie viele Runden der Rivale noch selbst stoppen darf) und
``nachlauf`` (wie lange nach dem letzten Stopp gewartet wird) variiert
werden, oder ist die 23.6%-Zahl ein Artefakt der Default-Werte?

Saison 2024, alle 24 Rennen: 161 echte Undercut-Duelle (Default
fenster=3, nachlauf=2), 38 erfolgreich - **23.6% Erfolgsquote, 95%-KI
[17.3%, 30.9%]**, Binomialtest gegen 50% mit p<0.0001. Der Undercut
gewinnt in diesen Daten deutlich seltener, als die verbreitete
Intuition ("Undercut ist die staerkere Waffe") nahelegt - eher das
Gegenteil: die Verteidigung (spaeter stoppen, auf der alten Mischung
verteidigen) gewinnt gut drei von vier dieser direkten Duelle.

Die Robustheitspruefung bestaetigt das ueber einen plausiblen
Parameterbereich, nicht nur beim Default:

    fenster=2, nachlauf=2:  n=137  Quote 21.2%  p=6.7e-12
    fenster=3, nachlauf=2:  n=161  Quote 23.6%  p=1.2e-11  (Default)
    fenster=5, nachlauf=2:  n=209  Quote 27.3%  p=3.6e-11
    fenster=3, nachlauf=1:  n=162  Quote 23.5%  p=7.6e-12
    fenster=3, nachlauf=3:  n=160  Quote 23.1%  p=5.5e-12

Ein engeres Zeitfenster (fenster=2) filtert die knappsten, eindeutigsten
Duelle heraus und zeigt die niedrigste Quote (21.2%); ein weiteres Fenster
(fenster=5) laesst auch zeitversetztere Stopp-Paare als "Duell" durch und
zieht die Quote am staerksten Richtung 50% (27.3%) - plausibel, weil ein
groesserer Rundenabstand zwischen den beiden Stopps die Verbindung zum
urspruenglichen Angriff verwaschen laesst. In der ganzen getesteten
Bandbreite bleibt die Kernaussage aber unveraendert: die Erfolgsquote
liegt durchgehend deutlich unter 50%, mit p-Werten, die in keinem Fall
auch nur in die Naehe von 0.05 kommen. Der Befund ist kein Artefakt einer
einzelnen Parameterwahl.

Je Rennen schwankt die Quote stark (0% bis 45.5%, siehe Balkengrafik) -
elf der 24 Rennen zeigen ueberhaupt einen erfolgreichen Undercut, aber
selbst das erfolgreichste Rennen (Grossbritannien, 5/11) bleibt unter
50%. Mit Stichproben von 1-16 Duellen je Rennen ist keins der
Einzelrennen fuer sich statistisch belastbar (siehe P40s Lektion ueber
Stichprobengroesse) - erst gepoolt ueber die Saison wird der Effekt
robust sichtbar.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")                      # kein Fenster, nur Dateien

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
    """VORGEHEN 2: alle Undercut-Duelle einer Saison sammeln."""
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
    """VORGEHEN 4: Erfolgsquote je Rennen, sortiert."""
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
    """VORGEHEN 3: Gesamtquote mit Konfidenzintervall gegen 50%."""
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
