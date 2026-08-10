"""
P35 - Rennstrategie-Optimierer: exakt statt geraten
===================================================

Der beste Boxenstopp-Plan eines Rennens, exakt berechnet statt aus Kandidaten
ausgewaehlt - und daneben die Frage, was die Unsicherheit ueber Safety Cars kostet.

Kategorie:   Reifen & Strategie
Niveau:      Profi
Aufwand:     6-8 h
Schwerpunkt: Strategie, Optimierung

WARUM DAS LOHNT
P15 rechnet den Undercut zwischen zwei Autos aus, P13 liefert die Degradation.
Was fehlt, ist die Klammer: welcher Plan ueber das ganze Rennen ist der beste?
Der uebliche Weg waere, eine Handvoll Plaene (Einstopp, Zweistopp, ...)
durchzurechnen und den schnellsten zu nehmen. Das ist unnoetig - das Problem
hat eine Struktur, die den exakten Rundenplan in Millisekunden hergibt, ueber
allen legalen Strategien statt ueber einer Auswahl.

Der Kern: die Kosten eines Stints haengen nur von Mischung und Laenge ab, nicht
davon, wo im Rennen er liegt. Damit ist die Aufgabe ein kuerzester Pfad in einem
gerichteten azyklischen Graphen. Knoten j heisst "Runde j steht an", eine Kante
s -> e+1 ist ein Stint ueber die Runden s..e. Jeder Pfad ist genau eine
Strategie, seine Laenge genau die Rennzeit.

Der zweite Teil beantwortet, was ein Plan ueberhaupt wert ist, wenn man den
Rennverlauf nicht kennt. Ein Safety Car macht Boxenstopps billig; wann er kommt,
weiss vorher niemand. Deshalb ist das richtige Objekt kein Plan, sondern eine
Politik: eine Regel, was pro Runde zu tun ist, abhaengig von Reifen, Alter und
aktueller Flagge. Die faellt aus derselben Rueckwaertsrechnung.

VORGEHEN
  1. Rundenzeitmodell je Mischung: Basiszeit + Degradation ueber Reifenalter
  2. Alle legalen Stints als Kanten eines DAG aufzaehlen
  3. Kuerzester Pfad mit Mischungs-Bitmaske (fuer die Zweimischungs-Regel)
  4. Bester Plan je Stoppzahl und Sensitivitaet gegen den Pitloss
  5. Safety Car als Markow-Kette, optimale Politik per Rueckwaertsinduktion
  6. Zerlegung: was kostet Nichtwissen, was bringt Reagieren

GENUTZTE FASTF1-BAUSTEINE
  - keine fuer das Modell selbst (core.py, ohne Netz testbar).
  - fuer die AUSBAUSTUFE: f1lab.race_config_from_session() speist die
    Mischungsparameter aus echter Degradation (P13) und echtem Pitloss.

AUSBAUSTUFE  [umgesetzt]
Verkehr. Bisher faehrt genau ein Auto gegen die Uhr - keine Position, kein
Ueberholen, kein Herauskommen hinter einem Langsameren. Genau das bricht die
Trennbarkeit der Stintkosten und damit den kuerzesten Pfad. Wer das angeht,
braucht ein anderes Verfahren (Simulation oder ein ganzzahliges Programm) und
bekommt dafuer das erste Modell, in dem Reagieren wirklich viel wert ist -
bewusst nicht umgesetzt, das waere ein eigenes Projekt.

Die zwei im urspruenglichen Docstring offen gelassenen Ausbauten sind jetzt
da: die Optimierungslogik (TyreModel bis hindsight_value) steckt seit der
App-Integration in f1lab.core neben undercut_gain (kein FastF1 noetig, siehe
Test test_core_functions_are_numpy_only), und
f1lab.session.race_config_from_session() baut eine RaceConfig aus echter
Degradation und echtem Pitloss statt von Hand gesetzter Zahlen. Gegenprobe
gegen Bahrain 2024 R: die geschaetzte Degradation (Hard 0.095 s/Runde, Soft
0.124 s/Runde) trifft P13s dort unabhaengig ermittelte Werte praktisch exakt
- derselbe zugrundeliegende Fit, nur diesmal als Eingabe fuer eine Optimierung
statt als Tabellenzeile. Das echte Rennen zeigt dabei etwas, das das
synthetische Beispiel nicht zeigen kann: mit nur zwei belastbaren Mischungen
(Medium fehlt in den zuverlaessigen Fits) schrumpft der Loesungsraum
gegenueber der Referenz mit drei Mischungen spuerbar - weniger Kandidaten,
nicht weil das Modell schwaecher waere, sondern weil die Saison diese
Session so gefahren wurde.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import f1lab
from f1lab import (
    SafetyCarProcess,
    TyreModel,
    expected_cost_of_plan,
    frontier_by_stops,
    hindsight_value,
    optimal_strategy,
    pit_loss_crossovers,
    solve_policy,
)
from f1lab.core import RaceConfig

warnings.filterwarnings("ignore")

OUT = Path(__file__).parent / "out"
OUT.mkdir(exist_ok=True)

REAL_EVENT = (2024, "Bahrain", "R")


def reference_race() -> RaceConfig:
    """Ein plausibles Einstopp-bis-Zweistopp-Rennen, 66 Runden.

    Die Zahlen sind gesetzt, nicht geschaetzt - fuer echte Werte siehe
    ``real_race()`` unten, das dieselbe RaceConfig aus f1lab.
    race_config_from_session() gegen eine echte Session baut.
    """
    return RaceConfig(
        n_laps=66,
        pit_loss=22.0,
        tyres=(
            TyreModel("SOFT", 79.4, 0.055, 0.0012, max_age=25),
            TyreModel("MEDIUM", 80.0, 0.032, 0.0004, max_age=40),
            TyreModel("HARD", 80.5, 0.022, 0.0001, max_age=55),
        ),
        min_stint=4,
        fuel_effect=0.055,
    )


def real_race():
    """AUSBAUSTUFE: dieselbe RaceConfig, aber aus echter Degradation (P13)
    und echtem Pitloss statt von Hand gesetzter Zahlen."""
    f1lab.enable_cache()
    ses = f1lab.load(*REAL_EVENT, telemetry=False)
    return ses, f1lab.race_config_from_session(ses)


def bericht(titel: str, cfg: RaceConfig) -> None:
    print(f"=== {titel} ===")
    print(f"Rennen: {cfg.n_laps} Runden, Pitloss {cfg.pit_loss:.1f} s, "
         f"{len(cfg.tyres)} Mischungen")
    for t in cfg.tyres:
        print(f"  {t.compound:8s} Basis {t.base_time:7.3f} s  "
             f"Degradation {t.deg_linear:+.4f} s/Runde  max_age {t.max_age}")

    beste = optimal_strategy(cfg)
    print("\nOPTIMUM")
    print("  " + beste.describe().replace("\n", "\n  "))

    print("\nBester Plan je Stoppzahl (Abstand zum Optimum):")
    for n, s in frontier_by_stops(cfg).items():
        if s is None:
            print(f"  {n}-Stopp: nicht moeglich")
        else:
            print(f"  {n}-Stopp: {' / '.join(s.compounds):28s} "
                 f"+{s.green_time - beste.green_time:6.2f} s  Box {s.pit_laps}")

    lo, hi = max(5.0, cfg.pit_loss - 15), cfg.pit_loss + 15
    print(f"\nStoppzahl kippt bei Pitloss: {pit_loss_crossovers(cfg, lo, hi)} s")

    prozess = SafetyCarProcess()
    wert, _politik = solve_policy(cfg, prozess)
    naiv = expected_cost_of_plan(cfg, beste, prozess)
    blind, se = hindsight_value(cfg, prozess, n=300)
    print("\nMIT UNSICHERHEIT UEBER DAS SAFETY CAR")
    print(f"  Optimum bei bekanntem Verlauf   {blind:9.2f} s  +- {se:.2f}")
    print(f"  optimale Politik                {wert:9.2f} s")
    print(f"  fester Plan, stur durchgezogen  {naiv:9.2f} s")
    print(f"  Preis des Nichtwissens  {wert - blind:5.2f} s")
    print(f"  Wert des Reagierens     {naiv - wert:5.2f} s")


def main():
    cfg_synth = reference_race()
    print(f"Kandidaten-Stints im synthetischen Beispiel: "
         f"{len(f1lab.stint_arcs(cfg_synth))}\n")
    bericht("Synthetisches Beispiel (Zahlen von Hand gesetzt)", cfg_synth)

    print(f"\n\n[AUSBAUSTUFE] {REAL_EVENT[1]} {REAL_EVENT[0]} laden, echte "
         f"RaceConfig bauen ...")
    ses, cfg_real = real_race()
    print(f"Kandidaten-Stints im echten Rennen: {len(f1lab.stint_arcs(cfg_real))}\n")
    bericht(f"{ses.event['EventName']} {REAL_EVENT[0]} (echte Degradation + Pitloss)",
           cfg_real)


if __name__ == "__main__":
    main()
