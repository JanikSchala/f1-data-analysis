"""
P41 - Verkehr: was das Rundenzeitmodell aus P35 nicht sieht
=============================================================

P35 rechnet die exakt beste Strategie fuer ein Auto gegen die Uhr - keine Position, kein Ueberholen, kein Herauskommen hinter einem Langsameren. Was aendert sich, wenn man das dazunimmt?

Kategorie:   Reifen & Strategie
Niveau:      Profi
Aufwand:     5-6 h
Schwerpunkt: Strategie, Simulation

WARUM DAS LOHNT
P35s eigener Docstring nennt "Verkehr" ausdruecklich als das, was der
kuerzeste Pfad nicht kann: sobald Position und Ueberholen dazukommen, ist
die Trennbarkeit der Stintkosten weg, und ein anderes Verfahren ist noetig
(Simulation statt DAG) - bewusst als eigenes Projekt offen gelassen. P38/P39
liefern inzwischen echte, quantifizierte Bausteine genau dafuer
(Ueberholschwierigkeit je Strecke, DRS-Anteil an Ueberholungen). Dieses
Projekt ist der Ort, an dem sie tatsaechlich einfliessen - nicht als
Erweiterung des DAG (der bleibt unangetastet), sondern als eigene
Simulationsschicht daneben.

VORGEHEN
  1. Zwei DAG-optimale Kandidatenplaene aus derselben echten RaceConfig
     nehmen (P35, Bahrain 2024 R): den echten Zweistopp-Optimum und den
     besten Dreistopp als plausible Alternative (f1lab.frontier_by_stops)
  2. Rundenzeiten je Plan ableiten (f1lab.lap_times_for_strategy)
  3. Gegen einen Rivalen rundenweise simulieren: f1lab.gap_evolution() mit
     einer Ueberholwahrscheinlichkeit statt sofortiger Positionsuebernahme
  4. Erwarteten Zeitverlust durch Verkehr ueber viele Zufallslaeufe
     (f1lab.traffic_cost) fuer beide Plaene und mehrere p_overtake-Szenarien
     vergleichen: aendert sich die Empfehlung aus P35, wenn Verkehr
     mitgerechnet wird?

GENUTZTE FASTF1-BAUSTEINE
  - f1lab.race_config_from_session() (P35) fuer die echte Kalibrierung
  - sonst keine - die Simulation selbst ist reines core.py (numpy/random),
    ohne Netz testbar (siehe tests/test_core.py)

Zwei Annahmen sind bewusst Szenario-Parameter, keine gemessenen Werte -
genau wie P35 schon seine Safety-Car-Wahrscheinlichkeiten als einstellbare
Eingabe behandelt, nicht als geschaetzte Konstante:

  - RIVALE: dieselbe Strategieform wie der Zweistopp-Plan (identische
    Boxenrunden), aber konstant DELTA=0.15 s/Runde langsamer - ein
    plausibler, aber erfundener Abstand zwischen zwei aehnlich schnellen
    Autos, nicht aus dieser Session gemessen. Damit heben sich
    Boxenstopp-Zeitpunkte aus dem Abstandsverlauf heraus (beide stoppen
    gleichzeitig), und die Dynamik kommt allein aus dem Tempounterschied -
    absichtlich sauber, um den Verkehrsmechanismus isoliert zu zeigen.
  - START_GAP=3.0 s: Abstand vor Runde 1, ebenfalls gesetzt, nicht
    gemessen.
  - p_overtake: drei Szenariowerte (0.05/0.15/0.35), an der Bandbreite aus
    P38 orientiert (schwer wie Monaco/Jeddah bis leicht wie Spanien), aber
    NICHT rechnerisch aus P38/P39 abgeleitet - dafuer fehlt die noetige
    Zwischengroesse (Wahrscheinlichkeit, pro Runde in Reichweite zu sein),
    die weder P38 noch P39 liefert. Eine erfundene Formel dafuer waere
    keine Kalibrierung, nur eine Annahme mit Kalibrierungs-Anstrich (siehe
    die Nachlese zu P35 in CLAUDE.md).

AUSBAUSTUFE  [umgesetzt]
Sensitivitaet: denselben Vergleich fuer eine Bandbreite an Startabstaenden
(1.5-4.0 s) wiederholen - kippt die Empfehlung irgendwo in einem
plausiblen Bereich, oder ist sie robust?

Bahrain 2024 R (57 Runden, Pitloss 25.0 s, HARD/SOFT als einzige
belastbaren Mischungen - dieselbe Kalibrierung wie P35): der
Zweistopp-Optimum (SOFT[1-14]->HARD[15-35]->HARD[36-57], 5495.63 s) gegen
den besten Dreistopp (5504.39 s, naiver Abstand +8.77 s). Gegen denselben
Rivalen (START_GAP=3.0 s, DELTA=0.15 s/Runde) simuliert (5000 Laeufe je
Szenario): der Abstand WEITET sich unter Verkehr, statt zu kippen - auf
+17.76 s (p=0.05, schwere Strecke), +16.81 s (p=0.15) und +14.49 s (p=0.35,
leichte Strecke). In allen drei Szenarien bleibt der Zweistopp die bessere
Wahl - aber die Sicherheitsmarge ist real fast doppelt so gross wie der
naive DAG-Wert vermuten laesst.

Der Grund ist selbst ein Befund, kein Artefakt: der Dreistopp faehrt
haeufiger auf frischen Reifen und naehert sich dem Rivalen dadurch frueher
im Rennen an als der Zweistopp - er verbringt schlicht mehr Rennstrecke in
Reichweite des Rivalen, bevor die Flagge faellt, unabhaengig davon, wie gut
seine Ueberholchancen pro Runde sind. Die Sensitivitaets-AUSBAUSTUFE
(Startabstand 1.5 bis 4.0 s, beide p_overtake-Extreme) bestaetigt: die
Empfehlung kippt in diesem gesamten plausiblen Bereich nie - der
Zweistopp bleibt durchgehend besser, mit einer mit dem Startabstand
leicht wachsenden Marge (je naeher am Rivalen gestartet wird, desto mehr
Zeit bleibt fuer Verkehr, um die Luecke zu vergroessern).

Ehrliche Grenze: bei DIESER Paarung (Zweistopp gegen Dreistopp aus
derselben Session) kippt die Rangfolge nirgends - das ist ein echtes
Ergebnis, keine gesuchte Dramatik. Ob es Paarungen mit knapperem
DAG-Abstand gibt, bei denen Verkehr tatsaechlich die Empfehlung dreht,
zeigt dieses Beispiel nicht - dafuer waere eine Session mit drei
belastbaren Mischungen und entsprechend engeren Kandidatenplaenen noetig,
hier begrenzt auf zwei (siehe P35s eigene Beobachtung dazu).
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import f1lab

warnings.filterwarnings("ignore")

REAL_EVENT = (2024, "Bahrain", "R")
DELTA_S_PRO_RUNDE = 0.15
START_GAP_S = 3.0
P_OVERTAKE_SZENARIEN = [(0.05, "schwer, wie Monaco/Jeddah in P38"),
                        (0.15, "mittel"),
                        (0.35, "leicht, wie Spanien in P38")]
N_SIM = 5000


def main():
    f1lab.enable_cache()

    print(f"[1/3] {REAL_EVENT[1]} {REAL_EVENT[0]} {REAL_EVENT[2]} laden, "
         "echte RaceConfig bauen (VORGEHEN 1) ...")
    ses = f1lab.load(*REAL_EVENT, telemetry=False)
    cfg = f1lab.race_config_from_session(ses)
    print(f"      {cfg.n_laps} Runden, Pitloss {cfg.pit_loss:.1f}s, "
         f"{len(cfg.tyres)} belastbare Mischungen")

    zweistopp = f1lab.optimal_strategy(cfg)
    frontier = f1lab.frontier_by_stops(cfg, up_to=3)
    dreistopp = frontier[3]
    print(f"\n      Zweistopp (Optimum): {zweistopp.green_time:.2f}s  "
         f"{zweistopp.describe()}")
    print(f"\n      Dreistopp (Alternative): {dreistopp.green_time:.2f}s  "
         f"{dreistopp.describe()}")
    print(f"\n      Naiver Abstand (ohne Verkehr): "
         f"{dreistopp.green_time - zweistopp.green_time:+.2f}s")

    print("\n[2/3] Rundenzeiten und Rivale bilden (VORGEHEN 2-3) ...")
    zweistopp_t = f1lab.lap_times_for_strategy(cfg, zweistopp)
    dreistopp_t = f1lab.lap_times_for_strategy(cfg, dreistopp)
    rivale_t = zweistopp_t + DELTA_S_PRO_RUNDE
    print(f"      Rivale: dieselbe Boxenrunden wie der Zweistopp, "
         f"{DELTA_S_PRO_RUNDE}s/Runde langsamer (Annahme, siehe Docstring)")

    print(f"\n[3/3] Verkehrs-Simulation, {N_SIM} Laeufe je Szenario "
         "(VORGEHEN 4) ...")
    for p_over, label in P_OVERTAKE_SZENARIEN:
        c2, se2 = f1lab.traffic_cost(zweistopp_t, rivale_t, START_GAP_S,
                                     p_over, n_sim=N_SIM, seed=1)
        c3, se3 = f1lab.traffic_cost(dreistopp_t, rivale_t, START_GAP_S,
                                     p_over, n_sim=N_SIM, seed=1)
        t2 = zweistopp.green_time + c2
        t3 = dreistopp.green_time + c3
        besser = "Zweistopp" if t2 < t3 else "DREISTOPP (Empfehlung gekippt!)"
        print(f"      p_overtake={p_over:.2f} ({label})")
        print(f"        Zweistopp: {zweistopp.green_time:8.2f}s frei + "
             f"{c2:5.2f}+-{se2:.2f}s Verkehr = {t2:8.2f}s")
        print(f"        Dreistopp: {dreistopp.green_time:8.2f}s frei + "
             f"{c3:5.2f}+-{se3:.2f}s Verkehr = {t3:8.2f}s")
        print(f"        Abstand mit Verkehr: {t3 - t2:+.2f}s  -> {besser}")

    print("\nAUSBAUSTUFE: Sensitivitaet gegen den Startabstand ...")
    for gap0 in (1.5, 2.0, 3.0, 4.0):
        zeile = f"      START_GAP={gap0:.1f}s: "
        for p_over, _ in (P_OVERTAKE_SZENARIEN[0], P_OVERTAKE_SZENARIEN[-1]):
            c2, _ = f1lab.traffic_cost(zweistopp_t, rivale_t, gap0, p_over,
                                       n_sim=N_SIM, seed=1)
            c3, _ = f1lab.traffic_cost(dreistopp_t, rivale_t, gap0, p_over,
                                       n_sim=N_SIM, seed=1)
            diff = (dreistopp.green_time + c3) - (zweistopp.green_time + c2)
            zeile += f"p={p_over:.2f}: {diff:+6.2f}s  "
        print(zeile)


if __name__ == "__main__":
    main()
