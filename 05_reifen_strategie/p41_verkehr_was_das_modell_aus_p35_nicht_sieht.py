"""simuliert verkehr gegen einen rivalen und prüft ob das ergebnis P35s optimale strategie ändert"""
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
