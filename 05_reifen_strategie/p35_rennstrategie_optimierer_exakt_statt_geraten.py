"""berechnet den optimalen boxenstopp-plan eines rennens als kürzesten pfad und eine safety-car-politik per rückwärtsinduktion"""
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
    """ein plausibles einstopp-bis-zweistopp-rennen, 66 runden.

    die zahlen sind von hand gesetzt, nicht geschätzt. für echte werte
    siehe ``real_race()`` unten.
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
    """dieselbe RaceConfig, aber aus echter degradation und echtem pitloss statt von hand gesetzter zahlen."""
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
