"""simuliert eine rundenzeit aus streckenkruemmung und kalibrierten fahrzeugparametern und prueft die uebertragung auf andere strecken sowie einen kraftstoff-stint"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")                      # schreibt nur dateien ohne fenster

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import f1lab
from f1lab.design import FG, GRID, MUTED, SERIEN, matplotlib_stil

warnings.filterwarnings("ignore")

OUT = Path(__file__).parent / "out"
OUT.mkdir(exist_ok=True)

REFERENZ = (2024, "Bahrain", "Q")
VERGLEICHSSTRECKEN = ["Japan", "Saudi Arabia", "Britain", "Hungary", "Monza",
                      "Austria"]

plt.rcParams.update(matplotlib_stil())


def zeichne_referenz(ax, dist: np.ndarray, speed_real: np.ndarray,
                     v_sim: np.ndarray, t_real: float, t_sim: float) -> None:
    ax.plot(dist, speed_real * 3.6, color=MUTED, lw=1.8, label="Echte Telemetrie")
    ax.plot(dist, v_sim * 3.6, color=SERIEN[0], lw=1.8, ls="--",
           label="Simulation (kalibriert)")
    ax.set_xlabel("Distanz [m]")
    ax.set_ylabel("Speed [km/h]")
    ax.set_title(f"{REFERENZ[1]} {REFERENZ[0]} Q - real {t_real:.2f}s, "
                f"simuliert {t_sim:.2f}s ({100 * (t_sim - t_real) / t_real:+.1f}%)",
                loc="left", color=FG, fontsize=13, pad=10)
    ax.legend(loc="upper right", frameon=False, labelcolor=FG, fontsize=9)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(alpha=0.3, linewidth=0.8, color=GRID)
    ax.set_axisbelow(True)


def zeichne_stint(ax, runde: np.ndarray, zeiten: np.ndarray,
                  fuel_kg: np.ndarray) -> None:
    """zeigt die rundenzeit ueber einen stint waehrend der tank leerer wird."""
    ax.plot(runde, zeiten, color=SERIEN[0], lw=1.8, marker="o", ms=3)
    ax.set_xlabel("Runde")
    ax.set_ylabel("Simulierte Rundenzeit [s]")
    ax.set_title("ZWEITE AUSBAUSTUFE: Kraftstoffmasse im Stint "
                f"({fuel_kg[0]:.0f} kg -> {fuel_kg[-1]:.0f} kg)",
                loc="left", color=FG, fontsize=13, pad=10)
    ax.text(0.98, 0.92, f"{zeiten[0] - zeiten[-1]:+.2f}s ueber den Stint",
           transform=ax.transAxes, ha="right", va="top", color=MUTED,
           fontsize=10)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(alpha=0.3, linewidth=0.8, color=GRID)
    ax.set_axisbelow(True)


def zeichne_uebertragung(ax, ergebnisse: pd.DataFrame) -> None:
    """zeigt die abweichung der uebertragenen simulation je strecke."""
    ergebnisse = ergebnisse.sort_values("diff_pct")
    farben = [SERIEN[0] if abs(v) < 5 else SERIEN[1]
             for v in ergebnisse["diff_pct"]]
    ax.barh(ergebnisse["strecke"], ergebnisse["diff_pct"], color=farben,
           height=0.6)
    ax.axvline(0, color=MUTED, lw=1)
    ax.set_xlabel("Abweichung Simulation von echter Rundenzeit [%]")
    ax.set_title("AUSBAUSTUFE: dieselben Fahrzeugparameter auf anderen Strecken",
                loc="left", color=FG, fontsize=13, pad=10)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="x", alpha=0.3, linewidth=0.8, color=GRID)
    ax.set_axisbelow(True)


def main():
    f1lab.enable_cache()

    print(f"[1/3] {REFERENZ[1]} {REFERENZ[0]} {REFERENZ[2]} laden, "
         "Streckenprofil (VORGEHEN 1) ...")
    ses_ref = f1lab.load(*REFERENZ, telemetry=True)
    dist, kappa, speed_real = f1lab.lap_speed_profile(ses_ref)
    t_real = float(ses_ref.laps.pick_fastest()["LapTime"].total_seconds())
    print(f"      {len(dist)} Telemetriepunkte, echte Rundenzeit {t_real:.3f}s")

    print("\n[2/3] Kalibrierung gegen die echte Geschwindigkeitsspur "
         "(VORGEHEN 2-3) ...")
    params = f1lab.calibrate_lap_model(dist, kappa, speed_real)
    print(f"      mu_g={params['mu_g']:.2f} m/s^2 ({params['mu_g'] / 9.81:.2f}g)  "
         f"a_accel={params['a_accel']:.2f} m/s^2  "
         f"a_brake={params['a_brake']:.2f} m/s^2  "
         f"v_top={params['v_top'] * 3.6:.1f} km/h")
    print(f"      RMSE Geschwindigkeit: {params['rmse_ms']:.2f} m/s")

    v_sim, t_sim = f1lab.simulate_lap(dist, kappa, params["mu_g"],
                                      params["a_accel"], params["a_brake"],
                                      params["v_top"])
    print(f"\n[3/3] Validierung (VORGEHEN 4): simuliert {t_sim:.3f}s gegen "
         f"real {t_real:.3f}s ({100 * (t_sim - t_real) / t_real:+.2f}%)")

    print(f"\nAUSBAUSTUFE: dieselben Fahrzeugparameter auf "
         f"{len(VERGLEICHSSTRECKEN)} anderen Strecken (2024 Q) ...")
    zeilen = []
    for gp in VERGLEICHSSTRECKEN:
        try:
            ses = f1lab.load(2024, gp, "Q", telemetry=True)
        except Exception:
            continue
        d2, k2, _ = f1lab.lap_speed_profile(ses)
        t2_real = float(ses.laps.pick_fastest()["LapTime"].total_seconds())
        _, t2_sim = f1lab.simulate_lap(d2, k2, params["mu_g"], params["a_accel"],
                                       params["a_brake"], params["v_top"])
        diff_pct = 100 * (t2_sim - t2_real) / t2_real
        zeilen.append({"strecke": gp, "real_s": round(t2_real, 2),
                       "sim_s": round(t2_sim, 2), "diff_pct": round(diff_pct, 1)})
        print(f"      {gp:15s} real {t2_real:7.2f}s  simuliert {t2_sim:7.2f}s  "
             f"{diff_pct:+6.1f}%")
    ergebnisse = pd.DataFrame(zeilen)
    print(f"\n      Mittlerer absoluter Fehler: "
         f"{ergebnisse['diff_pct'].abs().mean():.1f}%  "
         f"(mittlerer Fehler mit Vorzeichen: {ergebnisse['diff_pct'].mean():+.1f}%)")

    print("\nZWEITE AUSBAUSTUFE: Kraftstoffmasse im Stint "
         "(Tank brennt Runde fuer Runde ab) ...")
    stint_laps = 57  # bahrain gp 2024 echte renndistanz
    fuel_start = f1lab.core.FUEL_KG_PER_LAP * stint_laps
    fuel_kg = np.maximum(fuel_start - f1lab.core.FUEL_KG_PER_LAP
                        * np.arange(stint_laps), 0.0)
    zeiten_stint = f1lab.simulate_stint(dist, kappa, params["mu_g"],
                                        params["a_accel"], params["a_brake"],
                                        params["v_top"], fuel_start_kg=fuel_start,
                                        n_laps=stint_laps)
    steigung = np.polyfit(fuel_kg, zeiten_stint, 1)[0]
    print(f"      {stint_laps} Runden, {fuel_start:.1f} kg Startkraftstoff "
         f"({f1lab.core.FUEL_KG_PER_LAP} kg/Runde)")
    print(f"      Runde 1: {zeiten_stint[0]:.3f}s -> Runde {stint_laps}: "
         f"{zeiten_stint[-1]:.3f}s ({zeiten_stint[0] - zeiten_stint[-1]:+.3f}s "
         "ueber den Stint)")
    print(f"      Aus der Simulation gefittete Steigung: {steigung:.4f} s/kg "
         f"(P04-Referenzwert aus echten Rennrunden: "
         f"{f1lab.core.FUEL_S_PER_KG} s/kg)")

    print("\nGrafik ...")
    fig, ax = plt.subplots(3, 1, figsize=(13, 16))
    zeichne_referenz(ax[0], dist, speed_real, v_sim, t_real, t_sim)
    zeichne_uebertragung(ax[1], ergebnisse)
    zeichne_stint(ax[2], np.arange(1, stint_laps + 1), zeiten_stint, fuel_kg)
    fig.suptitle("Rundenzeit-Simulation: Punktmassenmodell", x=0.09, ha="left",
                fontsize=16, color=FG, y=0.995)
    plt.tight_layout()
    path = OUT / "rundenzeit_simulation.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"\n      -> {path}")


if __name__ == "__main__":
    main()
