"""
P37 - Rundenzeit-Simulation: ein Punktmassenmodell, kalibriert gegen echte Telemetrie
=======================================================================================

Kein Nachschlagen echter Rundenzeiten - ein physikalisches Modell (Kruemmung, Grip-/Beschleunigungsgrenzen) rechnet eine Runde aus.

Kategorie:   Strecke & Position
Niveau:      Profi
Aufwand:     6-8 h
Schwerpunkt: Datenanalyse, Physik/Numerik

WARUM DAS LOHNT
FastF1 selbst simuliert nichts - es liefert nur, was tatsaechlich gefahren
wurde. Dieses Projekt geht einen Schritt weiter: aus der Streckengeometrie
(Kruemmung entlang der echten Ideallinie) und wenigen physikalischen
Grenzwerten (Grip, Beschleunigung, Bremsverzoegerung, Hoechstgeschwindigkeit)
wird eine Rundenzeit numerisch integriert - dieselbe Grundidee, die
professionelle Rundenzeit-Simulatoren (OptimumLap und aehnliche Tools)
verwenden, hier komplett selbst gebaut und gegen echte Telemetrie kalibriert
und geprueft.

VORGEHEN
  1. Streckenkruemmung aus der realen X/Y-Ideallinie der schnellsten Runde
     numerisch bestimmen (geglaettet, dann zweifach differenziert)
  2. Vorwaerts-Rueckwaerts-Algorithmus: vorwaerts beschleunigungsbegrenzt,
     rueckwaerts bremsbegrenzt, dazwischen kurvengrip-begrenzt - das
     Standardverfahren fuer quasi-stationaere Punktmassen-Rundenzeitsimulation
  3. Vier Fahrzeugparameter (Kurven-Grenzbeschleunigung, Laengsbeschleunigung,
     Bremsverzoegerung, Hoechstgeschwindigkeit) gegen die echte
     Geschwindigkeitsspur einer Referenzrunde kalibrieren (kleinste Quadrate)
  4. Simulierte gegen echte Rundenzeit und Geschwindigkeitsspur validieren
  5. Die kalibrierten Fahrzeugparameter unveraendert auf andere Strecken
     anwenden und pruefen, ob das Modell dort noch stimmt

GENUTZTE FASTF1-BAUSTEINE
  - Lap.get_telemetry/add_distance (X/Y-Ideallinie, echte Geschwindigkeit)
  - scipy.signal.savgol_filter, scipy.optimize.least_squares

AUSBAUSTUFE  [umgesetzt]
Die kalibrierten Fahrzeugparameter (nicht die Streckengeometrie - die kommt
je Strecke aus deren eigener echter Telemetrie) unveraendert auf sechs
andere Strecken anwenden und die simulierte gegen die echte Pole-Rundenzeit
pruefen: traegt das an einer Strecke identifizierte "Auto" ueber die ganze
Saison?

Kalibriert an Bahrain 2024 Q (Referenz): Simulation trifft die echte
Rundenzeit auf 0.5% genau (89.63 s simuliert gegen 89.165 s real), mit
physikalisch plausiblen Parametern - Kurvengrenzbeschleunigung 25.7 m/s²
(~2.6g, glaubwuerdig fuer abtriebsstarke Kurven), Laengsbeschleunigung
7.4 m/s² (~0.75g, traktionsbegrenzt aus langsamen Kurven), Bremsverzoegerung
29.2 m/s² (~3.0g, ein plausibler Mittelwert ueber ganze Bremszonen -
Spitzenwerte liegen hoeher), Hoechstgeschwindigkeit 299 km/h.

Auf sechs anderen Strecken (2024 Q) zeigt sich: die Uebertragung gelingt nur
teilweise. Britain trifft mit 1.6% fast so gut wie die Kalibrierung selbst;
Austria (-3.8%) und Hungary (-9.3%) liegen noch im plausiblen Bereich.
Saudi Arabia (+11.9%), Monza (+7.4%) und vor allem Japan (+16.8%) weichen
deutlich staerker ab - im Mittel 8.5% absoluter Fehler ueber alle sechs
Strecken, mit einer leichten systematischen Tendenz, zu langsam zu simulieren
(mittlerer Fehler +4.1%, nicht 0%). Der wahrscheinlichste Grund ist keine
Modellschwaeche im Kern, sondern eine zu grobe Vereinfachung: das Modell
kennt nur die 2D-Kruemmung der Ideallinie, nicht Hoehenprofil (Suzuka mit
seinen starken Steigungen ist der schlechteste Fall), unterschiedliche
Fluegeleinstellungen je Strecke (Monza mit seinem Niedrig-Abtrieb-Setup
braeuchte einen hoeheren v_top als Bahrain) oder Reifentemperaturfenster.
Ein EIN Satz Fahrzeugparameter fuer die ganze Saison ist damit eine zu
starke Vereinfachung - realistische Simulatoren passen mindestens Abtrieb/
v_top je Strecke an, was dieses Modell bewusst nicht tut, um zu zeigen, wo
genau die Grenze der Vereinfachung liegt.
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
    """VORGEHEN 4."""
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


def zeichne_uebertragung(ax, ergebnisse: pd.DataFrame) -> None:
    """AUSBAUSTUFE: Abweichung der uebertragenen Simulation je Strecke."""
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

    print("\nGrafik ...")
    fig, ax = plt.subplots(2, 1, figsize=(13, 11))
    zeichne_referenz(ax[0], dist, speed_real, v_sim, t_real, t_sim)
    zeichne_uebertragung(ax[1], ergebnisse)
    fig.suptitle("Rundenzeit-Simulation: Punktmassenmodell", x=0.09, ha="left",
                fontsize=16, color=FG, y=0.995)
    plt.tight_layout()
    path = OUT / "rundenzeit_simulation.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"\n      -> {path}")


if __name__ == "__main__":
    main()
