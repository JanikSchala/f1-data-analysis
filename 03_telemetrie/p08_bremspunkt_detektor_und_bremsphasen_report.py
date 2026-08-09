"""
P08 - Bremspunkt-Detektor und Bremsphasen-Report
================================================

Aus dem Brake-Kanal automatisch alle Bremszonen extrahieren: Einsatzpunkt, Dauer, Delta-v, Verzoegerung.

Kategorie:   Telemetrie
Niveau:      Profi
Aufwand:     4-5 h
Schwerpunkt: Datenanalyse, Strategie

WARUM DAS LOHNT
Bremspunkte sind die wertvollste Einzelinformation im Fahrervergleich. Signal-Segmentierung ist echte Engineering-Arbeit, nicht nur Plotting.

VORGEHEN
  1. Brake-Kanal als bool auslesen, Flanken per diff finden
  2. Zusammenhaengende Bremszonen zu Intervallen gruppieren
  3. Je Zone Eintrittsspeed, Minimalspeed, Laenge, mittlere Verzoegerung berechnen
  4. Zonen zweier Fahrer nebeneinanderstellen

GENUTZTE FASTF1-BAUSTEINE
  - Telemetry Brake/Speed/Distance
  - Telemetry.add_distance
  - f1lab.braking_zones (Flankenerkennung, siehe unten)
  - f1lab.match_by_distance (Zonen-Zuordnung, AUSBAUSTUFE)

AUSBAUSTUFE  [umgesetzt]
Zonen beider Fahrer ueber die Distanz matchen und ausgeben, wer wo wie viele
Meter spaeter bremst.

Die Flankenerkennung selbst (VORGEHEN 1-3) dupliziert nicht mehr die eigene
Funktion aus der urspruenglichen Fassung, sondern ruft
f1lab.driver_braking_zones() - exakt dieselbe Funktion, die auch die
Telemetrie-Seite des Dashboards und P07 verwenden. Zwei eigene
Implementationen desselben Flanken-Algorithmus waeren genau die Art
Duplikat, die in diesem Projekt schon einmal zur Bug-Quelle wurde (siehe
CLAUDE.md). Die Zonen-Zuordnung (AUSBAUSTUFE) nutzt f1lab.match_by_distance
- P07 nutzt dieselbe Funktion fuer sein eigenes Zonen-Overlay.

Oesterreich 2024 Q, VER gegen LEC: Verstappen bremst in 4 von 5 gematchten
Zonen spaeter (bis zu 25.9 m) - und ist am Ende 0.730s schneller. Leclerc hat
dabei in allen fuenf Zonen die staerkere Verzoegerung (z.B. 3.59 g gegen
2.71 g in Zone 1), bremst aber jeweils frueher: der spaetere Bremspunkt zaehlt
hier mehr als die haertere Bremsung selbst - wer spaeter bremst, kommt
schneller in die Kurve, auch mit schwaecherer Verzoegerung.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")                      # kein Fenster, nur Dateien

import fastf1.plotting as f1plt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import f1lab
from f1lab.design import FG, GRID, matplotlib_stil

warnings.filterwarnings("ignore")

OUT = Path(__file__).parent / "out"
OUT.mkdir(exist_ok=True)

SEASON, EVENT, IDENT = 2024, "Austria", "Q"
D1, D2 = "VER", "LEC"
TOLERANZ_M = 150.0

plt.rcParams.update(matplotlib_stil())
f1plt.setup_mpl(mpl_timedelta_support=False, color_scheme="fastf1")


def bremsphasen_report(ses, driver: str) -> pd.DataFrame:
    """VORGEHEN 1-3: Bremszonen einer Runde, ueber f1lab.driver_braking_zones
    (dieselbe Funktion wie P07 und der Bremszonen-Reiter im Dashboard) - hier
    nur um Fahrerkuerzel und Zonennummer fuer den Report ergaenzt."""
    zonen = f1lab.driver_braking_zones(ses, driver)
    zonen.insert(0, "driver", driver)
    zonen.insert(1, "zone", range(1, len(zonen) + 1))
    return zonen


def vergleiche(z1: pd.DataFrame, z2: pd.DataFrame,
              toleranz_m: float = TOLERANZ_M) -> pd.DataFrame:
    """VORGEHEN 4 + AUSBAUSTUFE: Zonen nebeneinanderstellen und matchen."""
    paare = f1lab.match_by_distance(z1["start_m"], z2["start_m"], toleranz_m)
    zeilen = []
    for i, j in paare:
        a, b = z1.iloc[i], z2.iloc[j]
        zeilen.append({
            "start_m": round((a["start_m"] + b["start_m"]) / 2, 1),
            f"start_{a['driver']}": a["start_m"],
            f"start_{b['driver']}": b["start_m"],
            "delta_m": round(b["start_m"] - a["start_m"], 1),
            "spaeter": b["driver"] if b["start_m"] > a["start_m"] else a["driver"],
            f"decel_{a['driver']}_g": a["decel_g"],
            f"decel_{b['driver']}_g": b["decel_g"],
        })
    return pd.DataFrame(zeilen).sort_values("start_m", ignore_index=True)


def zeichne_verzoegerung(ax, vgl: pd.DataFrame, ses, d1: str, d2: str) -> None:
    farbe1 = f1plt.get_driver_color(d1, session=ses)
    farbe2 = f1plt.get_driver_color(d2, session=ses)

    breite = 0.35
    x = np.arange(len(vgl))
    ax.bar(x - breite / 2, vgl[f"decel_{d1}_g"], breite, color=farbe1, label=d1)
    ax.bar(x + breite / 2, vgl[f"decel_{d2}_g"], breite, color=farbe2, label=d2)
    ax.set_xticks(x, [f"#{i + 1}" for i in range(len(vgl))])
    ax.set_xlabel("Bremszone")
    ax.set_ylabel("Mittlere Verzoegerung [g]")
    ax.set_title("Verzoegerung je gematchter Zone", loc="left", color=FG,
                fontsize=13, pad=12)
    ax.legend(frameon=False, labelcolor=FG)
    ax.grid(axis="y", alpha=0.35, linewidth=0.8, color=GRID)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def zeichne_bremspunkte(ax, vgl: pd.DataFrame, ses, d1: str, d2: str) -> None:
    farbe = {d1: f1plt.get_driver_color(d1, session=ses),
            d2: f1plt.get_driver_color(d2, session=ses)}

    x = np.arange(len(vgl))
    farben = [farbe[s] for s in vgl["spaeter"]]
    ax.bar(x, vgl["delta_m"].abs(), color=farben)
    for i, (dm, spaeter) in enumerate(zip(vgl["delta_m"], vgl["spaeter"], strict=True)):
        ax.text(i, abs(dm) + 0.5, spaeter, ha="center", va="bottom",
               color=FG, fontsize=9)
    ax.set_xticks(x, [f"#{i + 1}" for i in range(len(vgl))])
    ax.set_xlabel("Bremszone")
    ax.set_ylabel("Spaeter gebremst [m]")
    ax.set_title("Wer bremst wo spaeter", loc="left", color=FG, fontsize=13,
                pad=12)
    ax.grid(axis="y", alpha=0.35, linewidth=0.8, color=GRID)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def main():
    f1lab.enable_cache()

    print(f"[1/3] {EVENT} {SEASON} {IDENT} laden (mit Telemetrie) ...")
    ses = f1lab.load(SEASON, EVENT, IDENT, telemetry=True)

    print("[2/3] Bremsphasen-Report ...")
    z1 = bremsphasen_report(ses, D1)
    z2 = bremsphasen_report(ses, D2)
    for z in (z1, z2):
        print(f"\n=== {z['driver'].iloc[0]} - {len(z)} Bremszonen ===")
        print(z.drop(columns=["driver"]).to_string(index=False))

    vgl = vergleiche(z1, z2)
    print(f"\n{len(vgl)} Zonen gematcht (Toleranz {TOLERANZ_M:.0f} m):")
    print(vgl[["start_m", "delta_m", "spaeter"]].to_string(index=False))

    l1 = ses.laps.pick_drivers(D1).pick_fastest()
    l2 = ses.laps.pick_drivers(D2).pick_fastest()
    schneller = D1 if l1["LapTime"] < l2["LapTime"] else D2
    diff = abs((l1["LapTime"] - l2["LapTime"]).total_seconds())
    spaeter_zonen = (vgl["spaeter"] == D2).sum()
    print(f"\n{D2} bremst in {spaeter_zonen} von {len(vgl)} Zonen spaeter, "
         f"{schneller} ist trotzdem {diff:.3f}s schneller.")

    print("\n[3/3] Grafik ...")
    fig, ax = plt.subplots(1, 2, figsize=(13, 5.5))
    zeichne_verzoegerung(ax[0], vgl, ses, D1, D2)
    zeichne_bremspunkte(ax[1], vgl, ses, D1, D2)
    fig.suptitle(f"{EVENT} {SEASON} {IDENT} - Bremsphasen-Report {D1} vs "
                f"{D2}", x=0.09, ha="left", fontsize=16, color=FG, y=1.02)
    plt.tight_layout()
    path = OUT / "bremsphasen_report.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"\n      -> {path}")


if __name__ == "__main__":
    main()
