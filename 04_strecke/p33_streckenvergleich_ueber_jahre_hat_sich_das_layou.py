"""
P33 - Streckenvergleich ueber Jahre: Hat sich das Layout geaendert?
===================================================================

Dieselbe Strecke in verschiedenen Saisons: Streckenlaenge aus Telemetrie, Speed-Profile und Rundenzeitentwicklung.

Kategorie:   Strecke & Position
Niveau:      Fortgeschritten
Aufwand:     3-4 h
Schwerpunkt: Datenanalyse, Engineering

WARUM DAS LOHNT
Zeigt, dass du Daten ueber Zeit hinweg vergleichbar machen kannst - Normalisierung ist eine der schwersten Aufgaben in echten Datenprojekten.

VORGEHEN
  1. Pole-Runde derselben Strecke fuer mehrere Saisons laden
  2. Speed ueber RelativeDistance interpolieren, damit die Kurven vergleichbar sind
  3. Speed-Profile uebereinanderlegen
  4. Rundenzeit und Kurvenanzahl je Jahr gegenueberstellen

GENUTZTE FASTF1-BAUSTEINE
  - Telemetry.add_distance
  - Telemetry.add_relative_distance
  - get_circuit_info
  - mehrere Sessions

AUSBAUSTUFE  [umgesetzt]
Die Regelaera als Kategorie ergaenzen (2018, Hochabtrieb-Ende der vorigen
Aera, gegen 2024, Ground-Effect-Aera seit 2022) und quantifizieren, wo die
neuen Autos schneller sind.

"5 Saisons" aus der urspruenglichen Formulierung war mit dem Cache nicht
umsetzbar: Telemetrie liegt nur fuer 2018 und 2024 vor (siehe CLAUDE.md).
Das trifft sich mit der AUSBAUSTUFE aber gut - genau die zwei Jahre, die den
groessten Regel-Umbruch der Telemetrie-Aera einrahmen.

Fuer den Streckenvergleich (VORGEHEN) bleibt Spanien die Wahl der
urspruenglichen Fassung - und liefert nebenbei eine Lektion in Datenkritik:
die Delta-Kurve zeigt bei 89% der Runde einen Ausschlag von +170 km/h, kein
Messfehler, sondern die vor 2023 entfernte Schikane vor der Start-Ziel-
Geraden. 2018 bremste dort jeder, 2024 nicht mehr. Fuer die AUSBAUSTUFE
("wo sind Ground-Effect-Autos schneller") ist Spanien deshalb ungeeignet -
der Effekt waere Layout, nicht Regelwerk. Sechs Strecken ohne bekannte
Layout-Aenderung in diesem Zeitraum (Oesterreich, Aserbaidschan, Bahrain,
Kanada, China, Monaco) uebernehmen die Quantifizierung stattdessen.

Ergebnis, gegen die naive Erwartung "neue Regeln = schnellere Autos": auf
allen sechs unveraenderten Strecken faellt die mittlere Rundengeschwindigkeit
2024 gegenueber 2018 (-1.7 bis -7.2 km/h), die Rundenzeit steigt an vier von
sechs. Plausible Erklaerung, nicht in den Daten selbst belegbar: das
Mindestgewicht stieg mit der Ground-Effect-Aera um rund 65 kg, waehrend 2018
das Ende eines langen Entwicklungszyklus der vorigen Regeln war - beides
wuerde in dieselbe Richtung wirken. Topspeed ist uneinheitlich (drei
Strecken schneller, drei langsamer) und zeigt kein Muster.
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
from f1lab.design import FG, GRID, MUTED, POSITIV, SERIEN, matplotlib_stil

warnings.filterwarnings("ignore")

OUT = Path(__file__).parent / "out"
OUT.mkdir(exist_ok=True)

GP = "Spain"
YEARS = (2018, 2024)
REGEL_STRECKEN = ["Austria", "Azerbaijan", "Bahrain", "Canada", "China", "Monaco"]
GRID_N = 1000

plt.rcParams.update(matplotlib_stil())


def speed_profil(ses) -> tuple[np.ndarray, dict]:
    """VORGEHEN 1-2: Pole-Runde, Speed ueber RelativeDistance interpoliert."""
    lap = ses.laps.pick_fastest()
    tel = lap.get_telemetry().add_distance().add_relative_distance()
    rel = tel["RelativeDistance"].to_numpy()
    v = tel["Speed"].to_numpy()
    ok = ~np.isnan(rel) & ~np.isnan(v)
    grid = np.linspace(0, 1, GRID_N)
    profil = np.interp(grid, rel[ok], v[ok])

    ci = ses.get_circuit_info()
    meta = {
        "jahr": ses.event.year, "pole": str(lap["Driver"]),
        "zeit_s": round(lap["LapTime"].total_seconds(), 3),
        "laenge_m": round(float(tel["Distance"].max()), 0),
        "kurven": len(ci.corners),
        "vmax": round(float(v.max()), 1), "vmean": round(float(np.nanmean(v)), 1),
    }
    return profil, meta


def zeichne_profile(ax_profil, ax_delta, grid: np.ndarray,
                    profile: dict[int, np.ndarray]) -> None:
    """VORGEHEN 3: Profile uebereinander, plus Delta-Panel."""
    jahre = sorted(profile)
    for i, y in enumerate(jahre):
        ax_profil.plot(grid * 100, profile[y], label=str(y), lw=1.6,
                       color=SERIEN[i])
    ax_profil.set_ylabel("Speed [km/h]")
    ax_profil.set_title(f"{GP} - Speed-Profil der Pole-Runde", loc="left",
                        color=FG, fontsize=13, pad=10)
    ax_profil.legend(loc="lower center", ncol=len(jahre), frameon=False,
                     labelcolor=FG)

    alt, neu = jahre[0], jahre[-1]
    delta = profile[neu] - profile[alt]
    ax_delta.plot(grid * 100, delta, color=SERIEN[2], lw=1.4)
    ax_delta.axhline(0, color=MUTED, lw=0.8)
    peak = int(np.argmax(np.abs(delta)))
    ax_delta.annotate(
        f"+{delta[peak]:.0f} km/h bei {grid[peak] * 100:.0f}%\n"
        f"(vor 2023 entfernte Schikane)",
        (grid[peak] * 100, delta[peak]), xytext=(-15, -15),
        textcoords="offset points", ha="right", va="top", color=MUTED,
        fontsize=9)
    ax_delta.set_ylabel(f"Delta {neu}-{alt} [km/h]")
    ax_delta.set_xlabel("Rundenanteil [%]")
    for ax in (ax_profil, ax_delta):
        ax.grid(alpha=0.3, linewidth=0.8, color=GRID)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)


def zeichne_regelaera(ax, vergleich: pd.DataFrame) -> None:
    """AUSBAUSTUFE: mittlere Geschwindigkeit 2024 gegen 2018, nur Strecken
    ohne bekannte Layout-Aenderung."""
    v = vergleich.sort_values("vmean_delta")
    farben = [POSITIV if x > 0 else SERIEN[1] for x in v["vmean_delta"]]
    ax.barh(v["strecke"], v["vmean_delta"], color=farben, height=0.6)
    ax.axvline(0, color=MUTED, lw=0.8)
    ax.margins(x=0.08)
    ax.set_xlabel("Delta mittlere Rundengeschwindigkeit 2024-2018 [km/h]")
    ax.set_title("Ground-Effect-Aera gegen Hochabtrieb-Ende: 6 unveraenderte "
                "Strecken", loc="left", color=FG, fontsize=13, pad=10)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="x", alpha=0.3, linewidth=0.8, color=GRID)
    ax.set_axisbelow(True)


def main():
    f1lab.enable_cache()

    print(f"[1/3] {GP}: {YEARS} laden ...")
    grid = np.linspace(0, 1, GRID_N)
    profile, meta = {}, []
    for y in YEARS:
        ses = f1lab.load(y, GP, "Q", telemetry=True)
        p, m = speed_profil(ses)
        profile[y] = p
        meta.append(m)
    info = pd.DataFrame(meta)
    print(info.to_string(index=False))

    print(f"\n[2/3] Regelaera-Vergleich ueber {len(REGEL_STRECKEN)} "
         f"unveraenderte Strecken ...")
    zeilen = []
    for gp in REGEL_STRECKEN:
        werte = {}
        for y in YEARS:
            ses = f1lab.load(y, gp, "Q", telemetry=True)
            _, m = speed_profil(ses)
            werte[y] = m
        zeilen.append({
            "strecke": gp,
            "vmax_delta": werte[YEARS[-1]]["vmax"] - werte[YEARS[0]]["vmax"],
            "vmean_delta": round(werte[YEARS[-1]]["vmean"] - werte[YEARS[0]]["vmean"], 1),
            "zeit_delta_s": round(werte[YEARS[-1]]["zeit_s"] - werte[YEARS[0]]["zeit_s"], 2),
        })
        print(f"      {gp:<12} {zeilen[-1]}")
    vergleich = pd.DataFrame(zeilen)
    print(f"\n      vmean auf allen {len(REGEL_STRECKEN)} Strecken "
         f"{'niedriger' if (vergleich['vmean_delta'] < 0).all() else 'gemischt'} "
         f"in {YEARS[-1]}; Rundenzeit steigt bei "
         f"{(vergleich['zeit_delta_s'] > 0).sum()}/{len(vergleich)}")

    print("\n[3/3] Grafik ...")
    fig = plt.figure(figsize=(15, 11))
    gs = fig.add_gridspec(3, 1, height_ratios=[1.3, 0.8, 1.1], hspace=0.5)
    zeichne_profile(fig.add_subplot(gs[0]), fig.add_subplot(gs[1]), grid, profile)
    zeichne_regelaera(fig.add_subplot(gs[2]), vergleich)
    fig.suptitle("Streckenvergleich ueber Jahre: Layout gegen Regelwerk",
                x=0.09, ha="left", fontsize=16, color=FG, y=0.995)
    path = OUT / "streckenvergleich_jahre.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"\n      -> {path}")


if __name__ == "__main__":
    main()
