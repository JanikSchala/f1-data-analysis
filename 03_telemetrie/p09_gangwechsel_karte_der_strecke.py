"""
P09 - Gangwechsel-Karte der Strecke
===================================

Die Ideallinie in XY-Koordinaten, eingefaerbt nach eingelegtem Gang - das ikonische FastF1-Bild.

Kategorie:   Telemetrie
Niveau:      Fortgeschritten
Aufwand:     2-3 h
Schwerpunkt: Datenanalyse, Engineering

WARUM DAS LOHNT
Optisch stark und technisch lehrreich zugleich: LineCollection, Positionsdaten und diskrete Farbskalen. Das Bild, das man sich an die Wand haengt.

VORGEHEN
  1. Telemetrie der schnellsten Runde holen (get_telemetry mergt Car+Pos)
  2. X/Y zu Segment-Paaren umbauen
  3. LineCollection mit diskreter Colormap ueber nGear
  4. Achsen gleich skalieren, Colorbar mit Gaengen 1-8

GENUTZTE FASTF1-BAUSTEINE
  - Lap.get_telemetry
  - Telemetry X/Y/nGear/Speed/Throttle/DRS
  - matplotlib LineCollection

AUSBAUSTUFE  [umgesetzt]
Dieselbe Karte fuer Speed, Throttle und DRS, als 2x2-Grid.

Zwei Anpassungen gegenueber der urspruenglichen Fassung:

Erstens die Session: Belgien 2024 Q war komplett nass (daher DRS durchgehend
deaktiviert - der DRS-Kanal der Pole-Runde stand die ganze Runde auf einem
einzigen, nicht dokumentierten Wert). Fuer eine Karte, die gerade DRS zeigen
soll, ist das ungeeignet. Monza 2024 Q ersetzt die Wahl: die Strecke mit den
laengsten DRS-Zonen im Kalender, trocken, mit echter Variation im Kanal.

Zweitens der DRS-Kanal selbst: FastF1s eigene Dokumentation bezeichnet die
Codes unterhalb von 10 als unsicher ("Unknown Distinction", "Noted
Sometimes"). Verwendet wird hier die in der Community uebliche Lesart -
gerade Werte ab 10 heissen offen, 8 heisst erkannt/im Aktivierungsbereich
aber noch nicht offen, alles andere zu -, mit drei statt zwei Farben, damit
der Unterschied zwischen "im Fenster" und "tatsaechlich offen" sichtbar
bleibt statt in einem einzigen "aktiv" zu verschwinden. Die Klassifikation
selbst ist f1lab.drs_state() (core.py, netzlos getestet) - dieselbe Funktion
nutzt auch der DRS-Kanal der Telemetrie-Seite im Dashboard.

Bei der Farbwahl zeigt sich ein Grenzfall von f1lab.design: die Rampe dort
ist fuer Groessen mit echtem Verlauf gedacht (Hoehenmeter, Degradation) und
funktioniert fuer Speed und Throttle genau deshalb gut - das sind echte
Kontinua. Gang ist das nicht: sieben klar getrennte, gleich wichtige
Zustaende, keine Groesse mit Zwischenwerten. Mit der einfarbigen Rampe waren
Gang 6, 7 und 8 auf der Karte praktisch nicht mehr zu unterscheiden (die
meiste Distanz liegt in den oberen Gaengen). Deshalb hier die Ausnahme:
Gang bekommt eine mehrfarbige, wahrnehmungsuniforme Skala (viridis) statt der
Rampe - dieselbe Wahl, die die urspruengliche Fassung schon getroffen hatte,
und aus genau diesem Grund richtig. DRS bleibt ein Zustand mit den
Wertungsfarben (zu/erkannt/offen), nicht mit einer Skala.
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
from matplotlib.collections import LineCollection
from matplotlib.colors import (
    BoundaryNorm,
    LinearSegmentedColormap,
    ListedColormap,
    Normalize,
)

import f1lab
from f1lab.design import FG, MUTED, POSITIV, RAMPE, SERIEN, matplotlib_stil

warnings.filterwarnings("ignore")

OUT = Path(__file__).parent / "out"
OUT.mkdir(exist_ok=True)

SEASON, EVENT, IDENT = 2024, "Monza", "Q"

plt.rcParams.update(matplotlib_stil())


def zeichne_karte(ax, x: np.ndarray, y: np.ndarray, werte: np.ndarray, *,
                  cmap, norm, titel: str, cbar_label: str,
                  cbar_ticks=None, boundaries=None,
                  cbar_ticklabels=None) -> None:
    """Eine eingefaerbte Streckenkarte - Kern von P09, wiederverwendet fuer
    alle vier Kanaele dieses Skripts."""
    punkte = np.column_stack([x, y]).reshape(-1, 1, 2)
    segmente = np.concatenate([punkte[:-1], punkte[1:]], axis=1)
    lc = LineCollection(segmente, cmap=cmap, norm=norm, linewidths=4)
    lc.set_array(werte[:-1])
    ax.add_collection(lc)

    pad = 0.05 * max(x.max() - x.min(), y.max() - y.min())
    ax.set_xlim(x.min() - pad, x.max() + pad)
    ax.set_ylim(y.min() - pad, y.max() + pad)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(titel, loc="left", color=FG, fontsize=13, pad=8)

    cb = ax.figure.colorbar(lc, ax=ax, boundaries=boundaries, fraction=0.045,
                            pad=0.02)
    if cbar_ticks is not None:
        cb.set_ticks(cbar_ticks)
    if cbar_ticklabels is not None:
        cb.set_ticklabels(cbar_ticklabels)
    cb.set_label(cbar_label, color=MUTED, fontsize=9)
    cb.ax.tick_params(colors=MUTED, length=0)
    cb.outline.set_visible(False)


def main():
    f1lab.enable_cache()

    print(f"[1/2] {EVENT} {SEASON} {IDENT} laden (mit Telemetrie) ...")
    ses = f1lab.load(SEASON, EVENT, IDENT, telemetry=True)
    lap = ses.laps.pick_fastest()
    tel = lap.get_telemetry()
    print(f"      Schnellste Runde: {lap['Driver']}, {lap['LapTime']}")

    x = tel["X"].to_numpy(dtype=float)
    y = tel["Y"].to_numpy(dtype=float)
    gear = tel["nGear"].to_numpy(dtype=float)
    speed = tel["Speed"].to_numpy(dtype=float)
    throttle = tel["Throttle"].to_numpy(dtype=float)
    drs = f1lab.drs_state(tel["DRS"].to_numpy())

    offen_pct = 100 * (drs == 2).mean()
    print(f"      DRS offen auf {offen_pct:.0f}% der Runde "
         f"(erkannt/im Fenster: {100 * (drs == 1).mean():.0f}%)")

    print("[2/2] Grafik ...")
    fig, ax = plt.subplots(2, 2, figsize=(14, 13))

    n_gaenge = int(gear.max() - gear.min() + 1)
    gang_cmap = plt.get_cmap("viridis", n_gaenge)
    zeichne_karte(
        ax[0, 0], x, y, gear, cmap=gang_cmap,
        norm=BoundaryNorm(np.arange(gear.min() - 0.5, gear.max() + 1.5), n_gaenge),
        titel="Gang", cbar_label="Gang",
        cbar_ticks=np.arange(gear.min(), gear.max() + 1))

    rampe_cmap = LinearSegmentedColormap.from_list("rampe", RAMPE)
    zeichne_karte(
        ax[0, 1], x, y, speed, cmap=rampe_cmap,
        norm=Normalize(speed.min(), speed.max()),
        titel="Speed", cbar_label="km/h")

    zeichne_karte(
        ax[1, 0], x, y, throttle, cmap=rampe_cmap,
        norm=Normalize(0, 100),
        titel="Throttle", cbar_label="%")

    drs_cmap = ListedColormap([MUTED, SERIEN[1], POSITIV])
    zeichne_karte(
        ax[1, 1], x, y, drs.astype(float), cmap=drs_cmap,
        norm=BoundaryNorm([-0.5, 0.5, 1.5, 2.5], drs_cmap.N),
        titel="DRS", cbar_label="", cbar_ticks=[0, 1, 2],
        cbar_ticklabels=["zu", "erkannt", "offen"])

    fig.suptitle(f"{ses.event['EventName']} {SEASON} {IDENT} - "
                f"{lap['Driver']}, schnellste Runde", x=0.08, ha="left",
                fontsize=16, color=FG, y=0.99)
    plt.tight_layout()
    path = OUT / "gangwechsel_karte.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"\n      -> {path}")


if __name__ == "__main__":
    main()
