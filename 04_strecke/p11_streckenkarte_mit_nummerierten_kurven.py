"""
P11 - Streckenkarte mit nummerierten Kurven
===========================================

get_circuit_info() liefert offizielle Kurvenpositionen und Marshal-Sektoren - daraus wird eine beschriftete Streckenkarte.

Kategorie:   Strecke & Position
Niveau:      Einsteiger
Aufwand:     2 h
Schwerpunkt: Datenanalyse, Engineering

WARUM DAS LOHNT
Kurvennummern sind die Sprache im Team ('T7 Untersteuern'). Wer Telemetrie auf Kurven mappen kann, macht Analysen fuer Ingenieure lesbar.

VORGEHEN
  1. Circuit-Info und Positionsdaten der schnellsten Runde laden
  2. Track um circuit_info.rotation drehen, damit die Karte richtig liegt
  3. Kurvennummern mit Versatz ausserhalb der Strecke annotieren
  4. Marshal-Sektoren farbig markieren

GENUTZTE FASTF1-BAUSTEINE
  - Session.get_circuit_info
  - CircuitInfo.corners
  - CircuitInfo.marshal_sectors
  - CircuitInfo.rotation

AUSBAUSTUFE  [umgesetzt]
Jedem Telemetriepunkt die naechstgelegene Kurve zuordnen und eine Tabelle
"Minimalgeschwindigkeit je Kurve" bauen.

Umgesetzt als f1lab.corner_speeds() (Fenster von +/- 60 m um die auf die
Referenzrunde projizierte Kurvendistanz, nicht per exakter naechster-
Nachbar-Zuordnung - siehe Docstring dort) statt hier eine zweite,
Track-spezifische Variante zu schreiben. Dieselbe Funktion ist die
Grundlage von P12s Fahrer-x-Kurve-Heatmap; P11 zeigt nur die eine Zeile
des Referenzfahrers - der ganze Sinn der Karte ist schliesslich, seine
eigene schnellste Runde zu verstehen, nicht das Feld zu vergleichen (dafuer
ist P12 da).

Zweite Ergaenzung, die im urspruenglichen VORGEHEN schon Punkt 4 war, aber
im Code fehlte: Marshal-Sektoren sind jetzt als abwechselnd eingefaerbte
Streckenabschnitte sichtbar, mit Nummern an den Sektorgrenzen - dieselbe
Projektions-Idee wie bei den Kurven, nur auf die Sektorgrenzen angewendet.
Die Projektion selbst steckt seit der App-Integration in
f1lab.marshal_sector_labels() statt hier lokal (zweiter Konsument: die
Strecke-Seite im Dashboard braucht dieselbe Funktion).

Session auf Bahrain 2024 Q gewechselt (Zandvoort hat rotation=0 - VORGEHEN
Punkt 2 waere unsichtbar geblieben, die Drehung haette nichts zu tun gehabt).
Bahrain dreht die Karte um 92 Grad.

ZWEITE AUSBAUSTUFE: ``CircuitInfo.marshal_lights`` (Positionen der
Gelbflaggen-Lichttafeln) war im ganzen Repository bislang ungenutzt, obwohl
FastF1 sie liefert. Als vierte Punktliste ergaenzt, dieselbe Projektions-
Idee wie Kurven/Marshal-Sektoren - anders als die Sektorgrenzen (kein
eigenes ``Distance``-Feld) auf dieselbe Weise wie diese ueber die naechste
Referenzrunden-Position bestimmt. `f1lab.marshal_light_labels()`, zweiter
Konsument die Strecke-Seite im Dashboard.
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
from matplotlib.collections import LineCollection

import f1lab
from f1lab.design import BG_HELL, FG, MUTED, PHASE, SERIEN, matplotlib_stil

warnings.filterwarnings("ignore")

OUT = Path(__file__).parent / "out"
OUT.mkdir(exist_ok=True)

SEASON, EVENT, IDENT = 2024, "Bahrain", "Q"
CORNER_OFFSET_M = 350.0

plt.rcParams.update(matplotlib_stil())


def rotate(xy: np.ndarray, angle_deg: float) -> np.ndarray:
    a = np.deg2rad(angle_deg)
    rot = np.array([[np.cos(a), np.sin(a)], [-np.sin(a), np.cos(a)]])
    return xy @ rot


def zeichne_karte(ax, ref: pd.DataFrame, ci_rotation: float,
                  corners: pd.DataFrame, sektoren: pd.DataFrame,
                  lichter: pd.DataFrame) -> None:
    xy = rotate(ref[["X", "Y"]].to_numpy(dtype=float), ci_rotation)
    dist = ref["Distance"].to_numpy()

    # VORGEHEN 4: Marshal-Sektoren als abwechselnd gefaerbte Abschnitte -
    # keine Kategorie mit eigener Bedeutung, deshalb zwei neutrale Toene im
    # Wechsel statt der Kategoriepalette.
    punkte = xy.reshape(-1, 1, 2)
    segmente = np.concatenate([punkte[:-1], punkte[1:]], axis=1)
    mitte = (dist[:-1] + dist[1:]) / 2
    grenzen = sektoren["distance"].to_numpy()
    sektor_idx = np.searchsorted(grenzen, mitte, side="right")
    palette = [BG_HELL, MUTED]
    farben = [palette[i % 2] for i in sektor_idx]
    lc = LineCollection(segmente, colors=farben, linewidths=7)
    ax.add_collection(lc)

    for s in sektoren.itertuples():
        idx = np.argmin(np.abs(dist - s.distance))
        ax.scatter(*xy[idx], s=40, color=FG, zorder=3, marker="|")

    # ZWEITE AUSBAUSTUFE: Gelbflaggen-Lichttafeln - eine vierte Punktliste
    # (siehe f1lab.marshal_light_labels), dieselbe Projektions-Idee wie
    # Kurven/Marshal-Sektoren, aber eine eigene Bedeutung: hier haengt eine
    # Lichttafel, keine Streckenmarkierung.
    for lp in lichter.itertuples():
        idx = np.argmin(np.abs(dist - lp.distance))
        ax.scatter(*xy[idx], s=90, color=PHASE["gelb"], marker="^",
                  edgecolors=FG, linewidths=0.8, zorder=4)

    # VORGEHEN 3: Kurvennummern mit Versatz ausserhalb der Strecke
    for c in corners.itertuples():
        ang = np.deg2rad(c.Angle)
        off = rotate(np.array([[CORNER_OFFSET_M * np.cos(ang),
                                CORNER_OFFSET_M * np.sin(ang)]]),
                    ci_rotation)[0]
        base = rotate(np.array([[c.X, c.Y]]), ci_rotation)[0]
        tip = base + off
        ax.plot([base[0], tip[0]], [base[1], tip[1]], color=MUTED, lw=1)
        ax.scatter(*tip, s=220, color=SERIEN[1], zorder=4)
        ax.text(*tip, c.label, color=FG, va="center", ha="center",
               fontsize=8, zorder=5, fontweight="bold")

    ax.set_aspect("equal")
    ax.axis("off")


def main():
    f1lab.enable_cache()

    print(f"[1/4] {EVENT} {SEASON} {IDENT} laden (mit Telemetrie) ...")
    ses = f1lab.load(SEASON, EVENT, IDENT, telemetry=True)
    ci = ses.get_circuit_info()
    ref_lap = ses.laps.pick_fastest()
    ref = ref_lap.get_telemetry().add_distance()
    print(f"      Referenzrunde: {ref_lap['Driver']}, Rotation "
         f"{ci.rotation:.0f} Grad, {len(ci.corners)} Kurven, "
         f"{len(ci.marshal_sectors)} Marshal-Sektoren")

    print("[2/4] Kurven, Marshal-Sektoren und Lichttafeln projizieren ...")
    corners = f1lab.corner_labels(ses)
    sektoren = f1lab.marshal_sector_labels(ses)
    lichter = f1lab.marshal_light_labels(ses)
    print(f"      {len(lichter)} Gelbflaggen-Lichttafeln (ZWEITE AUSBAUSTUFE)")

    print("\n[3/4] Minimalgeschwindigkeit je Kurve (AUSBAUSTUFE) ...")
    speeds = f1lab.corner_speeds(ses)
    referenz_speeds = speeds.loc[str(ref_lap["Driver"])].reindex(corners["label"])
    tabelle = corners[["label", "Distance"]].copy()
    tabelle["v_min_kmh"] = referenz_speeds.to_numpy()
    print(tabelle.round(1).to_string(index=False))

    print("\n[4/4] Grafik ...")
    fig, ax = plt.subplots(figsize=(11, 11))
    zeichne_karte(ax, ref, ci.rotation, corners, sektoren, lichter)
    ax.set_title(f"{ses.event['EventName']} {SEASON} - Kurvenlayout "
                f"({len(sektoren)} Marshal-Sektoren)", loc="left", color=FG,
                fontsize=15, pad=14)
    plt.tight_layout()
    path = OUT / "streckenkarte.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"\n      -> {path}")


if __name__ == "__main__":
    main()
