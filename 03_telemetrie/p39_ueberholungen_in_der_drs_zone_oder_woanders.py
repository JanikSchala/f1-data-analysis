"""
P39 - Ueberholungen: in der DRS-Zone oder woanders?
====================================================

DRS gilt als der groesste Ueberholtreiber der Hybrid-Aera - stimmt das mit den tatsaechlichen Ueberholorten ueberein, oder passiert ein relevanter Teil ausserhalb der Zone?

Kategorie:   Telemetrie
Niveau:      Profi
Aufwand:     4-5 h
Schwerpunkt: Datenanalyse, Engineering

WARUM DAS LOHNT
Kombiniert P20s Ueberholerkennung (Positionswechsel je Runde) mit P10s
DRS-Zonen zu einer raeumlichen Frage: WO auf der Strecke passiert ein
Ueberholvorgang, nicht nur WANN. Der Positionswechsel selbst ist nur
rundenweise bekannt - die genaue Stelle muss aus der Telemetrie des
Ueberholers rekonstruiert werden (FastF1s ``DriverAhead``-Kanal), und
dabei zeigt sich unterwegs ein methodischer Fallstrick mit DRS-Zonen aus
einer Rennrunde (siehe AUSBAUSTUFE).

VORGEHEN
  1. Ueberholereignisse einer Session laden (f1lab.overtake_events, P20)
  2. Je Ereignis in der Telemetrie des Ueberholers die letzte Stelle finden,
     an der der Ueberholte noch direkt davor war (DriverAhead-Kanal)
  3. DRS-Zonen derselben Strecke bestimmen (f1lab.drs_zones, P10)
  4. Ueberholorte gegen die Zonen pruefen und den Anteil "in der Zone"
     berechnen

GENUTZTE FASTF1-BAUSTEINE
  - Telemetry.add_driver_ahead (DriverAhead, DistanceToDriverAhead)
  - Telemetry.add_distance
  - Laps.pick_laps/pick_drivers

AUSBAUSTUFE  [umgesetzt]
Denselben Scan auf alle telemetriefaehigen Rennen der Saison 2024 anwenden
(siehe P38 fuer dieselbe 12-Strecken-Stichprobe) und pruefen, ob der
DRS-Anteil je nach Streckencharakter schwankt. Dritte AUSBAUSTUFE:
Ueberholorte gegen Bremszonen statt DRS-Zonen pruefen, gegen eine
Zufalls-Baseline statt eine willkuerliche Pufferzone.

Monza 2024 R (Referenz): 157 Ueberholungen, 126 davon (80.3%) in der
Telemetrie eindeutig lokalisiert - der Rest hatte im entscheidenden Moment
keinen sauberen DriverAhead-Treffer unter 30m (mehrere Positionswechsel in
derselben Runde, oder eine Datenluecke). Von den 126 lokalisierten liegen
101 (80.2%) in einer DRS-Zone.

Dabei zeigte sich unterwegs ein methodischer Fallstrick: DRS-Zonen zuerst
aus der Rennsession selbst bestimmt (wie in P10/P37 fuer Qualifying ueblich)
ergab NULL Zonen, obwohl Monza zwei besitzt. Grund: DRS braucht im Rennen
einen Rueckstand unter 1s auf das Auto davor, die schnellste Rennrunde
entsteht aber typisch in freier Fahrt genau OHNE Vordermann - DRS bleibt
dann auf der ganzen Runde zu, egal wie nah die physische Zone ist.
Behoben durch die Qualifying-Session desselben Events als DRS-Referenz
(dort gilt die Abstandsregel nicht), automatisch ueber die Rundennummer
statt den Streckennamen aufgeloest.

Saison-Scan ueber alle 12 telemetriefaehigen Rennen: DRS-Anteil je Strecke
zwischen 40.0% (Monaco, aber nur 5 lokalisierte Ereignisse - kaum
belastbar) und 88.4% (Spanien, 164 lokalisierte Ereignisse, deutlich
robuster). Median der Streckenwerte 71.6%, gepoolt (nach Ereigniszahl
gewichtet statt jede Strecke gleich zu zaehlen) 76.0% (812/1069) - beide
Zahlen zusammen zeigen: rund drei von vier Ueberholungen passieren
tatsaechlich in der DRS-Zone, aber ein spuerbares Viertel nicht (Fahrfehler
des Vordermanns, spaeteres Bremsen ohne DRS-Hilfe, oder Strategieluecken
nach Boxenstopps ausserhalb der Zone). Die beiden ueberholstaerksten
Strecken der Saison (Spanien 196, Bahrain 180 - siehe P38) haben auch die
hoechsten DRS-Anteile (88.4%, 83.6%) - ein Hinweis, aber aus n=12 kein
Beweis, dass DRS-Effektivitaet und rohe Ueberholzahl zusammenhaengen.

Abdeckung (lokalisiert/gesamt) liegt ueber alle 12 Strecken zwischen 75.0%
und 90.0%, ohne erkennbares Muster nach Streckentyp - die nicht lokalisierte
Restmenge ist am ehesten Methodenrauschen (Mehrfachueberholungen in einer
Runde, DriverAhead-Datenluecken), keine systematische Verzerrung in eine
Richtung.

DRITTE AUSBAUSTUFE  [umgesetzt]
Naheliegende Anschlussfrage: passieren Ueberholungen auch nah an
Bremszonen (f1lab.driver_braking_zones, P08), nicht nur in DRS-Zonen? Ein
erster Versuch (Ueberholort einfach als "innerhalb Bremszone + 30-100m
Puffer davor" zaehlen, wie beim DRS-Check) ergab fuer Monza nur 5-7%
Treffer - auf den ersten Blick ein Widerspruch zum DRS-Befund oben, aber
tatsaechlich die falsche Frage: eine Ueberholung MUSS nicht direkt am
Bremspunkt abgeschlossen sein, sie kann bereits auf der Geraden vorher
passieren. Richtiggestellt mit f1lab.lead_distance_to_zone(): statt
"drinnen oder draussen" wird der Abstand zur naechsten Bremszone in
Fahrtrichtung gemessen (mit Rundenumbruch).

Median dieses Abstands ueber die 126 lokalisierten Monza-Ueberholungen:
621m - und dagegen gestellt eine Zufalls-Baseline (20 000 gleichverteilte
Punkte auf der 5723m-Strecke, Median 527m, seed=42): die echten
Ueberholorte liegen NICHT naeher an einer Bremszone als der Zufall
(Anteil unter 200m: 10.3% echt gegen 21.0% Zufall - echte Ueberholungen
sind im letzten kurzen Stueck vor dem Bremspunkt sogar SELTENER als
Zufall, plausibel: wer bis dahin noch nicht vorbei ist, schafft es meist
nicht mehr vor der Bremszone), aber deutlich haeufiger im mittleren
Bereich (Anteil unter 800m: 92.1% echt gegen 68.9% Zufall). Das deckt sich
mit dem DRS-Befund, statt ihn zu widersprechen: die 700-800m-Baende vor
Monzas Bremszonen 1 und 4 SIND die DRS-Zonen (746m/766m lang) - "nah an
einer Bremszone, aber nicht direkt davor" beschreibt denselben Ort wie
"auf der DRS-Geraden".
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
from f1lab.design import FG, GRID, POSITIV, SERIEN, matplotlib_stil

warnings.filterwarnings("ignore")

OUT = Path(__file__).parent / "out"
OUT.mkdir(exist_ok=True)

REFERENZ = (2024, "Italy", "R")          # Monza: DRS-lastigste Strecke im Cache

plt.rcParams.update(matplotlib_stil())


def zeichne_streckenkarte(ax, ref_xy: np.ndarray, dist: np.ndarray,
                          zonen: pd.DataFrame, orte: pd.DataFrame) -> None:
    """VORGEHEN 2-4: Ueberholorte auf der Streckenkarte, DRS-Zonen markiert."""
    ax.plot(ref_xy[:, 0], ref_xy[:, 1], color=GRID, lw=6, zorder=1,
           solid_capstyle="round")
    for _, z in zonen.iterrows():
        maske = (dist >= z["start_m"]) & (dist <= z["end_m"])
        ax.plot(ref_xy[maske, 0], ref_xy[maske, 1], color=SERIEN[0], lw=6,
               zorder=2, solid_capstyle="butt")

    for in_zone, gruppe in orte.groupby("in_drs_zone"):
        idx = [int(np.argmin(np.abs(dist - d))) for d in gruppe["distance_m"]]
        ax.scatter(ref_xy[idx, 0], ref_xy[idx, 1],
                  color=POSITIV if in_zone else SERIEN[1], s=45, zorder=3,
                  edgecolor=FG, linewidth=0.5,
                  label="in DRS-Zone" if in_zone else "ausserhalb")

    ax.set_aspect("equal")
    ax.axis("off")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=2,
             frameon=False, labelcolor=FG, fontsize=10)
    ax.set_title(f"{REFERENZ[1]} {REFERENZ[0]} {REFERENZ[2]}: Ueberholorte "
                "gegen DRS-Zonen (blau)", loc="left", color=FG, fontsize=13,
                pad=10)


def zeichne_saisonvergleich(ax, ergebnisse: pd.DataFrame) -> None:
    """AUSBAUSTUFE: DRS-Anteil an Ueberholungen je Strecke, Saison 2024."""
    e = ergebnisse.sort_values("drs_pct")
    ax.barh(e["gp"], e["drs_pct"], color=SERIEN[0], height=0.6)
    ax.set_xlabel("Anteil Ueberholungen in der DRS-Zone [%]")
    ax.set_title("AUSBAUSTUFE: DRS-Anteil je Strecke, Saison 2024", loc="left",
                color=FG, fontsize=13, pad=10)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="x", alpha=0.3, linewidth=0.8, color=GRID)
    ax.set_axisbelow(True)


def zeichne_bremszonen_abstand(ax, vorlauf_echt: np.ndarray,
                               vorlauf_zufall: np.ndarray) -> None:
    """DRITTE AUSBAUSTUFE: Abstand zur naechsten Bremszone, echte
    Ueberholorte gegen Zufalls-Baseline."""
    bins = np.linspace(0, max(vorlauf_echt.max(), 1500), 25)
    ax.hist(vorlauf_zufall, bins=bins, density=True, color=GRID, alpha=0.7,
           label="Zufalls-Baseline (gleichverteilt)")
    ax.hist(vorlauf_echt, bins=bins, density=True, color=SERIEN[0], alpha=0.7,
           label="Echte Ueberholorte")
    ax.set_xlabel("Abstand zur naechsten Bremszone [m]")
    ax.set_ylabel("Dichte")
    ax.set_title("DRITTE AUSBAUSTUFE: Bremszonen-Naehe, echt gegen Zufall",
                loc="left", color=FG, fontsize=13, pad=10)
    ax.legend(loc="upper right", frameon=False, labelcolor=FG, fontsize=10)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(alpha=0.3, linewidth=0.8, color=GRID)
    ax.set_axisbelow(True)


def main():
    f1lab.enable_cache()

    print(f"[1/3] {REFERENZ[1]} {REFERENZ[0]} {REFERENZ[2]} laden, "
         "Ueberholorte bestimmen (VORGEHEN 1-2) ...")
    ses = f1lab.load(*REFERENZ, telemetry=True)
    events = f1lab.overtake_events(ses)
    orte = f1lab.overtake_locations(ses)
    print(f"      {len(events)} Ueberholungen, {len(orte)} lokalisiert "
         f"({100 * len(orte) / len(events):.1f}%)")

    print("\n[2/3] DRS-Zonen und Abgleich (VORGEHEN 3-4) ...")
    q = f1lab.load(REFERENZ[0], int(ses.event["RoundNumber"]), "Q",
                   telemetry=True)
    lap = q.laps.pick_fastest()
    zonen = f1lab.drs_zones(q, str(lap["Driver"]))
    print(zonen.round(0).to_string(index=False))
    anteil = 100 * orte["in_drs_zone"].mean()
    print(f"\n      {int(orte['in_drs_zone'].sum())}/{len(orte)} lokalisierte "
         f"Ueberholungen in einer DRS-Zone ({anteil:.1f}%)")

    ref_lap = ses.laps.pick_fastest()
    tel = ref_lap.get_telemetry().add_distance()
    ref_xy = tel[["X", "Y"]].to_numpy(dtype=float) / 10
    dist = tel["Distance"].to_numpy()

    print("\n[3/3] Saison-Scan ueber alle telemetriefaehigen Rennen 2024 "
         "(AUSBAUSTUFE) ...")
    inv = f1lab.cached_sessions()
    tel_rennen = sorted(inv[(inv["season"] == 2024) & (inv["ident"] == "R")
                            & inv["telemetry"]]["event"].unique())
    zeilen = []
    for gp in tel_rennen:
        s = f1lab.load(2024, gp, "R", telemetry=True)
        ev = f1lab.overtake_events(s)
        if ev.empty:
            continue
        o = f1lab.overtake_locations(s)
        if o.empty:
            continue
        zeilen.append({
            "gp": gp, "ueberholungen": len(ev), "lokalisiert": len(o),
            "abdeckung_pct": round(100 * len(o) / len(ev), 1),
            "drs_pct": round(100 * o["in_drs_zone"].mean(), 1),
        })
        print(f"      {gp:28s} {zeilen[-1]}")
    ergebnisse = pd.DataFrame(zeilen)
    print(f"\n      DRS-Anteil ueber {len(ergebnisse)} Strecken: "
         f"{ergebnisse['drs_pct'].min():.1f}-{ergebnisse['drs_pct'].max():.1f}%, "
         f"Median {ergebnisse['drs_pct'].median():.1f}%")

    print("\nDRITTE AUSBAUSTUFE: Abstand zur naechsten Bremszone, echt gegen "
         "Zufall ...")
    ref_bz = f1lab.driver_braking_zones(ses, str(ref_lap["Driver"]))
    strecke_m = float(dist.max())
    vorlauf_echt = f1lab.lead_distance_to_zone(
        orte["distance_m"], ref_bz["start_m"], strecke_m)
    rng = np.random.default_rng(42)
    zufall = rng.uniform(0, strecke_m, 20_000)
    vorlauf_zufall = f1lab.lead_distance_to_zone(
        zufall, ref_bz["start_m"], strecke_m)
    print(f"      Median Abstand: echt {np.median(vorlauf_echt):.0f}m gegen "
         f"Zufall {np.median(vorlauf_zufall):.0f}m")
    print(f"      Anteil < 200m:  echt {100 * (vorlauf_echt < 200).mean():.1f}% "
         f"gegen Zufall {100 * (vorlauf_zufall < 200).mean():.1f}%")
    print(f"      Anteil < 800m:  echt {100 * (vorlauf_echt < 800).mean():.1f}% "
         f"gegen Zufall {100 * (vorlauf_zufall < 800).mean():.1f}%")

    print("\nGrafik ...")
    fig = plt.figure(figsize=(15, 17))
    gs = fig.add_gridspec(3, 1, height_ratios=[1.3, 1, 1], hspace=0.4)
    zeichne_streckenkarte(fig.add_subplot(gs[0]), ref_xy, dist, zonen, orte)
    zeichne_saisonvergleich(fig.add_subplot(gs[1]), ergebnisse)
    zeichne_bremszonen_abstand(fig.add_subplot(gs[2]), vorlauf_echt,
                               vorlauf_zufall)
    fig.suptitle("Ueberholungen: in der DRS-Zone oder woanders?", x=0.09,
                ha="left", fontsize=16, color=FG, y=0.995)
    plt.tight_layout()
    path = OUT / "ueberholungen_drs_zone.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"\n      -> {path}")


if __name__ == "__main__":
    main()
