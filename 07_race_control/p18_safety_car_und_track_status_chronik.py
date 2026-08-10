"""
P18 - Safety-Car- und Track-Status-Chronik
==========================================

Alle gelben Phasen, VSC- und SC-Perioden exakt als Rundenintervalle - inklusive Auswirkung auf die Feldstreckung.

Kategorie:   Race Control & Regeln
Niveau:      Fortgeschritten
Aufwand:     3 h
Schwerpunkt: Strategie, Datenanalyse

WARUM DAS LOHNT
Safety Cars entscheiden Rennen. Wer sie automatisch aus track_status extrahiert und deren Effekt beziffert, liefert Strategie-Input statt nur Deskription.

VORGEHEN
  1. track_status decodieren (1 gruen, 2 gelb, 4 SC, 5 rot, 6/7 VSC)
  2. Aufeinanderfolgende gleiche Status zu Intervallen zusammenfassen
  3. Intervalle auf Rundennummern mappen
  4. Feldstreckung (Abstand P1 zu Letztem) vor/nach SC vergleichen

GENUTZTE FASTF1-BAUSTEINE
  - Session.track_status
  - Session.session_status
  - Laps.pick_track_status
  - Laps TrackStatus

AUSBAUSTUFE  [umgesetzt]
Berechne pro Fahrer, ob das Safety Car ihm Zeit geschenkt oder gekostet hat -
abhaengig davon, ob er direkt davor gestoppt hatte.

VORGEHEN 2 hatte einen echten Bug, kein Stilproblem: ``track_status`` ist ein
Log von Zustandswechseln (eine Zeile pro Wechsel), keine Zeitreihe -
"SCDeployed" erscheint typischerweise genau einmal, auch wenn das Safety Car
neun Runden bleibt. Nach gleichem Status zu gruppieren und dessen eigenes
Min/Max zu nehmen (die urspruengliche Fassung, wortgleich zur Vorlage)
ergibt deshalb fast ueberall Intervalle der Laenge 0 - jede Phase endet an
ihrem eigenen Startzeitpunkt, weil kein Status zweimal hintereinander
auftaucht. Betraf nicht nur dieses Skript: ``f1lab.track_status_phases()``
hatte denselben Fehler und wird von der Rennverlauf-Seite im Dashboard
genutzt, die ihn mit einer kosmetischen Mindestbreite ueberdeckt hat, ohne
dass die zugrundeliegenden Intervalle je stimmten. Fix in
``f1lab.core.status_intervals()`` (neu, mit Tests): ein Intervall endet
dort, wo das naechste beginnt.

Mit der Korrektur zeigt Kanada 2024 R zwei echte Safety-Car-Phasen (Runde
25-29 und 54-58, vorher faelschlich je eine einzelne Runde). VORGEHEN 4 zeigt
dabei eine Feinheit: die rohe mittlere Feldstreckung waehrend SC ist wegen
des Ausloese-Zwischenfalls hoeher als im Gruenphasen-Schnitt (der Unfall
selbst reisst die Streckung sofort auf), nicht niedriger - das Feld
kompaktiert sich erst im Verlauf der SC-Runden. Die aussagekraeftige Zahl
ist deshalb nicht der Mittelwert, sondern die Streckung am Ende der
SC-Phase (kurz vor dem Restart): 19.7 s bzw. 100.0 s, gegenueber 92-150 s im
gruenen Rennverlauf an vergleichbarer Stelle - das Feld staucht sich auf gut
ein Fuenftel bis zwei Drittel zusammen, aber eben erst gegen Ende der Phase.

AUSBAUSTUFE: Kanada allein liefert zu wenige Boxenstopps in der Naehe eines
Safety Cars fuer eine belastbare Aussage (2 Faelle "kurz davor gestoppt").
Saison-Scan wie in P05/P31 ueber alle 2024er Rennen mit SC/VSC-Phase (11 von
24) behebt das: Positionsaenderung von der Runde vor dem eigenen Stopp bis
zum Ende der Neutralisation, getrennt nach Stopp-Zeitpunkt. Wer 1-3 Runden
VOR der Deklaration stoppt, verliert im Mittel 1.12 Positionen; wer WAEHREND
SC/VSC stoppt, verliert im Mittel nur 0.86 (n=75/93) - ein echter, wenn auch
moderater Vorteil dafuer, das SC-Fenster abzuwarten statt kurz davor zu
stoppen, aber kein Freifahrtschein: auch der guenstige Zeitpunkt kostet im
Schnitt noch fast eine Position, weil oft ein grosser Teil des Feldes
gleichzeitig stoppt und sich die Reihenfolge in der Box neu mischt.

VORGEHEN 4s Kernrechnung (Feldstreckung, Baseline-vs-Minimum-Kompaktierung)
steckt seit der App-Integration in f1lab.session.field_spread()/
sc_compaction() statt hier lokal (zweiter Konsument: die Race-Control-Seite
im Dashboard). Der Saison-Scan selbst bleibt skriptlokal, wie in P05/P16/
P31 begruendet.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")                      # kein Fenster, nur Dateien

import fastf1
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import f1lab
from f1lab.design import FG, GRID, MUTED, PHASE, SERIEN, matplotlib_stil

warnings.filterwarnings("ignore")
fastf1.set_log_level("ERROR")   # Saison-Scan laedt 24 Sessions, siehe P05

OUT = Path(__file__).parent / "out"
OUT.mkdir(exist_ok=True)

SEASON, EVENT, IDENT = 2024, "Canada", "R"
SAISON_RENNEN = [
    "Bahrain", "Saudi Arabia", "Australia", "Japan", "China", "Miami",
    "Emilia Romagna", "Monaco", "Canada", "Spain", "Austria", "Britain",
    "Hungary", "Belgium", "Netherlands", "Italy", "Azerbaijan", "Singapore",
    "United States", "Mexico", "Brazil", "Las Vegas", "Qatar", "Abu Dhabi",
]
FENSTER_DAVOR = 3     # Runden vor SC-Beginn, die noch als "kurz davor" zaehlen

plt.rcParams.update(matplotlib_stil())


def _position_bei(pos: pd.Series, drv: str, lap: float) -> float:
    try:
        v = pos.loc[(drv, lap)]
        return float(v.iloc[0]) if isinstance(v, pd.Series) else float(v)
    except KeyError:
        return float("nan")


def saison_scan() -> pd.DataFrame:
    """AUSBAUSTUFE: Positionswechsel Runde-vor-eigenem-Stopp bis Phasenende,
    getrennt nach Stopp-Zeitpunkt relativ zur Neutralisation."""
    zeilen = []
    for gp in SAISON_RENNEN:
        try:
            ses = f1lab.load(SEASON, gp, "R", telemetry=False)
            phasen = f1lab.track_status_phases(ses)
        except Exception:
            continue
        neutral = phasen[phasen["label"].isin(["safety car", "vsc"])]
        if neutral.empty:
            continue
        laps = ses.laps
        pos = laps.set_index(["Driver", "LapNumber"])["Position"]

        for p in neutral.itertuples():
            for drv in laps["Driver"].unique():
                davor = laps[(laps["Driver"] == drv)
                            & (laps["LapNumber"] >= p.lap_start - FENSTER_DAVOR)
                            & (laps["LapNumber"] < p.lap_start)
                            & (laps["PitInTime"].notna())]
                waehrend = laps[(laps["Driver"] == drv)
                                & (laps["LapNumber"] >= p.lap_start)
                                & (laps["LapNumber"] <= p.lap_end)
                                & (laps["PitInTime"].notna())]
                if len(davor) > 0:
                    klasse = "kurz davor"
                    stopp_runde = int(davor.iloc[0]["LapNumber"])
                elif len(waehrend) > 0:
                    klasse = "waehrend SC/VSC"
                    stopp_runde = int(waehrend.iloc[0]["LapNumber"])
                else:
                    continue
                pv = _position_bei(pos, drv, stopp_runde - 1)
                pn = _position_bei(pos, drv, p.lap_end)
                if np.isnan(pv) or np.isnan(pn):
                    continue
                zeilen.append({"gp": gp, "driver": drv, "klasse": klasse,
                               "delta_pos": pv - pn})
    return pd.DataFrame(zeilen)


def zeichne_chronik(ax, phasen: pd.DataFrame, spread: pd.Series) -> None:
    """VORGEHEN 1-3: Track-Status-Baender ueber der Feldstreckung."""
    for p in phasen.itertuples():
        if p.label == "gruen":
            continue
        ax.axvspan(p.lap_start, max(p.lap_end, p.lap_start + 0.3),
                  color=PHASE.get(p.label, MUTED), alpha=0.35, lw=0)
    ax.plot(spread.index, spread.to_numpy(), color=FG, lw=1.2)
    for lbl in ("gelb", "safety car"):
        ax.plot([], [], color=PHASE[lbl], lw=8, alpha=0.35, label=lbl)
    ax.legend(loc="upper right", frameon=False, labelcolor=FG, fontsize=9)
    ax.set_xlabel("Runde")
    ax.set_ylabel("Feldstreckung P1-Letzter [s]")
    ax.set_title(f"{EVENT} {SEASON} - Track-Status und Feldstreckung",
                loc="left", color=FG, fontsize=12, pad=10)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(alpha=0.3, linewidth=0.8, color=GRID)
    ax.set_axisbelow(True)


def zeichne_kompaktierung(ax, komp: pd.DataFrame) -> None:
    """VORGEHEN 4."""
    x = np.arange(len(komp))
    w = 0.35
    ax.bar(x - w / 2, komp["baseline_s"], width=w, color=MUTED,
          label="Baseline (3 gruene Runden davor)")
    ax.bar(x + w / 2, komp["minimum_s"], width=w, color=SERIEN[1],
          label="Minimum waehrend SC")
    ax.set_xticks(x, [f"Runde {int(r.start)}-{int(r.ende)}"
                      for r in komp.itertuples()])
    for i, r in enumerate(komp.itertuples()):
        ax.text(i, max(r.baseline_s, r.minimum_s) + 5,
               f"-{r.kompaktierung_pct:.0f}%", ha="center", color=FG, fontsize=9)
    ax.set_ylabel("Feldstreckung [s]")
    ax.set_title("Kompaktierung: Minimum gegen Baseline", loc="left",
                color=FG, fontsize=12, pad=10)
    ax.legend(loc="upper right", frameon=False, labelcolor=FG, fontsize=9)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="y", alpha=0.3, linewidth=0.8, color=GRID)
    ax.set_axisbelow(True)


def zeichne_ausbaustufe(ax, scan: pd.DataFrame) -> None:
    """AUSBAUSTUFE."""
    g = scan.groupby("klasse")["delta_pos"].agg(["mean", "count"])
    g = g.reindex(["kurz davor", "waehrend SC/VSC"])
    farben = [SERIEN[1], SERIEN[0]]
    ax.bar(g.index, g["mean"], color=farben, width=0.5)
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.set_ylim(g["mean"].min() * 1.2, 0.22)
    for i, (_idx, row) in enumerate(g.iterrows()):
        ax.text(i, 0.03, f"n={int(row['count'])}", ha="center",
               va="bottom", color=MUTED, fontsize=9)
    ax.set_ylabel("mittlere Positionsaenderung\n(Runde vor Stopp -> SC-Ende)")
    ax.set_title(f"Saison-Scan {SEASON}: Stopp-Zeitpunkt relativ zu SC/VSC",
                loc="left", color=FG, fontsize=12, pad=10)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="y", alpha=0.3, linewidth=0.8, color=GRID)
    ax.set_axisbelow(True)


def main():
    f1lab.enable_cache()

    print(f"[1/3] {EVENT} {SEASON} {IDENT} laden (VORGEHEN 1-4) ...")
    ses = f1lab.load(SEASON, EVENT, IDENT, telemetry=False)
    phasen = f1lab.track_status_phases(ses)
    neutral = phasen[phasen["label"].isin(["safety car", "vsc"])]
    print(phasen[phasen["label"] != "gruen"]
         [["label", "lap_start", "lap_end", "duration_s"]]
         .round(1).to_string(index=False))

    print("\n[2/3] Feldstreckung: Baseline gegen Kompaktierungs-Minimum ...")
    spread = f1lab.field_spread(ses)
    komp = f1lab.sc_compaction(neutral, spread)
    print(komp.round(1).to_string(index=False))
    gruene_runden = phasen.loc[phasen["label"] == "gruen", "lap_start"]
    referenz = spread.reindex(gruene_runden).dropna().median()
    print(f"      Gruenphasen-Referenz (Median ueber Runden am Beginn "
         f"jeder gruenen Phase): {referenz:.1f} s")

    print(f"\n[3/3] AUSBAUSTUFE: Saison-Scan {SEASON} ({len(SAISON_RENNEN)} "
         f"Rennen) ...")
    scan = saison_scan()
    print(f"      {scan['gp'].nunique()} Rennen mit SC/VSC, "
         f"{len(scan)} auswertbare Faelle")
    print(scan.groupby("klasse")["delta_pos"].agg(["mean", "median", "count"])
         .round(2).to_string())

    print("\nGrafik ...")
    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1], hspace=0.4, wspace=0.28)
    zeichne_chronik(fig.add_subplot(gs[0, :]), phasen, spread)
    zeichne_kompaktierung(fig.add_subplot(gs[1, 0]), komp)
    zeichne_ausbaustufe(fig.add_subplot(gs[1, 1]), scan)
    fig.suptitle(f"{ses.event['EventName']} {SEASON} - Safety-Car-Chronik",
                x=0.09, ha="left", fontsize=16, color=FG, y=0.995)
    path = OUT / "safety_car_chronik.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"\n      -> {path}")


if __name__ == "__main__":
    main()
