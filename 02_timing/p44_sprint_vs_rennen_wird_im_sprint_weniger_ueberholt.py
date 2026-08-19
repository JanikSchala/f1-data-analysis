"""
P44 - Sprint gegen Rennen: wird im Sprint wirklich weniger ueberholt?
========================================================================

Sprints gelten im Kommentatoren-Jargon als das "verrueckte kurze Rennen"
ohne Strategie, mit vollem Risiko von der ersten Runde an. Zeigt sich das
in den Ueberholzahlen, oder ist ein Sprint tatsaechlich das ereignisaermere
der beiden Formate am selben Wochenende?

Kategorie:   Timing & Rundenanalyse
Niveau:      Fortgeschritten
Aufwand:     3-4 h
Schwerpunkt: Datenanalyse, Statistik

WARUM DAS LOHNT
Ein Sprint-Wochenende liefert einen eingebauten, paarweisen Vergleich, den
sonst kein Rennformat hat: dieselbe Strecke, dasselbe Wochenende, zwei
Renndistanzen auf demselben Asphalt. f1lab.overtakes_matrix() (P20) zaehlt
in beiden Sessions identisch (nur gruene Flagge, ohne Boxenstopp-Effekt) -
keine neue f1lab-Logik noetig, nur eine neue Gegenueberstellung eines
bereits vorhandenen Bausteins.

VORGEHEN
  1. Alle Sprint-Wochenenden aus dem Schedule finden (`EventFormat`
     enthaelt "sprint") - 2023 (neues Sprint-Shootout-Format) und 2024
  2. Ueberholungen je Runde in Sprint (S) und Rennen (R) desselben
     Wochenendes zaehlen, f1lab.overtakes_matrix()
  3. Nasse Sessions ausschliessen (INTERMEDIATE/WET-Reifen in Sprint ODER
     Rennen) - wie P43, ein trockenes/nasses Ungleichgewicht zwischen den
     beiden Sessions eines Wochenendes waere kein Formatunterschied mehr
  4. Gepaarter Vergleich (Wilcoxon, wie P42/P43) statt gepoolter Mittelwerte
     - haelt Strecke und Saison konstant, jedes Wochenende ist sein eigener
     Kontrollfall
  5. Je-Wochenende-Konsistenz: gilt der Unterschied bei JEDEM trockenen
     Sprint-Wochenende, oder nur im Schnitt?

GENUTZTE FASTF1-BAUSTEINE
  - session.laps (Positions-/Boxenstopp-/Flaggendaten), keine Telemetrie
  - EventSchedule["EventFormat"]
  - Laps.Compound (Regen-Erkennung, wie P43)
  - scipy.stats.wilcoxon

Ein erster Blick auf 2023 allein war irrefuehrend: Oesterreich und Belgien
2023 zeigten hoehere Ueberholraten im SPRINT als im Rennen - das
Gegenteil der spaeter bestaetigten Regel. Vor dem Verwerfen genauer
hingeschaut (wie beim P39-Bremszonen-Fund): beide Sprints liefen auf
INTERMEDIATE/WET, die zugehoerigen Rennen trocken - derselbe
Regen-Fallstrick wie in P43, hier zwischen den beiden Sessions eines
Wochenendes statt zwischen Rennen einer Saison. Ausgeschlossen statt die
Grundfrage zu verwerfen.

AUSBAUSTUFE  [umgesetzt]
Wo im Rennverlauf passieren die Ueberholungen - fruehe Startrunden-Action
oder spaetes Reifenabbau-Ueberholen? f1lab.overtake_events() (P39) liefert
die Rundennummer je Vorgang, als Anteil der Renndistanz vergleichbar
zwischen unterschiedlich langen Sessions.

Saison 2023+2024, **9 von 9 trockenen Sprint-Wochenenden** (Oesterreich
und Belgien 2023 sowie Sao Paulo 2024 wegen Regen in mindestens einer der
beiden Sessions ausgeschlossen): das Rennen ueberholt in JEDEM einzelnen
Fall haeufiger pro Runde als der Sprint desselben Wochenendes (Median
2.04 gegen 0.79 Ueberholungen/Runde, Faktor ~2.6x, Wilcoxon p=0.0039). Die
Spannweite reicht von knapp (Aserbaidschan 2023, 1.08x) bis extrem
(Oesterreich 2024, 6.96x) - aber nie in die andere Richtung.

Die AUSBAUSTUFE zeigt eine plausible Teilerklaerung: im Rennen passieren
41.9% aller Ueberholungen im ersten Renndrittel (Startrunden-Chaos mit vollem
Feld), im Sprint nur 20.6% - dafuer verschiebt sich das Sprint-Uberholen
deutlich staerker ins letzte Drittel (39.7% gegen 18.8% im Rennen). Eine
plausible Lesart: das Rennen hat sowohl den vollen Start-Chaos-Effekt
(grosses Feld, enge erste Kurven) als auch reifenabbaubedingtes
Spaet-Ueberholen, waehrend der kurze Sprint (meist ohne Pflichtstopp, ein
Stint) den Start-Effekt zwar auch hat, aber kaum Zeit fuer eine zweite
Ueberholwelle durch Reifenabbau uebrig laesst, bevor die Flagge faellt -
das erklaert die verschobene relative Verteilung, aber nicht vollstaendig
den absoluten Faktor 2.6x, der ueber beide Drittel hinweg besteht.
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
from scipy.stats import wilcoxon

import f1lab
from f1lab.design import FG, GRID, MUTED, SERIEN, matplotlib_stil

warnings.filterwarnings("ignore")

OUT = Path(__file__).parent / "out"
OUT.mkdir(exist_ok=True)

SAISONS = (2023, 2024)

plt.rcParams.update(matplotlib_stil())


def ist_nass(session) -> bool:
    """VORGEHEN 3: Regen ist ein eigener, viel groesserer Zeit-/Chaos-Effekt
    als das Rennformat selbst (wie P43)."""
    return session.laps["Compound"].isin(["INTERMEDIATE", "WET"]).any()


def sprint_namen_im_schedule(saison: int) -> list[str]:
    """VORGEHEN 1: Sprint-Wochenenden ueber EventFormat finden - das steht
    nur im rohen fastf1-Schedule, nicht in f1lab.event_dimension()."""
    sched_roh = fastf1.get_event_schedule(saison, include_testing=False)
    return sched_roh[sched_roh["EventFormat"]
                     .str.contains("sprint", case=False,
                                  na=False)]["EventName"].tolist()


def sprint_wochenenden(saisons) -> pd.DataFrame:
    """VORGEHEN 1-3: alle trockenen Sprint-Wochenenden ueber mehrere
    Saisons sammeln."""
    zeilen, nass = [], []
    for saison in saisons:
        for gp in sprint_namen_im_schedule(saison):
            try:
                ses_s = f1lab.load(saison, gp, "S", telemetry=False)
                ses_r = f1lab.load(saison, gp, "R", telemetry=False)
            except Exception:
                continue
            if ist_nass(ses_s) or ist_nass(ses_r):
                nass.append(f"{saison} {gp}")
                continue
            n_s = int(f1lab.overtakes_matrix(ses_s).values.sum())
            n_r = int(f1lab.overtakes_matrix(ses_r).values.sum())
            zeilen.append({
                "saison": saison, "gp": gp,
                "rate_sprint": n_s / ses_s.total_laps,
                "rate_rennen": n_r / ses_r.total_laps,
                "n_sprint": n_s, "n_rennen": n_r,
                "runden_sprint": ses_s.total_laps,
                "runden_rennen": ses_r.total_laps})
    return pd.DataFrame(zeilen), nass


def ueberholzeitpunkte(saisons) -> tuple[pd.Series, pd.Series]:
    """AUSBAUSTUFE: Anteil der Renndistanz, zu dem Ueberholungen passieren,
    gepoolt ueber alle trockenen Sprint-Wochenenden."""
    anteile_s, anteile_r = [], []
    for saison in saisons:
        for gp in sprint_namen_im_schedule(saison):
            try:
                ses_s = f1lab.load(saison, gp, "S", telemetry=False)
                ses_r = f1lab.load(saison, gp, "R", telemetry=False)
            except Exception:
                continue
            if ist_nass(ses_s) or ist_nass(ses_r):
                continue
            ev_s = f1lab.overtake_events(ses_s)
            ev_r = f1lab.overtake_events(ses_r)
            if not ev_s.empty:
                anteile_s.append(ev_s["lap"] / ses_s.total_laps)
            if not ev_r.empty:
                anteile_r.append(ev_r["lap"] / ses_r.total_laps)
    return (pd.concat(anteile_s, ignore_index=True),
            pd.concat(anteile_r, ignore_index=True))


def zeichne_vergleich(ax, df: pd.DataFrame) -> None:
    """VORGEHEN 4-5: gepaarter Vergleich je Wochenende, sortiert nach Faktor."""
    df = df.copy()
    df["faktor"] = df["rate_rennen"] / df["rate_sprint"]
    df = df.sort_values("faktor")
    labels = [f"{row.gp.replace(' Grand Prix', '')} {row.saison}"
             for row in df.itertuples()]
    y = np.arange(len(df))
    ax.barh(y - 0.19, df["rate_sprint"], height=0.36, color=SERIEN[0],
           label="Sprint")
    ax.barh(y + 0.19, df["rate_rennen"], height=0.36, color=SERIEN[1],
           label="Rennen")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    x_max = df[["rate_sprint", "rate_rennen"]].max(axis=1).max()
    for yi, faktor in zip(y, df["faktor"]):
        ax.text(x_max * 1.08, yi, f"{faktor:.2f}x", va="center", fontsize=8,
               color=MUTED)
    ax.set_xlim(0, x_max * 1.22)
    ax.set_xlabel("Ueberholungen pro Runde (nur gruene Flagge)")
    ax.set_title("Sprint gegen Rennen, je trockenes Sprint-Wochenende "
                "2023+2024", loc="left", color=FG, fontsize=13, pad=10)
    ax.legend(loc="lower right", bbox_to_anchor=(1.0, 1.01), ncol=2,
             frameon=False, labelcolor=FG, fontsize=9)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="x", alpha=0.3, linewidth=0.8, color=GRID)
    ax.set_axisbelow(True)


def zeichne_zeitpunkte(ax, anteile_s: pd.Series, anteile_r: pd.Series) -> None:
    """AUSBAUSTUFE: wann im Rennverlauf passieren die Ueberholungen?"""
    bins = np.linspace(0, 1, 11)
    ax.hist(anteile_s, bins=bins, density=True, alpha=0.6, color=SERIEN[0],
           label=f"Sprint (n={len(anteile_s)})")
    ax.hist(anteile_r, bins=bins, density=True, alpha=0.6, color=SERIEN[1],
           label=f"Rennen (n={len(anteile_r)})")
    ax.set_xlabel("Renndistanz, zu der die Ueberholung passiert")
    ax.set_ylabel("Dichte")
    ax.set_title("AUSBAUSTUFE: wann im Rennverlauf wird ueberholt?",
                loc="left", color=FG, fontsize=13, pad=10)
    ax.legend(loc="upper center", frameon=False, labelcolor=FG, fontsize=9)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="y", alpha=0.3, linewidth=0.8, color=GRID)
    ax.set_axisbelow(True)


def main():
    f1lab.enable_cache()

    print(f"[1-3/5] Sprint-Wochenenden {SAISONS}, nasse raus "
         "(VORGEHEN 1-3) ...")
    df, nass = sprint_wochenenden(SAISONS)
    print(f"      {len(df)} trockene Sprint-Wochenenden, "
         f"{len(nass)} wegen Regen ausgeschlossen: {nass}")
    for row in df.itertuples():
        faktor = row.rate_rennen / row.rate_sprint
        print(f"      {row.saison} {row.gp}: Sprint {row.rate_sprint:.2f}/"
             f"Runde, Rennen {row.rate_rennen:.2f}/Runde ({faktor:.2f}x)")

    print("\n[4/5] Gepaarter Test (VORGEHEN 4) ...")
    diff = df["rate_rennen"] - df["rate_sprint"]
    test = wilcoxon(diff)
    print(f"      Median Sprint: {df['rate_sprint'].median():.2f}/Runde")
    print(f"      Median Rennen: {df['rate_rennen'].median():.2f}/Runde")
    print(f"      Wilcoxon p={test.pvalue:.4f}")

    print("\n[5/5] Je-Wochenende-Konsistenz (VORGEHEN 5) ...")
    hoeher = (df["rate_rennen"] > df["rate_sprint"]).sum()
    print(f"      Rennen > Sprint in {hoeher}/{len(df)} Wochenenden")

    print("\nAUSBAUSTUFE: wann im Rennverlauf wird ueberholt? ...")
    anteile_s, anteile_r = ueberholzeitpunkte(SAISONS)
    for name, a in (("Sprint", anteile_s), ("Rennen", anteile_r)):
        print(f"      {name}: erstes Drittel={( a < 1 / 3).mean():.1%}, "
             f"mittleres={((a >= 1 / 3) & (a < 2 / 3)).mean():.1%}, "
             f"letztes={(a >= 2 / 3).mean():.1%}")

    print("\nGrafiken speichern ...")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 11),
                                   gridspec_kw={"height_ratios": [1.3, 1]})
    fig.patch.set_facecolor(plt.rcParams["figure.facecolor"])
    zeichne_vergleich(ax1, df)
    zeichne_zeitpunkte(ax2, anteile_s, anteile_r)
    fig.tight_layout()
    fig.savefig(OUT / "p44_sprint_vs_rennen.png", dpi=140)
    plt.close(fig)
    print(f"      {OUT / 'p44_sprint_vs_rennen.png'}")


if __name__ == "__main__":
    main()
