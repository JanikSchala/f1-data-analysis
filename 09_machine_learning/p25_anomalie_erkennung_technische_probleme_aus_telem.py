"""
P25 - Anomalie-Erkennung: Technische Probleme aus Telemetrie erkennen
=====================================================================

Isolation Forest auf Rundenprofilen, um Runden zu markieren, in denen etwas nicht stimmte - noch bevor der Fahrer funkt.

Kategorie:   Machine Learning
Niveau:      Profi
Aufwand:     5 h
Schwerpunkt: Datenanalyse, Engineering
Zusaetzliche Pakete: scikit-learn

WARUM DAS LOHNT
Predictive Maintenance in F1-Kontext. Anomalieerkennung ohne Labels ist genau das Problem, das Teams bei Zuverlaessigkeit tatsaechlich haben.

VORGEHEN
  1. Fuer jede Runde eines Fahrers ein Feature-Profil bauen
  2. Isolation Forest auf allen Runden trainieren
  3. Auffaellige Runden markieren und mit Race-Control-Messages abgleichen
  4. Anomalie-Score ueber den Rennverlauf plotten

GENUTZTE FASTF1-BAUSTEINE
  - Lap.get_car_data
  - Telemetry RPM/Speed/Throttle
  - sklearn IsolationForest

AUSBAUSTUFE  [umgesetzt]
Vergleiche die Flags mit den Ausfallgruenden aus Session.results['Status'] -
wie frueh haette das Modell gewarnt?

Baku 2024 R hat vier Ausfaelle (PER, SAI, STR, TSU) - genug, um die
AUSBAUSTUFE tatsaechlich zu pruefen statt nur zu behaupten. Dabei zeigt sich
sofort, warum `pick_wo_box().pick_accurate()` aus der Vorlage (VORGEHEN 1)
die interessanteste Runde regelmaessig wegfiltert: Strolls letzte Runde vor
dem Ausfall (Runde 45, 2:02 statt der ueblichen 1:48-1:49) traegt
IsAccurate=False UND eine gesetzte PitInTime - beides Kriterien, unter denen
die Vorlage sie ausgeschlossen haette. Ohne Boxenrunden zu filtern, faengt
das Modell diese Runde jedoch sofort (Score 0.66, Rang 13 von 922 sauberen
Runden).

Zweiter Fund beim Entfernen des Filters: viele der hoechsten Scores waren
gar keine Einzelfahrzeug-Probleme, sondern die VSC nach der PER/SAI-Kollision
in Runde 51 - jede einzige Runde 50 im Feld wird dadurch fuer JEDEN Fahrer
"anomal", nicht weil am Auto etwas kaputt war, sondern weil das ganze Feld
gleichzeitig abbremste. Eine feldweite Neutralisation sieht in einer
pro-Fahrer-z-Normalisierung genauso aus wie ein individueller Defekt - beides
sind grosse Abweichungen vom eigenen Normalverhalten. Deshalb werden
Runden unter Gelb/VSC/SC (via f1lab.track_status_phases(), siehe P18) jetzt
explizit ausgeschlossen, bevor das Modell trainiert wird - sonst waeren die
Top-Flags fast ausschliesslich Rennleitungs-Artefakte, keine Telemetrie-
Befunde.

Ehrliches Ergebnis der AUSBAUSTUFE: fuer PER/SAI (Kollision) gibt es
erwartungsgemaess keine Vorwarnung - eine Kollision kuendigt sich nicht in
der eigenen Telemetrie an. Fuer STR und TSU, die tatsaechlichen technischen
Probleme, liefert das Modell auf sauberen (nicht boxen-, nicht
neutralisationsbetroffenen) Runden VOR dem finalen Stopp praktisch keine
Vorwarnung - die auffaelligen Werte erscheinen erst in genau der Runde, in
der ohnehin schon in die Box gefahren wird. Das Modell bestaetigt also eher,
was Team und Fahrer in dem Moment bereits wissen, als dass es fruehzeitig
warnt. Fuer eine echte Fruehwarnung braeuchte es Signale INNERHALB einer
Runde (z. B. einen drifenden Sensorwert mitten auf der Geraden), nicht
Rundenaggregate.
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
from sklearn.ensemble import IsolationForest

import f1lab
from f1lab.design import FG, GRID, MUTED, PHASE, SERIEN, matplotlib_stil

warnings.filterwarnings("ignore")

OUT = Path(__file__).parent / "out"
OUT.mkdir(exist_ok=True)

SEASON, EVENT, IDENT = 2024, "Baku", "R"
CONTAMINATION = 0.04
FEAT = ["lap_time", "vmax", "vmean", "rpm_max", "rpm_mean", "throttle_mean",
       "brake_pct", "gear_mean"]

plt.rcParams.update(matplotlib_stil())


def rundenprofile(ses) -> pd.DataFrame:
    """VORGEHEN 1: Feature-Profil je Runde, ALLE Runden (nicht
    pick_wo_box().pick_accurate() wie in der Vorlage - siehe Docstring,
    das filtert die interessantesten Faelle regelmaessig weg). Box- und
    Neutralisations-Zugehoerigkeit bleiben als Metadaten erhalten, um sie
    gezielt statt blind auszuschliessen."""
    phasen = f1lab.track_status_phases(ses)
    neutral_laps = set()
    for p in phasen[phasen["label"] != "gruen"].itertuples():
        neutral_laps.update(range(p.lap_start, p.lap_end + 1))

    rows = []
    for lap in ses.laps.itertuples():
        if pd.isna(lap.LapTime):
            continue
        try:
            tel = ses.laps.loc[[lap.Index]].get_car_data()
        except Exception:
            continue
        if tel is None or len(tel) < 50:
            continue
        rows.append({
            "driver": lap.Driver, "lap": lap.LapNumber,
            "lap_time": lap.LapTime.total_seconds(),
            "vmax": tel["Speed"].max(), "vmean": tel["Speed"].mean(),
            "rpm_max": tel["RPM"].max(), "rpm_mean": tel["RPM"].mean(),
            "throttle_mean": tel["Throttle"].mean(),
            "brake_pct": 100 * tel["Brake"].astype(bool).mean(),
            "gear_mean": tel["nGear"].mean(),
            "boxenrunde": pd.notna(lap.PitInTime) or pd.notna(lap.PitOutTime),
            "neutralisiert": lap.LapNumber in neutral_laps,
            "erste_runde": lap.LapNumber == 1,
        })
    return pd.DataFrame(rows).dropna()


def anomalien_finden(df: pd.DataFrame) -> pd.DataFrame:
    """VORGEHEN 2: IsolationForest auf sauberen Runden (kein Feld-Ereignis,
    keine Startrunde - beides strukturell anders, kein Defekt). Boxenrunden
    bleiben drin, das ist genau der Fall aus dem Docstring."""
    sauber = df[~df["neutralisiert"] & ~df["erste_runde"]].copy()
    z = sauber.groupby("driver")[FEAT].transform(
        lambda s: (s - s.mean()) / (s.std() + 1e-9))
    iso = IsolationForest(contamination=CONTAMINATION, random_state=0)
    sauber["anomalie"] = iso.fit_predict(z)
    sauber["score"] = -iso.score_samples(z)
    return sauber


def mit_race_control_abgleichen(flags: pd.DataFrame, rcm: pd.DataFrame,
                                fenster: int = 2) -> pd.DataFrame:
    """VORGEHEN 3: passende Race-Control-Meldung je Flag suchen (Fahrer-
    kuerzel im Text, Rundennummer +/- Fenster)."""
    treffer = []
    for r in flags.itertuples():
        nahe = rcm[(rcm["Lap"] >= r.lap - fenster) & (rcm["Lap"] <= r.lap + fenster)]
        passt = nahe[nahe["Message"].str.contains(str(r.driver), na=False)]
        treffer.append(passt.iloc[0]["Message"] if len(passt) else "")
    out = flags.copy()
    out["race_control"] = treffer
    return out


def ausfall_analyse(df: pd.DataFrame, anomalien: pd.DataFrame,
                    ses) -> pd.DataFrame:
    """AUSBAUSTUFE: je Ausfall die letzte(n) Runde(n) und ob vorher eine
    Anomalie geflaggt wurde - getrennt nach Boxenrunden (die sind fast immer
    statistisch auffaellig, weil Boxengasse, unabhaengig vom Auto-Zustand)
    und echten Runden auf der Strecke. Nur Letztere zaehlen als echte
    Fruehwarnung; sonst wuerde jeder planmaessige Reifenwechsel frueh im
    Rennen faelschlich als "Warnung" durchgehen."""
    ausfaelle = ses.results[ses.results["Status"].isin(["Retired"])]
    zeilen = []
    for r in ausfaelle.itertuples():
        drv = r.Abbreviation
        letzte_runde = df.loc[df["driver"] == drv, "lap"].max()
        eigene = anomalien[anomalien["driver"] == drv].sort_values("lap")
        vor_ausfall = eigene[(eigene["anomalie"] == -1)
                             & (eigene["lap"] < letzte_runde)]
        auf_strecke = vor_ausfall[~vor_ausfall["boxenrunde"]]
        vorlauf = (letzte_runde - auf_strecke["lap"].max()
                  if not auf_strecke.empty else np.nan)
        zeilen.append({
            "driver": drv, "status": r.Status, "letzte_runde": letzte_runde,
            "anomalien_boxenrunden": len(vor_ausfall) - len(auf_strecke),
            "anomalien_auf_strecke": len(auf_strecke),
            "runden_vorlauf_auf_strecke": vorlauf,
        })
    return pd.DataFrame(zeilen)


def zeichne_verlauf(ax, df: pd.DataFrame, anomalien: pd.DataFrame,
                    phasen: pd.DataFrame) -> None:
    """VORGEHEN 4."""
    # lap_start==0 sind Meldungen vor dem eigentlichen Renn-Beginn
    # (Formationsrunde etc.) - fuer den Rennverlauf ohne Aussagekraft.
    for p in phasen[(phasen["label"] != "gruen") & (phasen["lap_start"] > 0)].itertuples():
        ax.axvspan(p.lap_start, max(p.lap_end, p.lap_start + 0.3),
                  color=PHASE.get(p.label, MUTED), alpha=0.3, lw=0)
    for _drv, g in anomalien.groupby("driver"):
        g = g.sort_values("lap")
        ax.plot(g["lap"], g["score"], lw=0.7, alpha=0.35, color=MUTED)
    flags = anomalien[anomalien["anomalie"] == -1]
    ax.scatter(flags["lap"], flags["score"], color=SERIEN[1], s=26, zorder=3,
              label="Anomalie (sauber)")
    for lbl in ("gelb", "vsc"):
        if lbl in phasen["label"].values:
            ax.plot([], [], color=PHASE[lbl], lw=8, alpha=0.3, label=lbl)
    ax.set_xlabel("Runde")
    ax.set_ylabel("Anomalie-Score")
    ax.set_title(f"{EVENT} {SEASON} - Anomalie-Score ueber den Rennverlauf",
                loc="left", color=FG, fontsize=12, pad=10)
    ax.legend(loc="upper left", frameon=False, labelcolor=FG, fontsize=9)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(alpha=0.3, linewidth=0.8, color=GRID)
    ax.set_axisbelow(True)


def zeichne_ausfaelle(ax, df: pd.DataFrame, ausfaelle: list[str]) -> None:
    """AUSBAUSTUFE: Rundenzeitverlauf der Ausfaller, letzte Runde markiert.
    Vier Fahrer > MAX_SERIEN=3 Hausfarben - qualitatives Colormap statt
    Wiederholung, sonst faerben zwei Fahrer identisch (siehe P24)."""
    farben = plt.get_cmap("tab10")
    for i, drv in enumerate(ausfaelle):
        g = df[df["driver"] == drv].sort_values("lap")
        farbe = farben(i) if len(ausfaelle) > len(SERIEN) else SERIEN[i]
        ax.plot(g["lap"], g["lap_time"], color=farbe, lw=1.6,
               marker=".", ms=4, label=drv)
        letzte = g["lap"].max()
        ax.axvline(letzte, color=farbe, lw=0.8, ls=":")
    ax.set_xlabel("Runde")
    ax.set_ylabel("Rundenzeit [s]")
    ax.set_title("AUSBAUSTUFE: Rundenzeit der vier Ausfaller", loc="left",
                color=FG, fontsize=12, pad=10)
    ax.legend(loc="upper left", frameon=False, labelcolor=FG, fontsize=9)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(alpha=0.3, linewidth=0.8, color=GRID)
    ax.set_axisbelow(True)


def main():
    f1lab.enable_cache()

    print(f"[1/4] {EVENT} {SEASON} {IDENT} laden (VORGEHEN 1) ...")
    ses = f1lab.load(SEASON, EVENT, IDENT, telemetry=True, messages=True)
    df = rundenprofile(ses)
    print(f"      {len(df)} Runden gesamt, "
         f"{df['neutralisiert'].sum()} unter Gelb/VSC/SC, "
         f"{df['boxenrunde'].sum()} Boxenrunden (bleiben drin)")

    print("\n[2/4] Isolation Forest auf sauberen Runden (VORGEHEN 2) ...")
    anomalien = anomalien_finden(df)
    print(f"      {len(anomalien)} sauber (ohne Neutralisation/Startrunde), "
         f"{(anomalien['anomalie'] == -1).sum()} geflaggt "
         f"({CONTAMINATION:.0%})")

    print("\n[3/4] Top-Anomalien gegen Race Control abgleichen (VORGEHEN 3) ...")
    top = anomalien[anomalien["anomalie"] == -1].sort_values(
        "score", ascending=False).head(15)
    top = mit_race_control_abgleichen(top, ses.race_control_messages)
    print(top[["driver", "lap", "lap_time", "boxenrunde", "score",
              "race_control"]].round(2).to_string(index=False))

    print("\n[4/4] AUSBAUSTUFE: Ausfallanalyse ...")
    ausfall_tab = ausfall_analyse(df, anomalien, ses)
    print(ausfall_tab.to_string(index=False))

    print("\nGrafik ...")
    phasen = f1lab.track_status_phases(ses)
    fig, ax = plt.subplots(2, 1, figsize=(13, 10))
    zeichne_verlauf(ax[0], df, anomalien, phasen)
    zeichne_ausfaelle(ax[1], df, list(ausfall_tab["driver"]))
    fig.suptitle(f"{ses.event['EventName']} {SEASON} - Anomalie-Erkennung",
                x=0.09, ha="left", fontsize=15, color=FG, y=1.0)
    plt.tight_layout()
    path = OUT / "anomalie_erkennung.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"\n      -> {path}")


if __name__ == "__main__":
    main()
