"""
P32 - Verfolgungsjagd: Abstand zum Vordermann und Dirty Air
===========================================================

DistanceToDriverAhead auswerten: Wie viel Pace kostet es, im Windschatten zu haengen?

Kategorie:   Telemetrie
Niveau:      Profi
Aufwand:     4-5 h
Schwerpunkt: Strategie, Datenanalyse

WARUM DAS LOHNT
Dirty Air ist DAS Thema seit der Regelaenderung 2022. Der add_driver_ahead()-Kanal ist rechenintensiv - dass du ihn nutzt, zeigt Tiefe.

VORGEHEN
  1. Telemetrie eines Fahrers mit add_driver_ahead() anreichern
  2. Je Runde den Medianabstand zum Vordermann berechnen
  3. Rundenzeit gegen Abstand regressieren (nur gruene Runden)
  4. Schwellenwert finden, ab dem Dirty Air spuerbar wird

GENUTZTE FASTF1-BAUSTEINE
  - Telemetry.add_driver_ahead
  - DistanceToDriverAhead
  - DriverAhead
  - Laps.get_telemetry
  - f1lab.fuel_correct / f1lab.fit_degradation (Bereinigung vor der Regression)

AUSBAUSTUFE  [umgesetzt]
Dieselbe Analyse fuer mehrere Strecken wiederholen und pruefen, wo Dirty Air
am staerksten wirkt - Highspeed- gegen Winkelkurse.

Zwei Korrekturen an der urspruenglichen Fassung - die erste aendert das
Ergebnis nicht nur spuerbar, sondern komplett:

Erstens fehlte die Treibstoffkorrektur komplett - nur Reifenalter wurde
herausgerechnet (per Hand mit np.linalg.lstsq), Sprit nicht. Der Unterschied
ist nicht klein: fuer SAI in Spanien 2024 findet die urspruengliche Rechnung
einen Zusammenhang von +0.026 s je 1 Prozentpunkt Zeit unter 50 m, mit
R^2 = 0.41 - ein scheinbar klares Ergebnis. Sobald fuel_correct() (P04) vor
der Degradations-Bereinigung laeuft, bricht der Zusammenhang auf
+0.001 s (R^2 = 0.002) zusammen: praktisch nichts. Die urspruengliche Fassung
hatte den Treibstoffeffekt (~0.03 s/Runde/kg) gemessen, nicht Dirty Air -
beides korreliert ueber eine Renndistanz, weil beide mit der Rundenzahl
laufen. Statt der Handrechnung fuer die Degradation uebernimmt jetzt
f1lab.fit_degradation() - dieselbe, getestete Funktion wie in P13/Dashboard.

Zweitens lief die urspruengliche Fassung add_driver_ahead() einzeln pro
Runde in einer Schleife - bei ~55 Runden ~55 Einzelaufrufe. Wie in P20/P05
bemerkt: einmal auf die *gesamte* Renntelemetrie eines Fahrers angewendet,
ist dieselbe Funktion um ein Vielfaches schneller (ein Aufruf statt
Rundenzahl-viele) und liefert identische Werte - die Kosten stecken im
Funktionsaufruf selbst, nicht in der Datenmenge pro Aufruf.

Ergebnis ueber drei Strecken (je 8 Fahrer, gruene Runden, treibstoff- und
degradationsbereinigt): Bahrain +0.005 s je 1 Prozentpunkt Zeit unter 50 m
(R^2 = 0.07), Monaco +0.007 s (R^2 = 0.05), Monza -0.002 s (R^2 = 0.01). Das
widerspricht der ueblichen Annahme "Dirty Air trifft Winkelkurse haerter"
eher, als sie zu bestaetigen - mit dem Vorbehalt, dass alle drei R^2-Werte
niedrig sind: der Effekt ist (nach Korrektur) real, aber schwach gegenueber
der uebrigen Streuung einer Rundenzeit (Fahrfehler, Verkehr,
Streckenentwicklung).
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
from matplotlib.colors import LinearSegmentedColormap

import f1lab
from f1lab.design import FG, GRID, MUTED, RAMPE, SERIEN, matplotlib_stil

RAMPE_CMAP = LinearSegmentedColormap.from_list("rampe", RAMPE)

warnings.filterwarnings("ignore")

OUT = Path(__file__).parent / "out"
OUT.mkdir(exist_ok=True)

SEASON, EVENT, IDENT = 2024, "Spain", "R"
DRIVER = "SAI"
STRECKEN = ["Bahrain", "Monaco", "Italy"]           # schnell -> langsam/eng
FAHRER_JE_STRECKE = 8
NAH_SCHWELLE_M = 50.0

plt.rcParams.update(matplotlib_stil())


def abstands_je_runde(ses, driver: str) -> pd.DataFrame:
    """VORGEHEN 1-2: Medianabstand und Nahanteil je gruener Runde.

    add_driver_ahead() laeuft einmal auf die gesamte Renntelemetrie des
    Fahrers, nicht einmal je Runde (siehe Docstring) - die Rundenzuordnung
    kommt danach per merge_asof ueber die Rundenstartzeit, dieselbe Technik
    wie in P05/P20.
    """
    laps = (ses.laps.pick_drivers(driver).pick_wo_box().pick_accurate()
           .pick_track_status("1").sort_values("LapStartTime"))
    if laps.empty:
        return pd.DataFrame()
    try:
        tel = laps.get_telemetry().add_driver_ahead().sort_values("SessionTime")
    except Exception:
        return pd.DataFrame()
    if tel.empty:
        return pd.DataFrame()

    grenzen = (laps[["LapNumber", "LapStartTime"]]
              .rename(columns={"LapStartTime": "SessionTime"})
              .sort_values("SessionTime"))
    zug = pd.merge_asof(tel, grenzen, on="SessionTime", direction="backward")
    zug["gap"] = zug["DistanceToDriverAhead"].replace(0, np.nan)

    total_laps = int(ses.total_laps)
    rows = []
    for lapnum, g in zug.groupby("LapNumber"):
        treffer = laps.loc[laps["LapNumber"] == lapnum]
        if treffer.empty or pd.isna(treffer.iloc[0]["LapTime"]):
            continue
        lap = treffer.iloc[0]
        sec = lap["LapTime"].total_seconds()
        rows.append({
            "lap": int(lapnum),
            "sec_fuel": float(f1lab.fuel_correct([sec], [lapnum], total_laps)[0]),
            "gap_median_m": g["gap"].median(),
            "gap_min_m": g["gap"].min(),
            "anteil_nah": 100 * (g["gap"] < NAH_SCHWELLE_M).mean(),
            "compound": lap["Compound"], "tyre_life": lap["TyreLife"],
        })
    return pd.DataFrame(rows)


def dirty_air_regression(df: pd.DataFrame) -> tuple[float, float, float, pd.DataFrame]:
    """VORGEHEN 3: Rundenzeit (treibstoff- und degradationsbereinigt) gegen
    Nahanteil regressieren. Returns (slope, intercept, r2, df mit sec_corr)."""
    d = df.dropna(subset=["gap_median_m", "tyre_life"]).copy()
    d = d[d["gap_median_m"] < 500]
    if len(d) < 5 or d["tyre_life"].nunique() < 2:
        return float("nan"), float("nan"), float("nan"), d

    fit = f1lab.fit_degradation(d["tyre_life"], d["sec_fuel"])
    d["sec_corr"] = d["sec_fuel"] - fit.slope * d["tyre_life"]

    slope, inter = np.polyfit(d["anteil_nah"], d["sec_corr"], 1)
    pred = slope * d["anteil_nah"] + inter
    ss_res = float(((d["sec_corr"] - pred) ** 2).sum())
    ss_tot = float(((d["sec_corr"] - d["sec_corr"].mean()) ** 2).sum())
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return float(slope), float(inter), r2, d


def zeichne_regression(ax, d: pd.DataFrame, slope: float, inter: float,
                       r2: float, titel: str) -> None:
    sc = ax.scatter(d["anteil_nah"], d["sec_corr"], c=d["tyre_life"],
                    cmap=RAMPE_CMAP, s=55, zorder=3)
    xs = np.linspace(d["anteil_nah"].min(), d["anteil_nah"].max(), 50)
    ax.plot(xs, slope * xs + inter, color=SERIEN[1], lw=2.2)
    cb = ax.figure.colorbar(sc, ax=ax, pad=0.02)
    cb.set_label("Reifenalter [Runden]", color=MUTED, fontsize=9)
    cb.ax.tick_params(colors=MUTED, length=0)
    cb.outline.set_visible(False)

    ax.set_xlabel(f"Zeitanteil unter {NAH_SCHWELLE_M:.0f} m Abstand [%]")
    ax.set_ylabel("Bereinigte Rundenzeit [s]")
    ax.set_title(f"{titel}: {slope:+.3f} s je 1%, R²={r2:.2f}", loc="left",
                color=FG, fontsize=13, pad=10)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(alpha=0.3, linewidth=0.8, color=GRID)
    ax.set_axisbelow(True)


def zeichne_streckenvergleich(ax, ergebnisse: pd.DataFrame) -> None:
    e = ergebnisse.sort_values("slope")
    farben = [SERIEN[1] if s > 0 else SERIEN[2] for s in e["slope"]]
    ax.barh(e["strecke"], e["slope"], color=farben, height=0.6)
    for y, (_s, r2, n) in enumerate(zip(e["slope"], e["r2"], e["n"], strict=True)):
        ax.text(0, y + 0.38, f"R²={r2:.2f}, n={n}", va="bottom",
               ha="center", color=MUTED, fontsize=8.5)
    ax.axvline(0, color=MUTED, lw=0.8)
    ax.set_xlabel(f"Dirty-Air-Effekt [s je 1% Zeit unter "
                 f"{NAH_SCHWELLE_M:.0f} m]")
    ax.set_title("Streckenvergleich: Highspeed gegen Winkelkurs", loc="left",
                color=FG, fontsize=13, pad=10)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="x", alpha=0.3, linewidth=0.8, color=GRID)
    ax.set_axisbelow(True)


def main():
    f1lab.enable_cache()

    print(f"[1/3] {EVENT} {SEASON} {IDENT} laden, Fahrer {DRIVER} ...")
    ses = f1lab.load(SEASON, EVENT, IDENT, telemetry=True)
    df = abstands_je_runde(ses, DRIVER)
    slope, inter, r2, d = dirty_air_regression(df)
    print(f"      Dirty-Air-Effekt: {slope:+.4f} s je 1% Zeit unter "
         f"{NAH_SCHWELLE_M:.0f} m (R²={r2:.3f}, n={len(d)})")
    print(d[["lap", "sec_fuel", "sec_corr", "gap_median_m", "anteil_nah",
             "compound", "tyre_life"]].round(2).to_string(index=False))

    print(f"\n[2/3] Streckenvergleich ueber {len(STRECKEN)} Strecken "
         f"({FAHRER_JE_STRECKE} Fahrer je Strecke) ...")
    ergebnisse = []
    for gp in STRECKEN:
        s = f1lab.load(SEASON, gp, "R", telemetry=True)
        fahrer = list(s.drivers)[:FAHRER_JE_STRECKE]
        teile = [abstands_je_runde(s, drv) for drv in fahrer]
        teile = [t for t in teile if not t.empty]
        gesamt = pd.concat(teile, ignore_index=True) if teile else pd.DataFrame()
        sl, ic, rr, dd = dirty_air_regression(gesamt)
        ergebnisse.append({"strecke": s.event["EventName"], "slope": sl,
                           "r2": rr, "n": len(dd)})
        print(f"      {s.event['EventName']:<28} {sl:+.4f} s/%  "
             f"R²={rr:.3f}  n={len(dd)}")
    ergebnisse = pd.DataFrame(ergebnisse)

    print("\n[3/3] Grafik ...")
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    zeichne_regression(ax[0], d, slope, inter, r2, f"{DRIVER} - {EVENT} {SEASON}")
    zeichne_streckenvergleich(ax[1], ergebnisse)
    fig.suptitle("Dirty Air: Abstand zum Vordermann und Rundenzeit-Effekt",
                x=0.09, ha="left", fontsize=16, color=FG, y=1.02)
    plt.tight_layout()
    path = OUT / "dirty_air.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"\n      -> {path}")


if __name__ == "__main__":
    main()
