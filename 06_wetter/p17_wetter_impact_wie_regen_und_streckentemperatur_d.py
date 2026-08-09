"""
P17 - Wetter-Impact: Wie Regen und Streckentemperatur die Pace veraendern
=========================================================================

Wetterdaten im Minutentakt auf jede Runde joinen und den Effekt von TrackTemp auf die Rundenzeit quantifizieren.

Kategorie:   Wetter & Bedingungen
Niveau:      Fortgeschritten
Aufwand:     3-4 h
Schwerpunkt: Datenanalyse, Strategie

WARUM DAS LOHNT
Streckentemperatur steuert das Reifenfenster. Ein sauberer Zeit-Join zwischen zwei unterschiedlich getakteten Streams ist eine klassische Data-Engineering-Uebung.

VORGEHEN
  1. weather_data laden und den Verlauf plotten
  2. get_weather_data() je Runde zuordnen (naechster Messpunkt)
  3. Rundenzeit gegen TrackTemp regressieren, Fuel-Effekt vorher rausrechnen
  4. Trocken-/Nass-Phasen ueber Rainfall-Flag segmentieren

GENUTZTE FASTF1-BAUSTEINE
  - Session.weather_data
  - Laps.get_weather_data
  - AirTemp/TrackTemp/Humidity/Rainfall/WindSpeed

AUSBAUSTUFE  [umgesetzt]
Baue einen Klassifikator, der allein aus Rundenzeit-Streuung und Speed-Traps
erkennt, ob gerade Intermediates gefahren werden.

Zwei Sessions statt einer, aus zwei verschiedenen Gruenden:

Fuer VORGEHEN 1-3 scheitert die im Vorschlag skizzierte einfache Regression
(Rundenzeit gegen TrackTemp, ueber alle Fahrer gepoolt) an jeder trockenen
2024er Session, die probiert wurde - R² zwischen 0.002 und 0.06, obwohl die
Streckentemperatur teils ueber 10°C schwankt. Der Grund: Reifenalter und
Fahrer-Tempounterschiede dominieren die Streuung so stark, dass der
Temperatureffekt komplett darin untergeht. Erst eine Regression, die
Fahrer-Mittel (statt Rohzeit) und TyreLife als zweite erklaerende Variable
mitfuehrt, macht ihn sichtbar: Japan 2024 R (trockene Runden, TrackTemp
31.3-39.0°C) - Reifenalter allein erklaert R²=0.199, mit Streckentemperatur
zusaetzlich R²=0.413. Der TrackTemp-Koeffizient (+0.215 s/°C, t=16.6 bei
n=755) ist damit eindeutig kein Rauschen, aber er war in der einfachen
gepoolten Regression unsichtbar - eine Lektion ueber Konfundierung, nicht
nur ueber Wetter.

Fuer VORGEHEN 4 und die AUSBAUSTUFE ist Japan ungeeignet (durchgehend
trocken). Kanada 2024 R hatte echten Regen mit sechs Phasenwechseln und,
wichtiger, echten Mischungswechsel zwischen Intermediates und Slicks -
noetig, um einen Klassifikator gegen echte Labels zu pruefen statt nur zu
behaupten. Dabei faellt auf: das Rainfall-Flag ist nicht dasselbe wie
"Strecke slick-tauglich" - von 665 Runden ohne Regenmeldung liefen 644
trotzdem auf Intermediates, weil die Strecke noch zu nass zum Wechseln war.

AUSBAUSTUFE: Klassifikator auf Feld-Aggregaten je Runde (Rundenzeit-Streuung
ueber alle Fahrer, mittlere Speed-Trap-Geschwindigkeit), Zielgroesse "laeuft
das Feld mehrheitlich auf Regenreifen". Logistische Regression,
Leave-one-out-Kreuzvalidierung (56 Runden): 96.4% Trefferquote gegen 66.1%
Mehrheitsklassen-Basislinie - und die Rundenzeit-Streuung traegt praktisch
nichts bei, die mittlere Speed-Trap-Geschwindigkeit allein reicht fuer
dieselbe Genauigkeit (283 km/h im Trockenen, 273 km/h im Nassen, mit
deutlich groesserer Streuung).
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import LeaveOneOut, cross_val_predict

import f1lab
from f1lab.design import FG, GRID, MUTED, SERIEN, matplotlib_stil

warnings.filterwarnings("ignore")

OUT = Path(__file__).parent / "out"
OUT.mkdir(exist_ok=True)

TEMP_EVENT = ("Japan", 2024, "R")     # durchgehend trocken, echte Temperaturspanne
NASS_EVENT = ("Canada", 2024, "R")    # echter Regen, echter Mischungswechsel

plt.rcParams.update(matplotlib_stil())


def wetter_join(ses) -> pd.DataFrame:
    """VORGEHEN 2: je Runde der naechstgelegene Wettermesspunkt."""
    laps = (ses.laps.pick_wo_box().pick_accurate().pick_track_status("1")).copy()
    wl = laps.get_weather_data().reset_index(drop=True)
    laps = laps.reset_index(drop=True)
    merged = pd.concat([laps, wl.loc[:, ~wl.columns.isin(laps.columns)]], axis=1)
    merged["sec"] = merged["LapTime"].dt.total_seconds()
    merged["corr"] = f1lab.fuel_correct(
        merged["sec"], merged["LapNumber"], ses.total_laps)
    return merged


def temperatureffekt(merged: pd.DataFrame) -> dict:
    """VORGEHEN 3: erst die naive gepoolte Regression (wie im Vorschlag
    skizziert), dann kontrolliert um Fahrer-Niveau und Reifenalter."""
    dry = merged[~merged["Rainfall"]].dropna(
        subset=["TrackTemp", "corr", "TyreLife"]).copy()

    naiv_slope, naiv_inter = np.polyfit(dry["TrackTemp"], dry["corr"], 1)
    naiv_pred = naiv_slope * dry["TrackTemp"] + naiv_inter
    naiv_r2 = 1 - ((dry["corr"] - naiv_pred) ** 2).sum() / \
        ((dry["corr"] - dry["corr"].mean()) ** 2).sum()

    dry["rel"] = dry["corr"] - dry.groupby("Driver")["corr"].transform("median")
    dry = dry[dry["rel"].abs() < 3]
    y = dry["rel"].to_numpy()

    x_tyre = np.column_stack([dry["TyreLife"], np.ones(len(dry))])
    c_tyre, *_ = np.linalg.lstsq(x_tyre, y, rcond=None)
    r2_tyre = 1 - ((y - x_tyre @ c_tyre) ** 2).sum() / ((y - y.mean()) ** 2).sum()

    x_voll = np.column_stack([dry["TrackTemp"], dry["TyreLife"], np.ones(len(dry))])
    c_voll, *_ = np.linalg.lstsq(x_voll, y, rcond=None)
    pred_voll = x_voll @ c_voll
    r2_voll = 1 - ((y - pred_voll) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    resid = y - pred_voll
    sigma2 = (resid ** 2).sum() / (len(y) - 3)
    se = np.sqrt(np.diag(np.linalg.inv(x_voll.T @ x_voll)) * sigma2)

    # Partial-Residual-Plot: TyreLife-Anteil herausgerechnet, damit die
    # TrackTemp-Wirkung isoliert sichtbar wird (die rohe (TrackTemp, rel)-
    # Punktwolke ist durch den nicht entfernten TyreLife-Effekt zu verrauscht).
    dry["partial"] = y - c_voll[1] * dry["TyreLife"]

    return {
        "naiv_slope": naiv_slope, "naiv_r2": naiv_r2,
        "r2_tyre_only": r2_tyre, "r2_voll": r2_voll,
        "coef_temp": c_voll[0], "intercept": c_voll[2],
        "se_temp": se[0], "n": len(y), "dry": dry,
    }


def wetterphasen(ses) -> pd.DataFrame:
    """VORGEHEN 4: Phasenwechsel ueber das Rainfall-Flag."""
    w = ses.weather_data
    gruppe = (w["Rainfall"] != w["Rainfall"].shift()).cumsum()
    return (w.groupby(gruppe)
            .agg(nass=("Rainfall", "first"), start=("Time", "min"),
                end=("Time", "max"))
            .reset_index(drop=True))


def klassifikator(ses) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """AUSBAUSTUFE: Feld-Aggregate je Runde, keine Compound-Spalte als
    Feature - nur was auch ohne Boxenfunk beobachtbar waere."""
    laps = ses.laps.pick_accurate().copy()
    laps["sec"] = laps["LapTime"].dt.total_seconds()
    je_runde = laps.groupby("LapNumber").agg(
        std_sec=("sec", "std"), mean_speedFL=("SpeedFL", "mean"),
        n=("sec", "count")).dropna()
    je_runde = je_runde[je_runde["n"] >= 10]

    mehrheit = laps.groupby("LapNumber")["Compound"].agg(
        lambda s: s.value_counts().idxmax())
    je_runde["compound"] = mehrheit.reindex(je_runde.index)
    je_runde["nass"] = je_runde["compound"].isin(
        ["INTERMEDIATE", "WET"]).astype(int)

    X = je_runde[["std_sec", "mean_speedFL"]].to_numpy()
    y = je_runde["nass"].to_numpy()
    pred = cross_val_predict(LogisticRegression(), X, y, cv=LeaveOneOut())
    return je_runde, y, pred


def zeichne_wetterprofil(ax, ses) -> None:
    """VORGEHEN 1."""
    w = ses.weather_data
    t = w["Time"].dt.total_seconds() / 60
    ax.plot(t, w["TrackTemp"], color=SERIEN[1], lw=1.8, label="Strecke")
    ax.plot(t, w["AirTemp"], color=SERIEN[0], lw=1.8, label="Luft")
    ax.set_xlabel("Sessionzeit [min]")
    ax.set_ylabel("Temperatur [°C]")
    ax.set_title(f"{TEMP_EVENT[0]} {TEMP_EVENT[1]} - Temperaturverlauf",
                loc="left", color=FG, fontsize=12, pad=10)
    ax.legend(loc="best", frameon=False, labelcolor=FG, fontsize=9)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(alpha=0.3, linewidth=0.8, color=GRID)
    ax.set_axisbelow(True)


def zeichne_temperatureffekt(ax, erg: dict) -> None:
    """VORGEHEN 3."""
    d = erg["dry"]
    ax.scatter(d["TrackTemp"], d["partial"], s=10, color=MUTED, alpha=0.4,
              edgecolors="none")
    xs = np.linspace(d["TrackTemp"].min(), d["TrackTemp"].max(), 50)
    ax.plot(xs, erg["coef_temp"] * xs + erg["intercept"], color=SERIEN[1], lw=2.2)
    ax.set_xlabel("Streckentemperatur [°C]")
    ax.set_ylabel("Rundenzeit ggue. Fahrer-Median,\num TyreLife bereinigt [s]")
    ax.set_title(f"Japan: +{erg['coef_temp']:.3f} s/°C bei TyreLife-"
                f"Kontrolle (R² {erg['r2_tyre_only']:.2f}->{erg['r2_voll']:.2f})",
                loc="left", color=FG, fontsize=11.5, pad=10)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(alpha=0.3, linewidth=0.8, color=GRID)
    ax.set_axisbelow(True)


def zeichne_phasen(ax, phasen: pd.DataFrame, ses) -> None:
    """VORGEHEN 4."""
    for p in phasen.itertuples():
        farbe = SERIEN[1] if p.nass else SERIEN[0]
        ax.axvspan(p.start.total_seconds() / 60, p.end.total_seconds() / 60,
                  color=farbe, alpha=0.7)
    for lbl, farbe in (("trocken", SERIEN[0]), ("Regen gemeldet", SERIEN[1])):
        ax.plot([], [], color=farbe, lw=8, alpha=0.7, label=lbl)
    ax.legend(loc="upper right", frameon=False, labelcolor=FG, fontsize=9)
    ax.set_xlabel("Sessionzeit [min]")
    ax.set_yticks([])
    ax.set_title(f"{NASS_EVENT[0]} {NASS_EVENT[1]} - {len(phasen)} "
                f"Wetter-Phasen", loc="left", color=FG, fontsize=12, pad=10)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)


def zeichne_klassifikator(ax, je_runde: pd.DataFrame, y: np.ndarray,
                          pred: np.ndarray) -> None:
    """AUSBAUSTUFE."""
    richtig = pred == y
    for label, istnass, marker in ((0, False, "o"), (1, True, "^")):
        m = (y == label)
        ax.scatter(je_runde.loc[m, "mean_speedFL"], je_runde.loc[m, "std_sec"],
                  s=[70 if r else 130 for r in richtig[m]],
                  color=SERIEN[1] if istnass else SERIEN[0],
                  marker=marker,
                  edgecolors=["none" if r else FG for r in richtig[m]],
                  linewidths=1.6,
                  label=("Regenreifen" if istnass else "Slicks"), zorder=3)
    acc = richtig.mean()
    ax.set_xlabel("Mittlere Speed-Trap-Geschwindigkeit [km/h]")
    ax.set_ylabel("Rundenzeit-Streuung im Feld [s]")
    ax.set_title(f"{NASS_EVENT[0]} {NASS_EVENT[1]} - Klassifikator, "
                f"LOO-Trefferquote {acc:.1%} (Kreis=Fehler)", loc="left",
                color=FG, fontsize=11.5, pad=10)
    ax.legend(loc="upper left", frameon=False, labelcolor=FG, fontsize=9)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(alpha=0.3, linewidth=0.8, color=GRID)
    ax.set_axisbelow(True)


def main():
    f1lab.enable_cache()

    print(f"[1/4] {TEMP_EVENT[0]} {TEMP_EVENT[1]} laden (VORGEHEN 1-3) ...")
    ses_temp = f1lab.load(TEMP_EVENT[1], TEMP_EVENT[0], TEMP_EVENT[2],
                          telemetry=False, weather=True)
    w = ses_temp.weather_data
    print(w[["Time", "AirTemp", "TrackTemp", "Humidity", "Rainfall", "WindSpeed"]]
         .describe().round(2).to_string())

    merged = wetter_join(ses_temp)
    print(f"\n[2/4] {len(merged)} Runden mit Wetter verknuepft")

    print("\n[3/4] Temperatureffekt (VORGEHEN 3) ...")
    erg = temperatureffekt(merged)
    print(f"      naive gepoolte Regression: {erg['naiv_slope']:+.4f} s/°C, "
         f"R²={erg['naiv_r2']:.3f} - praktisch kein Signal")
    print(f"      kontrolliert (Fahrer-Median abgezogen, TyreLife als "
         f"zweite Variable, n={erg['n']}):")
    print(f"        R² nur TyreLife:        {erg['r2_tyre_only']:.3f}")
    print(f"        R² + TrackTemp:         {erg['r2_voll']:.3f}")
    print(f"        TrackTemp-Koeffizient:  {erg['coef_temp']:+.4f} s/°C "
         f"(se={erg['se_temp']:.4f}, t={erg['coef_temp'] / erg['se_temp']:.1f})")

    print(f"\n[4/4] {NASS_EVENT[0]} {NASS_EVENT[1]} laden (VORGEHEN 4 + "
         f"AUSBAUSTUFE) ...")
    ses_nass = f1lab.load(NASS_EVENT[1], NASS_EVENT[0], NASS_EVENT[2],
                          telemetry=False, weather=True)
    phasen = wetterphasen(ses_nass)
    print(f"      {len(phasen)} Wetter-Phasen:")
    print(phasen.assign(
        start_min=lambda d: (d["start"].dt.total_seconds() / 60).round(1),
        end_min=lambda d: (d["end"].dt.total_seconds() / 60).round(1),
    )[["nass", "start_min", "end_min"]].to_string(index=False))

    print("\n      Klassifikator (AUSBAUSTUFE) ...")
    je_runde, y, pred = klassifikator(ses_nass)
    acc = (pred == y).mean()
    basislinie = max(y.mean(), 1 - y.mean())
    print(f"      n={len(y)} Runden, Leave-one-out-Trefferquote: {acc:.3f} "
         f"(Mehrheitsklasse waere {basislinie:.3f})")
    print(f"      Konfusionsmatrix:\n{confusion_matrix(y, pred)}")
    print(f"      Speed-Trap trocken: {je_runde.loc[y == 0, 'mean_speedFL'].mean():.1f} km/h, "
         f"nass: {je_runde.loc[y == 1, 'mean_speedFL'].mean():.1f} km/h")

    print("\nGrafik ...")
    fig, ax = plt.subplots(2, 2, figsize=(15, 10))
    zeichne_wetterprofil(ax[0, 0], ses_temp)
    zeichne_temperatureffekt(ax[0, 1], erg)
    zeichne_phasen(ax[1, 0], phasen, ses_nass)
    zeichne_klassifikator(ax[1, 1], je_runde, y, pred)
    fig.suptitle("Wetter-Impact auf Pace und Reifenwahl", x=0.09, ha="left",
                fontsize=16, color=FG, y=1.0)
    plt.tight_layout()
    path = OUT / "wetter_impact.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"\n      -> {path}")


if __name__ == "__main__":
    main()
