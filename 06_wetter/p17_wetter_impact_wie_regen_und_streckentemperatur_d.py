"""quantifiziert den effekt von streckentemperatur und regen auf die rundenzeit und klassifiziert nass-/trockenphasen"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")                      # kein fenster, nur dateien

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

import f1lab
from f1lab.design import FG, GRID, MUTED, SERIEN, matplotlib_stil

warnings.filterwarnings("ignore")

OUT = Path(__file__).parent / "out"
OUT.mkdir(exist_ok=True)

TEMP_EVENT = ("Japan", 2024, "R")     # durchgehend trocken mit echter temperaturspanne
NASS_EVENT = ("Canada", 2024, "R")    # echter regen mit echtem mischungswechsel

plt.rcParams.update(matplotlib_stil())


def zeichne_wetterprofil(ax, ses) -> None:
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

    merged = f1lab.weather_join(ses_temp)
    print(f"\n[2/4] {len(merged)} Runden mit Wetter verknuepft")

    print("\n[3/4] Temperatureffekt (VORGEHEN 3) ...")
    erg = f1lab.temperature_effect(merged)
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
    phasen = f1lab.weather_phases(ses_nass)
    print(f"      {len(phasen)} Wetter-Phasen:")
    print(phasen.assign(
        start_min=lambda d: (d["start"].dt.total_seconds() / 60).round(1),
        end_min=lambda d: (d["end"].dt.total_seconds() / 60).round(1),
    )[["nass", "start_min", "end_min"]].to_string(index=False))

    print("\n      Klassifikator (AUSBAUSTUFE) ...")
    je_runde, y, pred = f1lab.wet_dry_classifier(ses_nass)
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
