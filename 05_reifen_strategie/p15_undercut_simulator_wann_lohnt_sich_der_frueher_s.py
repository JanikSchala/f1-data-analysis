"""
P15 - Undercut-Simulator: Wann lohnt sich der frueher Stopp?
============================================================

Ein Modell aus Degradation, Pitloss und Out-Lap-Performance, das den Undercut-Gewinn in Sekunden ausrechnet.

Kategorie:   Reifen & Strategie
Niveau:      Profi
Aufwand:     5-6 h
Schwerpunkt: Strategie

WARUM DAS LOHNT
Das ist Race Strategy in Reinform. Ein funktionierender Simulator mit aus Daten geschaetzten Parametern ist das staerkste Einzelprojekt fuer eine Strategie-Rolle.

VORGEHEN
  1. Pitloss aus echten Daten schaetzen (Delta In-/Out-Lap zur Normalrunde)
  2. Degradation je Compound aus Projekt 13 uebernehmen
  3. Out-Lap-Malus und Warm-up-Verhalten quantifizieren
  4. Simulation ueber Undercut-Fenster von 1-5 Runden
  5. Sensitivitaets-Heatmap: Deg x Pitloss

GENUTZTE FASTF1-BAUSTEINE
  - Laps PitInTime/PitOutTime
  - TyreLife
  - Laps.pick_box_laps
  - eigene Simulation

AUSBAUSTUFE  [umgesetzt]
Erweitere um Verkehr: modelliere, wie viele Sekunden der Undercut kostet,
wenn du hinter einem langsameren Auto herauskommst.

Alle fuenf VORGEHEN-Punkte mit echten Werten aus Spanien 2024 R statt
angenommenen Konstanten. Zwei Ueberraschungen dabei:

(1) Der ueblicherweise angenommene Out-Lap-Malus von 0.5-1.0 s (VORGEHEN 3)
laesst sich in den Daten nicht finden. f1lab.pit_loss() misst 23.1 s volle
Boxengassenzeit (in-lap 4.9 s, out-lap 18.2 s ueber Normalrunde - dominiert
vom Tempolimit, nicht vom kalten Reifen). Die erste NACH der Boxengasse
gewertete Runde (TyreLife=1, schon in f1lab.clean_laps()) zeigt gegen den
eigenen Degradations-Fit praktisch keinen Rest-Malus (Median -0.15 s ueber
51 belastbare Stints, nicht von 0 unterscheidbar). Der Simulator nutzt
deshalb 0.0 s als datengestuetzten Out-Lap-Malus statt der ueblichen
Faustregel - das macht den Undercut in diesem Modell noch attraktiver, als
die Faustregel vermuten laesst.

(2) undercut_gain() waechst mit wachsendem n_laps unbegrenzt, sobald die
neue Mischung dauerhaft flacher degradiert als die alte (SOFT 0.127 s/Runde
gegen MEDIUM 0.089 s/Runde in Spanien) - es gibt kein inneres Optimum, "je
laenger warten, desto besser" ist im Modell immer wahr. Das ist kein Bug,
sondern eine Grenze des linearen Modells: es ignoriert, dass Degradation ab
einem Punkt in einen Cliff kippt (P13), dass Reifen nicht ewig halten, und
dass Verkehr/Undercut-Fenster real begrenzt sind. Statt eines erfundenen
Optimums zeigt die Sensitivitaets-Grafik deshalb den Gewinn bei einem festen,
strategisch ueblichen Fenster (3 Runden) ueber Degradations-Differenz und
Out-Lap-Malus (VORGEHEN 5 nennt "Deg x Pitloss" - Pitloss faellt aber laut
der eigenen Kommentierung in core.undercut_gain() aus der Rechnung heraus,
weil er fuer beide Fahrer gleich anfaellt; die tatsaechlich wirksamen zwei
Groessen sind Degradations-Differenz und Out-Lap-Malus).

AUSBAUSTUFE Verkehr: die naheliegende Zahl - wie viel kostet Distanz zum
Vordermann - ist ueber f1lab.dirty_air_effect() (P32) fuer alle 20 Fahrer
in Spanien 2024 R geschaetzt. Ergebnis: kein einziger Fahrer-Fit erreicht
das im Projekt sonst verwendete Belastbarkeits-Kriterium R^2 >= 0.3
(DegradationFit.is_reliable) - Median-Steigung +0.0031 s je Prozentpunkt
Nahanteil, Spanne -0.006 bis +0.009, hoechstes R^2 0.14. Eine echte
Verkehrs-Erweiterung des Simulators braucht deshalb mehr als ein Rennen;
mit diesem einen laesst sie sich nicht seriös parametrisieren. Als Beleg
dafuer, wie selten der Fall im Rennen ueberhaupt vorkommt: von allen
Boxenstopps mit mindestens 3 nachfolgenden gewerteten Runden gerieten nur
zwei (Ocon, Zhou) ueberhaupt in echten Nahverkehr (>=50 % Nahanteil,
zusammen drei betroffene Runden) direkt nach dem Stopp.
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

warnings.filterwarnings("ignore")

OUT = Path(__file__).parent / "out"
OUT.mkdir(exist_ok=True)

SEASON, EVENT, IDENT = 2024, "Spain", "R"
RAMPE_CMAP = LinearSegmentedColormap.from_list("rampe", RAMPE)

plt.rcParams.update(matplotlib_stil())


def pitloss_zerlegen(ses) -> tuple[float, float, float]:
    """VORGEHEN 1: Pitloss wie f1lab.pit_loss(), aber in/out getrennt
    ausgewiesen statt nur die Summe."""
    laps = ses.laps.copy()
    laps["sec"] = laps["LapTime"].dt.total_seconds()
    baseline = (f1lab.clean_laps(ses).groupby("Driver")["LapTime"]
               .median().dt.total_seconds())
    in_laps = laps[laps["PitInTime"].notna()]
    out_laps = laps[laps["PitOutTime"].notna()]
    loss_in = (in_laps["sec"] - in_laps["Driver"].map(baseline)).dropna()
    loss_out = (out_laps["sec"] - out_laps["Driver"].map(baseline)).dropna()
    return float(loss_in.median()), float(loss_out.median()), \
        float(loss_in.median() + loss_out.median())


def out_lap_malus_schaetzen(ses) -> np.ndarray:
    """VORGEHEN 3: Residuum der ersten gewerteten Runde eines Stints
    (TyreLife=Minimum) gegen den eigenen Degradations-Fit. Positiv heisst
    langsamer als der Trend vorhersagt - das waere der gesuchte Malus."""
    laps = f1lab.clean_laps(ses).copy()
    laps["sec"] = laps["LapTime"].dt.total_seconds()
    laps["corrected"] = f1lab.fuel_correct(
        laps["sec"], laps["LapNumber"], ses.total_laps)
    resid = []
    for (_drv, _stint), g in laps.groupby(["Driver", "Stint"]):
        g = g.sort_values("TyreLife")
        if len(g) < 6:
            continue
        try:
            fit = f1lab.fit_degradation(g["TyreLife"], g["corrected"])
        except ValueError:
            continue
        if not fit.is_reliable:
            continue
        erste = g.iloc[0]
        pred = fit.slope * erste["TyreLife"] + fit.intercept
        resid.append(erste["corrected"] - pred)
    return np.array(resid)


def verkehr_nach_stopp(ses) -> pd.DataFrame:
    """AUSBAUSTUFE, Teil 1: dirty_air_effect() je Fahrer, plus wie oft ein
    Fahrer unmittelbar nach dem eigenen Stopp in echten Nahverkehr geraet."""
    stint_map = ses.laps.set_index(["Driver", "LapNumber"])["Stint"]
    zeilen = []
    for drv in ses.drivers:
        abbr = ses.get_driver(drv)["Abbreviation"]
        try:
            df = f1lab.close_following(ses, abbr)
        except Exception:
            continue
        if df.empty:
            continue
        slope, _inter, r2, d = f1lab.dirty_air_effect(df)
        if slope != slope:
            continue
        d = d.copy()
        d["stint"] = [stint_map.get((abbr, lap), np.nan) for lap in d["lap"]]
        nach_stopp = d[(d["stint"] > 1) & (d["tyre_life"] <= 3)]
        zeilen.append({
            "driver": abbr, "slope": slope, "r2": r2, "n": len(d),
            "nah_nach_stopp": int((nach_stopp["anteil_nah"] >= 50).sum()),
        })
    return pd.DataFrame(zeilen)


def zeichne_fenster(ax, deg_alt: float, deg_neu: float, malus: float) -> None:
    """VORGEHEN 4: Gewinn ueber das Undercut-Fenster, 1-8 Runden."""
    ns = np.arange(1, 9)
    gains = [f1lab.undercut_gain(deg_alt, deg_neu, n, malus) for n in ns]
    ax.plot(ns, gains, marker="o", color=SERIEN[0], lw=2, ms=6)
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.set_xlabel("Undercut-Fenster [Runden]")
    ax.set_ylabel("Gewinn [s]")
    ax.set_title(f"SOFT ({deg_alt:.3f}) vs. MEDIUM ({deg_neu:.3f} s/Runde)",
                loc="left", color=FG, fontsize=11.5, pad=10)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(alpha=0.3, linewidth=0.8, color=GRID)
    ax.set_axisbelow(True)


def zeichne_sensitivitaet(ax, deg_neu: float) -> None:
    """VORGEHEN 5 (angepasst, siehe Docstring): Gewinn bei 3 Runden ueber
    Degradations-Differenz und Out-Lap-Malus."""
    # Untergrenze bei deg_neu: darunter degradiert die "alte" Mischung
    # langsamer als die neue - kein realistisches Undercut-Szenario, wuerde
    # die Skala nur mit stark negativen Werten stauchen.
    degs_alt = np.round(np.arange(deg_neu, 0.21, 0.01), 3)
    malusse = np.round(np.arange(0.0, 1.3, 0.1), 2)
    grid = np.array([[f1lab.undercut_gain(d, deg_neu, 3, m) for m in malusse]
                     for d in degs_alt])
    im = ax.imshow(grid, aspect="auto", cmap=RAMPE_CMAP, origin="lower",
                   extent=[malusse[0], malusse[-1], degs_alt[0], degs_alt[-1]])
    ax.axhline(0.1267, color=FG, lw=1, ls="--")
    ax.text(malusse[-1], 0.1267, " gemessen (SOFT)", color=FG, fontsize=8,
           va="bottom", ha="right")
    ax.axvline(0.0, color=FG, lw=1, ls=":")
    ax.text(0.02, degs_alt[0], "gemessen (kein Malus)", color=FG, fontsize=8,
           va="bottom", ha="left")
    ax.set_xlabel("Out-Lap-Malus [s]")
    ax.set_ylabel("Degradation alte Mischung [s/Runde]")
    ax.set_title("Gewinn bei 3 Runden Fenster [s]", loc="left", color=FG,
                fontsize=11.5, pad=10)
    cb = ax.figure.colorbar(im, ax=ax, pad=0.02)
    cb.set_label("Gewinn [s]", color=MUTED, fontsize=9)
    cb.ax.tick_params(colors=MUTED, length=0)
    cb.outline.set_visible(False)


def zeichne_verkehr(ax, tab: pd.DataFrame) -> None:
    """AUSBAUSTUFE, Teil 2: Steigung je Fahrer, gegen die im Projekt sonst
    genutzte Belastbarkeitsschwelle R^2 >= 0.3."""
    t = tab.sort_values("slope")
    farben = [SERIEN[0] if r >= 0.3 else MUTED for r in t["r2"]]
    ax.barh(t["driver"], t["slope"], color=farben, height=0.65)
    ax.axvline(0, color=MUTED, lw=0.8)
    ax.set_xlabel("Steigung [s pro Prozentpunkt Nahanteil]")
    ax.set_title("Dirty-Air-Steigung je Fahrer - kein Fit erreicht R²>=0.3",
                loc="left", color=FG, fontsize=12, pad=10)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="x", alpha=0.3, linewidth=0.8, color=GRID)
    ax.set_axisbelow(True)


def main():
    f1lab.enable_cache()

    print(f"[1/5] {EVENT} {SEASON} {IDENT} laden (mit Telemetrie, fuer die "
         f"Verkehrs-Ausbaustufe) ...")
    ses = f1lab.load(SEASON, EVENT, IDENT, telemetry=True)

    print("\n[2/5] Pitloss (VORGEHEN 1) ...")
    loss_in, loss_out, pitloss = pitloss_zerlegen(ses)
    print(f"      in-lap {loss_in:.2f}s + out-lap {loss_out:.2f}s = "
         f"{pitloss:.2f}s (f1lab.pit_loss() zur Kontrolle: "
         f"{f1lab.pit_loss(ses):.2f}s)")

    print("\n[3/5] Degradation je Mischung (VORGEHEN 2, aus P13-Bausteinen) ...")
    je_compound = f1lab.degradation_by_compound(ses)
    print(je_compound.to_string())
    deg_alt = float(je_compound.loc["SOFT", "mean"])
    deg_neu = float(je_compound.loc["MEDIUM", "mean"])
    print(f"      DEG_ALT (SOFT) = {deg_alt:.4f}, DEG_NEU (MEDIUM) = {deg_neu:.4f}")

    print("\n[4/5] Out-Lap-Malus (VORGEHEN 3) ...")
    resid = out_lap_malus_schaetzen(ses)
    print(f"      n={len(resid)}  Median={np.median(resid):+.3f}s  "
         f"Mittel={resid.mean():+.3f}s -> kein messbarer Rest-Malus, "
         f"Simulator nutzt 0.0s")
    malus = 0.0

    print("\n[5/5] Simulation (VORGEHEN 4) und Sensitivitaet (VORGEHEN 5) ...")
    for n in range(1, 9):
        g = f1lab.undercut_gain(deg_alt, deg_neu, n, malus)
        print(f"      {n} Runden fruehe stoppen -> {g:+.2f}s")

    print("\nAUSBAUSTUFE: Verkehr ...")
    verkehr = verkehr_nach_stopp(ses)
    print(verkehr.round(4).to_string(index=False))
    print(f"      Median Steigung {verkehr['slope'].median():+.4f}s/%pkt, "
         f"max R^2 {verkehr['r2'].max():.2f}, Fits mit R^2>=0.3: "
         f"{(verkehr['r2'] >= 0.3).sum()}/{len(verkehr)}")
    print(f"      Stopps mit >=50% Nahanteil in den 3 Runden danach: "
         f"{int(verkehr['nah_nach_stopp'].sum())}")

    print("\nGrafik ...")
    fig = plt.figure(figsize=(15, 11))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1], hspace=0.45, wspace=0.3)
    zeichne_fenster(fig.add_subplot(gs[0, 0]), deg_alt, deg_neu, malus)
    zeichne_sensitivitaet(fig.add_subplot(gs[0, 1]), deg_neu)
    zeichne_verkehr(fig.add_subplot(gs[1, :]), verkehr)
    fig.suptitle(f"{ses.event['EventName']} {SEASON} - Undercut-Simulator",
                x=0.09, ha="left", fontsize=16, color=FG, y=0.995)
    path = OUT / "undercut_simulator.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"\n      -> {path}")


if __name__ == "__main__":
    main()
