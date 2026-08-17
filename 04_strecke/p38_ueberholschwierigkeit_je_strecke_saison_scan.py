"""
P38 - Ueberholschwierigkeit je Strecke: Saison-Scan gegen die Streckengeometrie
================================================================================

"Hier kann man nicht ueberholen" ist eine der haeufigsten Behauptungen im F1-Kommentar - stimmt das mit der Streckenform ueberein, oder ist es nur eine Geschichte?

Kategorie:   Strecke & Position
Niveau:      Fortgeschritten
Aufwand:     3-4 h
Schwerpunkt: Datenanalyse, Statistik

WARUM DAS LOHNT
Kombiniert zwei bereits vorhandene Bausteine (P20s Ueberholmatrix, P02/P11s
Streckengeometrie) zu einer neuen Frage ueber eine ganze Saison hinweg, statt
ein einzelnes Rennen zu betrachten. Und eine Lektion in Korrelationskritik:
mit nur 12 Strecken mit Telemetrie im Cache ist jede Korrelation empfindlich
gegen einzelne Punkte - das gehoert offen dazu, nicht nur die Zahl selbst.

VORGEHEN
  1. Ueberholungen je Rennen ueber die ganze Saison 2024 zaehlen
     (f1lab.overtakes_matrix, braucht keine Telemetrie)
  2. Streckengeometrie (Kurvenzahl, Laenge) fuer die telemetriefaehigen
     Rennen laden (f1lab.circuit_dimension, P02)
  3. Kurven pro Kilometer als Proxy fuer "Strassenkurs-Charakter" ableiten
     (viele enge Kurven auf kurzer Strecke gegen wenige weite auf langer)
  4. Korrelation Ueberholungen gegen Streckenlaenge/Kurvenzahl/Kurven-pro-km
     pruefen (Pearson, dazu Spearman als robusterer Gegencheck)

GENUTZTE FASTF1-BAUSTEINE
  - session.laps (Positions-/Boxenstopp-/Flaggendaten, keine Telemetrie noetig
    fuer VORGEHEN 1)
  - get_circuit_info, get_pos_data (ueber f1lab.circuit_dimension)
  - scipy.stats.pearsonr/spearmanr

AUSBAUSTUFE  [umgesetzt]
Robustheit der Korrelation pruefen: einmal ohne den staerksten Einzelpunkt
(Monaco), einmal mit dem ausreisserrobusten Spearman-Rangkorrelation statt
Pearson. Zweite AUSBAUSTUFE: dieselbe Frage an Safety-Car-/VSC-Dauer statt
Ueberholungen stellen (f1lab.track_status_phases, P18) - ist DAS eine
Streckeneigenschaft?

Alle 24 Rennen 2024 im Cache liefern Ueberholzahlen von 6 (Monaco) bis 215
(Las Vegas). Fuer die 12 Rennen mit Telemetrie zeigt sich: die reine
Streckenlaenge sagt nichts voraus (Pearson r=0.111, p=0.73) - eine lange
Strecke ist nicht automatisch ueberholfreundlich. Kurvenzahl allein
korreliert schon deutlicher negativ (r=-0.629, p=0.028), Kurven pro
Kilometer - der eigentliche "Strassenkurs-Charakter", der auch kurze,
enge Strecken wie Monaco von langen, offenen wie Spielberg unterscheidet -
am staerksten (r=-0.712, p=0.009): mehr Kurven auf derselben Streckenlaenge,
weniger Ueberholungen.

Die AUSBAUSTUFE zeigt die Grenze dieser Zahl bei n=12 ehrlich: Monaco ist
sowohl das Extrem bei Kurven/km (5.8) als auch bei Ueberholungen (6) -
ohne diesen einen Punkt faellt die Korrelation auf r=-0.573 (p=0.065, nicht
mehr signifikant bei alpha=0.05, aber Richtung und Groessenordnung bleiben).
Die ausreisserrobuste Spearman-Rangkorrelation ueber alle 12 Strecken
bestaetigt den Befund unabhaengig von der Metrik-Empfindlichkeit gegenueber
Monaco (r=-0.643, p=0.024) - der Zusammenhang ist also nicht allein ein
Monaco-Artefakt, aber mit einer einzigen sehr eintypischen Strecke im
Datensatz ist Vorsicht angebracht. Drei Kandidaten-Metriken getestet: selbst
mit einer konservativen Mehrfachtest-Korrektur (alpha/3=0.017) bleibt nur
Kurven/km klar signifikant.

Was diese Zahl NICHT erklaert: die Ueberholzahlen selbst sind stark von
Rennverlauf gepraegt, nicht nur von Geometrie - Las Vegas (215) und Ungarn
(193, ausserhalb der 12-Strecken-Stichprobe ohne Telemetrie) waren beide
chaotische Rennen mit grossen Reifen-/Strategieunterschieden, waehrend ein
sauberes Rennen auf derselben Strecke deutlich weniger Ueberholungen zeigen
koennte. Die Korrelation misst "diese eine Saison", keine
Streckeneigenschaft im Vakuum.

ZWEITE AUSBAUSTUFE  [umgesetzt]
Dieselbe Frage ("ist das eine Streckeneigenschaft?") an eine zweite
Rennverlaufs-Groesse gestellt: Safety-Car-/VSC-Dauer je Rennen
(f1lab.track_status_phases, P18), wieder gegen Kurven pro Kilometer.
13/24 Rennen 2024 hatten mindestens eine Neutralisation (Bereich 0-1397s).
Anders als bei Ueberholungen zeigt sich hier PRAKTISCH KEIN Zusammenhang
(Pearson r=+0.044, p=0.89; Spearman r=+0.276, p=0.39, n=12) - eine
kurvenreiche Strecke ueberholt schwerer, aber sie ist nicht dadurch schon
unfallanfaelliger. Plausibel: Safety-Car-Ausloeser sind ueberwiegend
Kollisionen, technische Ausfaelle und Wetter - Ereignisse, die eher mit
Feldgroesse, erstem-Kurve-Chaos oder Zuverlaessigkeit zusammenhaengen als
mit der Streckengeometrie selbst. Die zwei Rennen mit der laengsten
Neutralisationsdauer (Katar 1397s, China 1295s) liegen bei Kurven/km nahe
der Mitte der Stichprobe, nicht an einem Extrem.

Zusammengenommen zeigen P38 und seine zweite AUSBAUSTUFE denselben
methodischen Punkt aus zwei Richtungen: nicht jede auffaellige
Rennstatistik ist eine Streckeneigenschaft, nur weil man sie gegen
Streckengeometrie auftragen KANN - Ueberholschwierigkeit ist es (mit den
oben genannten Einschraenkungen), Safety-Car-Haeufigkeit in dieser
Stichprobe nicht.
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
from scipy.stats import pearsonr, spearmanr

import f1lab
from f1lab.design import FG, GRID, MUTED, SERIEN, matplotlib_stil

warnings.filterwarnings("ignore")

OUT = Path(__file__).parent / "out"
OUT.mkdir(exist_ok=True)

SEASON = 2024

plt.rcParams.update(matplotlib_stil())


def zeichne_saison(ax, ueberholungen: pd.DataFrame) -> None:
    """VORGEHEN 1: Ueberholungen je Rennen, ganze Saison."""
    u = ueberholungen.sort_values("overtakes")
    farben = [SERIEN[0] if gp != "Monaco Grand Prix" else SERIEN[1]
             for gp in u["gp"]]
    ax.barh(u["gp"], u["overtakes"], color=farben, height=0.65)
    ax.set_xlabel("Ueberholungen (gruene Flagge, ohne Boxenstopp-Effekt)")
    ax.set_title(f"Ueberholungen je Rennen, Saison {SEASON} "
                f"({len(u)} Rennen)", loc="left", color=FG, fontsize=13,
                pad=10)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="x", alpha=0.3, linewidth=0.8, color=GRID)
    ax.set_axisbelow(True)


def zeichne_korrelation(ax, geo: pd.DataFrame, r: float, p: float) -> None:
    """VORGEHEN 4 + AUSBAUSTUFE: Kurven/km gegen Ueberholungen, 12 Strecken
    mit Telemetrie."""
    ax.scatter(geo["kurven_pro_km"], geo["overtakes"], s=70, color=SERIEN[0],
              zorder=3)
    for _, row in geo.iterrows():
        ax.annotate(row["circuit"], (row["kurven_pro_km"], row["overtakes"]),
                   xytext=(6, 4), textcoords="offset points", color=MUTED,
                   fontsize=9)
    steigung, achse0 = np.polyfit(geo["kurven_pro_km"], geo["overtakes"], 1)
    x_fit = np.linspace(geo["kurven_pro_km"].min(), geo["kurven_pro_km"].max(), 50)
    ax.plot(x_fit, steigung * x_fit + achse0, color=MUTED, lw=1.2, ls="--")
    ax.set_xlabel("Kurven pro Kilometer")
    ax.set_ylabel("Ueberholungen")
    ax.set_title(f"Kurvendichte gegen Ueberholungen, {len(geo)} Strecken mit "
                f"Telemetrie (Pearson r={r:.2f}, p={p:.3f})", loc="left",
                color=FG, fontsize=13, pad=10)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(alpha=0.3, linewidth=0.8, color=GRID)
    ax.set_axisbelow(True)


def zeichne_sc_korrelation(ax, geo: pd.DataFrame, r: float, p: float) -> None:
    """ZWEITE AUSBAUSTUFE: Kurven/km gegen Safety-Car-/VSC-Dauer, dieselben
    12 Strecken."""
    ax.scatter(geo["kurven_pro_km"], geo["sc_dauer_s"], s=70, color=SERIEN[2],
              zorder=3)
    for _, row in geo.iterrows():
        ax.annotate(row["circuit"], (row["kurven_pro_km"], row["sc_dauer_s"]),
                   xytext=(6, 4), textcoords="offset points", color=MUTED,
                   fontsize=9)
    ax.set_xlabel("Kurven pro Kilometer")
    ax.set_ylabel("Safety-Car-/VSC-Dauer [s]")
    ax.set_title(f"ZWEITE AUSBAUSTUFE: Kurvendichte gegen SC-Dauer "
                f"(Pearson r={r:+.2f}, p={p:.2f}) - keine Streckeneigenschaft",
                loc="left", color=FG, fontsize=13, pad=10)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(alpha=0.3, linewidth=0.8, color=GRID)
    ax.set_axisbelow(True)


def main():
    f1lab.enable_cache()

    print(f"[1/3] Ueberholungen je Rennen, Saison {SEASON} (VORGEHEN 1, keine "
         "Telemetrie noetig) ...")
    inv = f1lab.cached_sessions()
    rennen = sorted(inv[(inv["season"] == SEASON) & (inv["ident"] == "R")]
                    ["event"].unique())
    zeilen = []
    for gp in rennen:
        try:
            ses = f1lab.load(SEASON, gp, "R", telemetry=False)
        except Exception:
            continue
        n = int(f1lab.overtakes_matrix(ses).values.sum())
        zeilen.append({"gp": gp, "overtakes": n})
    ueberholungen = pd.DataFrame(zeilen)
    print(f"      {len(ueberholungen)} Rennen, "
         f"{ueberholungen['overtakes'].min()}-{ueberholungen['overtakes'].max()} "
         "Ueberholungen (Min-Max)")

    print("\n[2/3] Streckengeometrie fuer die telemetriefaehigen Rennen "
         "(VORGEHEN 2-3) ...")
    tel_rennen = sorted(inv[(inv["season"] == SEASON) & (inv["ident"] == "R")
                            & inv["telemetry"]]["event"].unique())
    geo = f1lab.circuit_dimension([(SEASON, gp) for gp in tel_rennen])
    geo = geo.merge(ueberholungen, on="gp", how="inner")
    geo["kurven_pro_km"] = geo["corners"] / (geo["length_m"] / 1000)
    print(geo[["gp", "circuit", "corners", "length_m", "kurven_pro_km",
              "overtakes"]].to_string(index=False))

    print(f"\n[3/3] Korrelationen ueber {len(geo)} Strecken (VORGEHEN 4, "
         "AUSBAUSTUFE) ...")
    r_len, p_len = pearsonr(geo["length_m"], geo["overtakes"])
    r_corn, p_corn = pearsonr(geo["corners"], geo["overtakes"])
    r_kpk, p_kpk = pearsonr(geo["kurven_pro_km"], geo["overtakes"])
    print(f"      Laenge:        r={r_len:+.3f}  p={p_len:.3f}")
    print(f"      Kurvenzahl:    r={r_corn:+.3f}  p={p_corn:.3f}")
    print(f"      Kurven/km:     r={r_kpk:+.3f}  p={p_kpk:.3f}")

    ohne_monaco = geo[geo["circuit"] != "Monaco"]
    r_om, p_om = pearsonr(ohne_monaco["kurven_pro_km"], ohne_monaco["overtakes"])
    r_sp, p_sp = spearmanr(geo["kurven_pro_km"], geo["overtakes"])
    print(f"\n      AUSBAUSTUFE ohne Monaco (n={len(ohne_monaco)}): "
         f"r={r_om:+.3f}  p={p_om:.3f}")
    print(f"      AUSBAUSTUFE Spearman (n={len(geo)}):        "
         f"r={r_sp:+.3f}  p={p_sp:.3f}")

    print("\nZWEITE AUSBAUSTUFE: Safety-Car-/VSC-Dauer gegen dieselbe "
         "Streckengeometrie ...")
    sc_zeilen = []
    for gp in rennen:
        ses = f1lab.load(SEASON, gp, "R", telemetry=False)
        ph = f1lab.track_status_phases(ses)
        neutral = ph[ph["label"].isin(["safety car", "vsc"])]
        sc_zeilen.append({"gp": gp, "sc_phasen": len(neutral),
                          "sc_dauer_s": round(float(neutral["duration_s"].sum()), 0)})
    sc = pd.DataFrame(sc_zeilen)
    print(f"      {(sc['sc_phasen'] > 0).sum()}/{len(sc)} Rennen mit "
         "SC/VSC")
    geo = geo.merge(sc, on="gp", how="inner")
    r_sc, p_sc = pearsonr(geo["kurven_pro_km"], geo["sc_dauer_s"])
    r_sc_sp, p_sc_sp = spearmanr(geo["kurven_pro_km"], geo["sc_dauer_s"])
    print(f"      Kurven/km gegen SC-Dauer (n={len(geo)}): "
         f"Pearson r={r_sc:+.3f} p={p_sc:.3f}, "
         f"Spearman r={r_sc_sp:+.3f} p={p_sc_sp:.3f}")

    print("\nGrafik ...")
    fig, ax = plt.subplots(3, 1, figsize=(13, 17))
    zeichne_saison(ax[0], ueberholungen)
    zeichne_korrelation(ax[1], geo, r_kpk, p_kpk)
    zeichne_sc_korrelation(ax[2], geo, r_sc, p_sc)
    fig.suptitle("Ueberholschwierigkeit je Strecke: Saison-Scan", x=0.09,
                ha="left", fontsize=16, color=FG, y=0.995)
    plt.tight_layout()
    path = OUT / "ueberholschwierigkeit_saison_scan.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"\n      -> {path}")


if __name__ == "__main__":
    main()
