"""
P40 - Startplatz-Paritaet: hat die Startseite einen echten Effekt?
====================================================================

"Die schmutzige Seite" ist eine der aeltesten Behauptungen im Motorsport - weniger Gummi auf einer Grid-Haelfte soll Startplaetze dort benachteiligen. Zeigt sich das in echten Daten, und haelt es ueber mehrere Jahre an derselben Strecke?

Kategorie:   Timing & Rundenanalyse
Niveau:      Profi
Aufwand:     4-5 h
Schwerpunkt: Datenanalyse, Statistik

WARUM DAS LOHNT
Reine Positions-/Ergebnisdaten (kein Telemetrie-Download noetig), aber der
ganze Cache auf einmal (so viele Saisons wie fuer eine Strecke vorliegen)
- und eine Lektion in Stichprobengroesse: der naheliegende erste Versuch
(eine Saison, alle Strecken gepoolt) findet nichts, weil die "saubere"
Seite je Strecke wechselt und sich beim Poolen gegenseitig aufhebt. Erst
je Strecke einzeln, ueber mehrere Jahre gepoolt, wird ein Effekt sichtbar
- und dann muss er gegen Mehrfachtests UND gegen Saison-zu-Saison-
Konsistenz geprueft werden, nicht nur gegen einen p-Wert.

VORGEHEN
  1. Startplatz gegen Position am Ende von Runde 1 fuer jeden Start im
     ganzen Cache laden (f1lab.grid_lap1_positions, keine Telemetrie
     noetig)
  2. Boxenstarts ausschliessen (wie P31)
  3. Parität zuordnen (gerade/ungerade Startplatz) und je Strecke mit
     mindestens 4 Saisons im Cache testen: ungerade gegen gerade (t-Test)
  4. Fuer die auffaelligsten Strecken pruefen, ob die Richtung des Effekts
     in JEDER einzelnen Saison gleich ist, nicht nur im gepoolten Mittel

GENUTZTE FASTF1-BAUSTEINE
  - Session.results (GridPosition)
  - Laps.pick_laps(1), PitOutTime
  - scipy.stats.ttest_ind

Unerwarteter Fund unterwegs: anders als P38/P39 (reine lokale Timing-Daten,
komplett offline reproduzierbar) braucht ``Session.results`` fuer manche
Saisons echten Netzzugriff auf Ergast/jolpica, um GridPosition
nachzuladen - `f1lab.enable_cache()` ohne ``offline=True`` erlaubt das,
und der erste Durchlauf traf wiederholt das Rate-Limit (429). FastF1 faengt
das intern ab und faellt auf eine gecachte Antwort zurueck, bricht also
nicht ab - aber jeder Lauf, der neue 429-Antworten in eine erfolgreiche
umwandelt, vergroessert den lokalen Cache dauerhaft. Drei Laeufe
hintereinander lieferten deshalb wachsende Zahlen (2441 -> 3283 -> 3418
Starts), bevor sie sich stabilisierten - dieselbe Netzwerk-Abhaengigkeit
wie in P16 (11_Boxenstopps.py), hier aber unerwartet, weil GridPosition
wie eine reine Lokaldaten-Spalte aussieht.

AUSBAUSTUFE  [umgesetzt]
Mehrfachtest-Korrektur: bei 25 getesteten Strecken sind bei reinem Zufall
im Schnitt ~1.2 "signifikante" Treffer bei alpha=0.05 zu erwarten - wie
viele der gefundenen ueberleben eine Bonferroni-Korrektur (alpha/25)?

3418 Starts (ohne Boxenstarts) ueber 180 Rennen, 37 Strecken (stabilisierter
Stand, siehe oben). Fuer die 25 Strecken mit mindestens 4 Saisons im Cache:
fuenf zeigen einen signifikanten Paritaets-Effekt bei alpha=0.05 - Saudi-
Arabien (p=0.004, ungerade +1.26 Positionen), Australien (p=0.011, +1.03),
Niederlande (p=0.031, -0.93), Mexiko (p=0.040, -1.40), Belgien (p=0.041,
+1.06). Das ist mehr als die ~1.2 Zufallstreffer bei 25 Tests, aber KEINE
der fuenf Strecken uebersteht eine Bonferroni-Korrektur (alpha/25=0.0020) -
selbst Saudi-Arabiens p=0.004 nicht ganz.

Die AUSBAUSTUFE zeigt trotzdem einen echten Unterschied zwischen den fuenf
Kandidaten: Saudi-Arabien und die Niederlande zeigen in JEDER einzelnen
verfuegbaren Saison dieselbe Richtung (Saudi-Arabien 2021-2025 durchweg
ungerade im Vorteil, +0.22 bis +2.40; Niederlande 2019-2025 durchweg
gerade im Vorteil, -0.21 bis -2.32) - eine viel staerkere Evidenz als der
gepoolte p-Wert allein. Mexiko haelt die Richtung in 4 von 5 Saisons (nur
2022 kippt leicht positiv, +0.20). Australien und Belgien dagegen nicht:
Australien kippt 2023 in die Gegenrichtung (-0.99) und der gepoolte Befund
haengt spuerbar an einem einzelnen Ausreisser (2026, +4.21, die kuerzeste
und unvollstaendigste Saison im Cache); Belgien kippt sogar zweimal
(2020 -0.88, 2022 -1.39) bei sonst durchweg positiven, teils extremen
Werten (2019 +3.22) - der unruhigste der fuenf Befunde trotz
"signifikantem" p-Wert.

Das Gesamtbild: eine echte, konsistente Paritaets-Wirkung existiert
vermutlich an mindestens zwei Strecken (Saudi-Arabien, Niederlande), ist
aber die Ausnahme, nicht die Regel - 20 von 25 getesteten Strecken zeigen
keinen belastbaren Effekt, und selbst die staerksten Kandidaten ueberleben
keine strenge statistische Korrektur. "Schmutzige Seite" ist damit weder
Mythos noch Universalgesetz, sondern eine streckenspezifische Ausnahme.
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
from scipy.stats import ttest_ind

import f1lab
from f1lab.design import FG, GRID, MUTED, SERIEN, matplotlib_stil

warnings.filterwarnings("ignore")

OUT = Path(__file__).parent / "out"
OUT.mkdir(exist_ok=True)

MIN_SAISONS = 4
KONSISTENZ_STRECKEN = ["Saudi Arabian Grand Prix", "Australian Grand Prix",
                       "Belgian Grand Prix", "Dutch Grand Prix",
                       "Mexico City Grand Prix"]

plt.rcParams.update(matplotlib_stil())


def zeichne_strecken(ax, ergebnisse: pd.DataFrame) -> None:
    """VORGEHEN 3-4: Paritaets-Effekt je Strecke, signifikante hervorgehoben."""
    e = ergebnisse.sort_values("diff")
    farben = [SERIEN[1] if p < 0.05 else MUTED for p in e["p"]]
    ax.barh(e["gp"], e["diff"], color=farben, height=0.65)
    ax.axvline(0, color=MUTED, lw=1)
    ax.set_xlabel("Ungerade minus gerade Startplaetze [Positionen nach Runde 1]")
    ax.set_title(f"Paritaets-Effekt je Strecke ({len(e)} Strecken, "
                f"mind. {MIN_SAISONS} Saisons; rot = p<0.05)", loc="left",
                color=FG, fontsize=13, pad=10)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="x", alpha=0.3, linewidth=0.8, color=GRID)
    ax.set_axisbelow(True)


def zeichne_konsistenz(ax, saison_diffs: pd.DataFrame) -> None:
    """AUSBAUSTUFE: haelt die Richtung des Effekts jede einzelne Saison?"""
    strecken = KONSISTENZ_STRECKEN
    breite = 0.8 / max(len(saison_diffs["season"].unique()), 1)
    for i, s in enumerate(sorted(saison_diffs["season"].unique())):
        werte = saison_diffs[saison_diffs["season"] == s].set_index("gp")
        werte = werte.reindex(strecken)
        x = np.arange(len(strecken)) + i * breite
        ax.bar(x, werte["diff"], width=breite, color=SERIEN[i % len(SERIEN)],
              label=str(s))
    ax.axhline(0, color=MUTED, lw=1)
    ax.set_xticks(np.arange(len(strecken)) + 0.35)
    ax.set_xticklabels([s.replace(" Grand Prix", "") for s in strecken])
    ax.set_ylabel("Ungerade minus gerade [Positionen]")
    ax.set_title("AUSBAUSTUFE: haelt die Richtung in jeder einzelnen Saison?",
                loc="left", color=FG, fontsize=13, pad=10)
    ax.legend(loc="upper right", frameon=False, labelcolor=FG, fontsize=8,
             ncol=3)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="y", alpha=0.3, linewidth=0.8, color=GRID)
    ax.set_axisbelow(True)


def main():
    f1lab.enable_cache()

    print("[1/3] Startplatz gegen Runde-1-Position, ganzer Cache (VORGEHEN "
         "1-2, keine Telemetrie noetig) ...")
    inv = f1lab.cached_sessions()
    rennen = inv[inv["ident"] == "R"][["season", "event"]].drop_duplicates()

    alle = []
    saison_diffs = []
    for _, r in rennen.iterrows():
        saison, gp = int(r["season"]), r["event"]
        try:
            ses = f1lab.load(saison, gp, "R", telemetry=False)
            m = f1lab.grid_lap1_positions(ses)
        except Exception:
            continue
        if m.empty:
            continue
        m["parity"] = np.where(m["grid"].astype(int) % 2 == 0,
                               "gerade", "ungerade")
        m["gp"] = gp
        m["season"] = saison
        alle.append(m)
        if gp in KONSISTENZ_STRECKEN:
            ung = m.loc[m["parity"] == "ungerade", "gewinn"]
            ger = m.loc[m["parity"] == "gerade", "gewinn"]
            saison_diffs.append({"gp": gp, "season": saison,
                                 "diff": ung.mean() - ger.mean()})

    df = pd.concat(alle, ignore_index=True)
    print(f"      {len(df)} Starts (ohne Boxenstarts) ueber "
         f"{df[['season', 'gp']].drop_duplicates().shape[0]} Rennen, "
         f"{df['gp'].nunique()} Strecken")

    print(f"\n[2/3] t-Test je Strecke mit mindestens {MIN_SAISONS} Saisons "
         "(VORGEHEN 3) ...")
    zeilen = []
    for gp, g in df.groupby("gp"):
        if g["season"].nunique() < MIN_SAISONS:
            continue
        ung = g.loc[g["parity"] == "ungerade", "gewinn"]
        ger = g.loc[g["parity"] == "gerade", "gewinn"]
        t, p = ttest_ind(ung, ger)
        zeilen.append({"gp": gp, "saisons": g["season"].nunique(),
                       "n": len(g), "diff": round(ung.mean() - ger.mean(), 2),
                       "p": round(p, 4)})
    ergebnisse = pd.DataFrame(zeilen).sort_values("p")
    print(ergebnisse.to_string(index=False))

    signifikant = ergebnisse[ergebnisse["p"] < 0.05]
    bonferroni = 0.05 / len(ergebnisse)
    print(f"\n      {len(signifikant)}/{len(ergebnisse)} Strecken mit p<0.05 "
         f"(Zufallserwartung: {0.05 * len(ergebnisse):.1f})")
    print(f"      Bonferroni-Schwelle bei {len(ergebnisse)} Tests: "
         f"{bonferroni:.4f} - ueberlebt: "
         f"{(ergebnisse['p'] < bonferroni).sum()}/{len(ergebnisse)}")

    print("\n[3/3] Saison-Konsistenz der fuenf auffaelligsten Strecken "
         "(AUSBAUSTUFE) ...")
    sd = pd.DataFrame(saison_diffs)
    for gp in KONSISTENZ_STRECKEN:
        werte = sd.loc[sd["gp"] == gp, "diff"]
        gleiche_richtung = (werte > 0).all() or (werte < 0).all()
        print(f"      {gp:28s} {len(werte)} Saisons, gleiche Richtung: "
             f"{gleiche_richtung}  ({[round(v, 2) for v in werte]})")

    print("\nGrafik ...")
    fig, ax = plt.subplots(2, 1, figsize=(13, 13))
    zeichne_strecken(ax[0], ergebnisse)
    zeichne_konsistenz(ax[1], sd)
    fig.suptitle("Startplatz-Paritaet: hat die Startseite einen echten "
                "Effekt?", x=0.09, ha="left", fontsize=16, color=FG, y=0.995)
    plt.tight_layout()
    path = OUT / "startplatz_paritaet.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"\n      -> {path}")


if __name__ == "__main__":
    main()
