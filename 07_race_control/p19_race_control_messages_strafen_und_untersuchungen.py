"""
P19 - Race Control Messages: Strafen und Untersuchungen automatisch auswerten
=============================================================================

Die Textmeldungen der Rennleitung parsen: Strafen, Verwarnungen, Track-Limits, gestrichene Runden.

Kategorie:   Race Control & Regeln
Niveau:      Fortgeschritten
Aufwand:     3-4 h
Schwerpunkt: Datenanalyse, Engineering

WARUM DAS LOHNT
Textparsing plus Regex auf einem echten, unsauberen Feed. Zeigt, dass du auch mit unstrukturierten Daten arbeitest, nicht nur mit fertigen DataFrames.

VORGEHEN
  1. race_control_messages laden und Kategorien inspizieren
  2. Regex fuer Fahrer-Nummer, Vergehen und Strafmass bauen
  3. Track-Limit-Verstoesse je Fahrer und Kurve zaehlen
  4. Mit Deleted/DeletedReason aus den Laps gegenpruefen

GENUTZTE FASTF1-BAUSTEINE
  - Session.race_control_messages
  - Laps Deleted/DeletedReason
  - re

AUSBAUSTUFE  [umgesetzt]
Baue daraus einen Saison-Report: welche Strecke produziert die meisten
Track-Limit-Verstoesse, und an welcher Kurve genau.

VORGEHEN 2s Strafen-Regex traf in der urspruenglichen Fassung 0 von 6
echten Strafmeldungen in Oesterreich 2024 R (getestet, nicht vermutet) - der
Grund ist keine Tippgenauigkeit, sondern eine falsche Reihenfolge-Annahme.
Die Vorlage erwartet "CAR 14 (ALO) ... 10 SECOND", die echte FIA-Formulierung
ist umgekehrt: "10 SECOND TIME PENALTY FOR CAR 14 (ALO) - CAUSING A
COLLISION". Ausserdem heisst der Boxenstopp-Strafentyp in echten Meldungen
"STOP/GO PENALTY", nicht "STOP AND GO" wie im Vorschlag. Regex korrigiert
und gegen alle Strafmeldungen der Saison 2024 geprueft: 49/49 Treffer.
"DRIVE THROUGH PENALTY" und "REPRIMAND" sind in der Formulierung an dieselbe
Struktur angelehnt, kamen 2024 aber kein einziges Mal vor - ungetestet,
weil es diese Saison keinen einzigen Beleg dafuer gibt.

Der TRACKLIM-Regex aus der Vorlage war dagegen von Anfang an korrekt (16/16
in Oesterreich).

VORGEHEN 4 (Gegenpruefung mit Deleted/DeletedReason) findet eine echte
Diskrepanz statt nur eine Bestaetigung - und dabei zaehlt die Runden-
nummer aus der Meldung selbst, nicht die Runde, in der die Meldung gepostet
wurde (die Loeschung wird oft 1-2 Runden spaeter verbucht, ein erster
Regex-Entwurf verwechselte das und produzierte 16 Fehltreffer statt der 2
echten). Von 16 Track-Limit-Meldungen in Oesterreich mit eindeutiger
Rundennummer im Text finden sich nur 14 als Deleted=True in den Lap-Daten.
Die zwei fehlenden sind Meldungen der Form "LAP DELETED" (nicht "TIME ...
DELETED") fuer Rundennummer 1 (Ricciardo, seine allererste Runde) und Runde
51 (Norris, explizit als Boxenrunde markiert "(PIT)") - beides Runden ohne
eine im Timing gefuehrte Rundenzeit, auf die sich ein Deleted-Flag setzen
liesse. Eine dritte Kategorie, in Oesterreich nicht vertreten aber in
mehreren anderen 2024er Rennen ("(NEXT LAP)" statt einer Rundennummer),
nennt gar keine explizite Runde und bleibt aus der Gegenpruefung
folgerichtig aussen vor. FastF1s Deleted-Spalte ist damit fuer
Track-Limit-Auswertungen leicht unvollstaendig; der Text aus
race_control_messages ist hier die vollstaendigere Quelle.

AUSBAUSTUFE als Saison-Scan (P05/P18-Muster) ueber alle 24 Rennen 2024:
Austin (COTA) fuehrt klar mit 43 Track-Limit-Meldungen vor Kanada (31) und
Abu Dhabi (28) - und mehr als jede zweite COTA-Meldung (17 von 43) betrifft
eine einzige Kurve, Turn 12, bekannt fuer ihre breiten Kerbs am Ende der
Zielgeraden-Kombination.

Die Regex-Parser und die Deleted-Gegenpruefung stecken seit der
App-Integration in f1lab.session (parse_penalties()/parse_track_limits()/
track_limit_crosscheck()) statt hier lokal (zweiter Konsument: die
Race-Control-Seite im Dashboard, auf einer beliebigen vom Nutzer gewaehlten
Session statt fest auf Oesterreich). Der Saison-Scan bleibt skriptlokal.

ZWEITE AUSBAUSTUFE: zwei bislang ungenutzte FastF1-Felder ergaenzt.

``race_control_messages["Flag"]``/``["RacingNumber"]`` waren im ganzen
Repository nie gelesen - VORGEHEN 2/3 parsen Flaggen-Ereignisse komplett per
Regex auf den Freitext, obwohl FastF1 Flaggenfarbe und Fahrzeugnummer schon
getrennt mitliefert. `f1lab.blue_flags()` nutzt das direkt: in Oesterreich
2024 R wurden 62 blaue Flaggen gezeigt, angefuehrt von Sargeant (15) -
blaue Flaggen sind aber kein Vergehen, sondern die Aufforderung, einen
Ueberrunder durchzulassen. Viele blaue Flaggen heissen "wurde oft
ueberrundet", nicht "hat oft gestoert".

``Laps.DeletedReason`` stand im urspruenglichen Docstring bereits als
"genutzter Baustein", wurde aber nie tatsaechlich gelesen - nur das
boolesche ``Deleted`` floss in VORGEHEN 4 ein. `f1lab.
deleted_reason_crosscheck()` dreht die bestehende Gegenpruefung um: statt
zu pruefen, ob jede Text-Meldung eine Deleted=True-Runde findet (das deckte
FastF1s unvollstaendige Deleted-Spalte auf, siehe oben), prueft sie, ob
jede Deleted=True-Runde mit Track-Limits-Grund auch eine passende
Race-Control-Meldung im Regex-Treffer findet - die andere Fehlerrichtung.
Fuer Oesterreich 2024 R: 0 fehlend, die Regex-Abdeckung war hier bereits
vollstaendig (49/49 bzw. 16/16 waren schon dokumentiert) - eine
Bestaetigung, kein neuer Fund, aber jetzt tatsaechlich in beide Richtungen
geprueft statt nur einer.
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
import pandas as pd

import f1lab
from f1lab.design import FG, GRID, MUTED, SERIEN, matplotlib_stil

warnings.filterwarnings("ignore")
fastf1.set_log_level("ERROR")   # Saison-Scan laedt 24 Sessions, siehe P05

OUT = Path(__file__).parent / "out"
OUT.mkdir(exist_ok=True)

SEASON, EVENT, IDENT = 2024, "Austria", "R"
SAISON_RENNEN = [
    "Bahrain", "Saudi Arabia", "Australia", "Japan", "China", "Miami",
    "Emilia Romagna", "Monaco", "Canada", "Spain", "Austria", "Britain",
    "Hungary", "Belgium", "Netherlands", "Italy", "Azerbaijan", "Singapore",
    "United States", "Mexico", "Brazil", "Las Vegas", "Qatar", "Abu Dhabi",
]

plt.rcParams.update(matplotlib_stil())


def saison_scan() -> pd.DataFrame:
    """AUSBAUSTUFE: Track-Limit-Meldungen je Strecke und Kurve."""
    zeilen = []
    for gp in SAISON_RENNEN:
        try:
            ses = f1lab.load(SEASON, gp, "R", telemetry=False, messages=True)
        except Exception:
            continue
        lim = f1lab.parse_track_limits(ses.race_control_messages)
        for r in lim.itertuples():
            zeilen.append({"gp": gp, "driver": r.driver, "turn": r.turn})
    return pd.DataFrame(zeilen)


def zeichne_strafen(ax, pen: pd.DataFrame) -> None:
    """VORGEHEN 2."""
    if pen.empty:
        ax.text(0.5, 0.5, "keine Strafen", ha="center", va="center", color=MUTED)
        ax.axis("off")
        return
    je_fahrer = pen.groupby("driver").size().sort_values()
    ax.barh(je_fahrer.index, je_fahrer.to_numpy(), color=SERIEN[1], height=0.6)
    ax.set_xlabel("Anzahl Strafen")
    ax.set_title(f"{EVENT} {SEASON} - Strafen je Fahrer ({len(pen)} gesamt)",
                loc="left", color=FG, fontsize=12, pad=10)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="x", alpha=0.3, linewidth=0.8, color=GRID)
    ax.set_axisbelow(True)


def zeichne_tracklimits_lokal(ax, lim: pd.DataFrame) -> None:
    """VORGEHEN 3."""
    je_kurve = lim.groupby("turn").size().sort_index()
    ax.bar([f"T{t}" for t in je_kurve.index], je_kurve.to_numpy(),
          color=SERIEN[0], width=0.6)
    ax.set_ylabel("Track-Limit-Meldungen")
    ax.set_title(f"{EVENT} {SEASON} - Track Limits je Kurve ({len(lim)} gesamt)",
                loc="left", color=FG, fontsize=12, pad=10)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="y", alpha=0.3, linewidth=0.8, color=GRID)
    ax.set_axisbelow(True)


def zeichne_saison_ranking(ax, scan: pd.DataFrame) -> None:
    """AUSBAUSTUFE, Teil 1: Strecken-Ranking."""
    je_strecke = scan.groupby("gp").size().sort_values(ascending=False).head(10)
    farben = [SERIEN[1] if i == 0 else MUTED for i in range(len(je_strecke))]
    ax.barh(je_strecke.index[::-1], je_strecke.to_numpy()[::-1],
          color=farben[::-1], height=0.6)
    ax.set_xlabel("Track-Limit-Meldungen")
    ax.set_title(f"Saison {SEASON}: die 10 Strecken mit den meisten "
                f"Meldungen", loc="left", color=FG, fontsize=12, pad=10)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="x", alpha=0.3, linewidth=0.8, color=GRID)
    ax.set_axisbelow(True)


def zeichne_blaue_flaggen(ax, bf: pd.DataFrame) -> None:
    """ZWEITE AUSBAUSTUFE: blaue Flaggen je Fahrer, aus Flag/RacingNumber."""
    if bf.empty:
        ax.text(0.5, 0.5, "keine blauen Flaggen", ha="center", va="center",
               color=MUTED)
        ax.axis("off")
        return
    je_fahrer = bf.groupby("driver").size().sort_values()
    ax.barh(je_fahrer.index, je_fahrer.to_numpy(), color=SERIEN[2], height=0.6)
    ax.set_xlabel("Blaue Flaggen")
    ax.set_title(f"{EVENT} {SEASON} - Blaue Flaggen je Fahrer ({len(bf)} "
                f"gesamt, aus Flag/RacingNumber statt Regex)", loc="left",
                color=FG, fontsize=12, pad=10)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="x", alpha=0.3, linewidth=0.8, color=GRID)
    ax.set_axisbelow(True)


def zeichne_top_strecke(ax, scan: pd.DataFrame, top_gp: str) -> None:
    """AUSBAUSTUFE, Teil 2: Kurven-Aufschluesselung der schlimmsten Strecke."""
    je_kurve = (scan[scan["gp"] == top_gp].groupby("turn").size()
               .sort_values(ascending=False))
    farben = [SERIEN[1] if i == 0 else MUTED for i in range(len(je_kurve))]
    ax.bar([f"T{t}" for t in je_kurve.index], je_kurve.to_numpy(), color=farben,
          width=0.6)
    anteil = 100 * je_kurve.iloc[0] / je_kurve.sum()
    ax.set_ylabel("Track-Limit-Meldungen")
    ax.set_title(f"{top_gp} {SEASON}: T{je_kurve.index[0]} allein "
                f"{anteil:.0f} % aller Meldungen", loc="left", color=FG,
                fontsize=12, pad=10)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="y", alpha=0.3, linewidth=0.8, color=GRID)
    ax.set_axisbelow(True)


def main():
    f1lab.enable_cache()

    print(f"[1/5] {EVENT} {SEASON} {IDENT} laden (VORGEHEN 1) ...")
    ses = f1lab.load(SEASON, EVENT, IDENT, telemetry=False, messages=True)
    rcm = ses.race_control_messages
    print(rcm["Category"].value_counts().to_string())

    print("\n[2/5] Strafen und Track Limits parsen (VORGEHEN 2-3) ...")
    pen = f1lab.parse_penalties(rcm)
    lim = f1lab.parse_track_limits(rcm)
    print(f"      {len(pen)} Strafen erkannt:")
    print(pen[["lap", "driver", "strafmass", "grund"]].to_string(index=False))
    print(f"\n      {len(lim)} Track-Limit-Meldungen, je Fahrer:")
    print(lim["driver"].value_counts().to_string())
    print("\n      je Kurve:")
    print(lim["turn"].value_counts().sort_index().to_string())

    print("\n[3/5] Gegenpruefung mit Laps.Deleted (VORGEHEN 4) ...")
    fehlend, deleted, n_mit_runde = f1lab.track_limit_crosscheck(ses, rcm)
    print(f"      {len(deleted)} Runden mit Deleted=True in den Lap-Daten "
         f"gegen {n_mit_runde} Text-Meldungen mit eindeutiger Rundennummer "
         f"({len(lim) - n_mit_runde} weitere ohne explizite Runde, "
         f"'(NEXT LAP)' o.ae. - siehe Docstring)")
    if not fehlend.empty:
        print(f"      Im Text erwaehnt, aber nicht als Deleted=True in den "
             f"Laps zu finden ({len(fehlend)}):")
        print(fehlend.to_string(index=False))

    print("\n[4/5] ZWEITE AUSBAUSTUFE: Flag/RacingNumber und DeletedReason ...")
    bf = f1lab.blue_flags(ses, rcm)
    print(f"      {len(bf)} blaue Flaggen, je Fahrer:")
    print(bf.groupby("driver").size().sort_values(ascending=False).to_string())

    umgekehrt_fehlend = f1lab.deleted_reason_crosscheck(ses, rcm)
    print(f"\n      Umgekehrte Gegenpruefung (DeletedReason -> Text-Meldung "
         f"vorhanden?): {len(umgekehrt_fehlend)} fehlend")
    if not umgekehrt_fehlend.empty:
        print(umgekehrt_fehlend.to_string(index=False))

    print(f"\n[5/5] AUSBAUSTUFE: Saison-Scan {SEASON} ({len(SAISON_RENNEN)} "
         f"Rennen) ...")
    scan = saison_scan()
    je_strecke = scan.groupby("gp").size().sort_values(ascending=False)
    print(je_strecke.head(10).to_string())
    top_gp = je_strecke.index[0]
    print(f"\n      {top_gp}: Kurven-Aufschluesselung")
    print(scan[scan["gp"] == top_gp]["turn"].value_counts().to_string())

    print("\nGrafik ...")
    fig, ax = plt.subplots(2, 2, figsize=(15, 10))
    zeichne_strafen(ax[0, 0], pen)
    zeichne_tracklimits_lokal(ax[0, 1], lim)
    zeichne_saison_ranking(ax[1, 0], scan)
    zeichne_top_strecke(ax[1, 1], scan, top_gp)
    fig.suptitle("Race Control: Strafen und Track Limits", x=0.09, ha="left",
                fontsize=16, color=FG, y=1.0)
    plt.tight_layout()
    path = OUT / "race_control.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"\n      -> {path}")

    print("\nGrafik ZWEITE AUSBAUSTUFE ...")
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    zeichne_blaue_flaggen(ax2, bf)
    plt.tight_layout()
    path2 = OUT / "race_control_blaue_flaggen.png"
    fig2.savefig(path2, dpi=130, bbox_inches="tight")
    plt.close(fig2)
    print(f"      -> {path2}")


if __name__ == "__main__":
    main()
