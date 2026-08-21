"""parst race-control-meldungen zu strafen und track-limit-verstößen und prüft sie gegen die lap-daten"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")                      # kein fenster, nur dateien

import fastf1
import matplotlib.pyplot as plt
import pandas as pd

import f1lab
from f1lab.design import FG, GRID, MUTED, SERIEN, matplotlib_stil

warnings.filterwarnings("ignore")
fastf1.set_log_level("ERROR")   # saison-scan lädt 24 sessions, das wäre sonst sehr geschwätzig

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
    """track-limit-meldungen je strecke und kurve."""
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
    """strecken-ranking."""
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
    """blaue flaggen je fahrer, aus Flag/RacingNumber statt regex."""
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
    """kurven-aufschlüsselung der strecke mit den meisten meldungen."""
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
