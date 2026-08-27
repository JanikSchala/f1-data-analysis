"""wie oft wird aus der pole tatsaechlich ein sieg, und hat sich das ueber die vier grossen regelaeren seit 1994 veraendert"""
from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")                      # kein Fenster, nur Dateien

import fastf1
import matplotlib.pyplot as plt
import pandas as pd
from fastf1.ergast import Ergast
from scipy.stats import binomtest, pearsonr

import f1lab
from f1lab.design import FG, GRID, MUTED, SERIEN, matplotlib_stil

warnings.filterwarnings("ignore")
fastf1.set_log_level("ERROR")

OUT = Path(__file__).parent / "out"
OUT.mkdir(exist_ok=True)

ERSTE_SAISON = 1994    # Ergast/jolpica hat davor keine strukturierten Startplatz-Daten
LETZTE_SAISON = 2026

# vier grosse reglement-aeren, grobe, allgemein anerkannte grenzen - nicht
# jede kleine regeländerung (z.b. 2017er aero-update) zaehlt als eigene aera.
ERAS = [
    (1994, 2008, "Refueling-Aera"),
    (2009, 2013, "Kein Refueling, V8"),
    (2014, 2021, "V6-Hybrid-Turbo"),
    (2022, LETZTE_SAISON, "Ground Effect"),
]


def _mit_wiederholung(fn, *args, versuche: int = 5, **kwargs):
    """Ergast/jolpica limitiert die Anfragerate bei Serienabfragen ueber viele Saisons."""
    for i in range(versuche):
        try:
            return fn(*args, **kwargs)
        except Exception:
            if i == versuche - 1:
                return None
            time.sleep(3 * (i + 1))
    return None


def era_von(saison: int) -> str:
    for start, ende, name in ERAS:
        if start <= saison <= ende:
            return name
    return "unbekannt"


def saison_verlauf(erg: Ergast) -> pd.DataFrame:
    """je saison EIN Aufruf statt eines je Rennen: get_race_results() mit
    grid_position=1 liefert direkt nur die Zeilen des Startplatz-1-Fahrers,
    ueber alle Runden der Saison hinweg (siehe VORGEHEN)."""
    rows = []
    for saison in range(ERSTE_SAISON, LETZTE_SAISON + 1):
        res = _mit_wiederholung(erg.get_race_results, season=saison,
                                grid_position=1)
        if res is None or not res.content:
            continue
        for beschreibung, df in zip(res.description.itertuples(), res.content):
            if df.empty:
                continue
            r = df.iloc[0]
            rows.append({
                "season": saison, "round": int(beschreibung.round),
                "race": beschreibung.raceName,
                "sieg": bool(r["position"] == 1),
                "dnf": not str(r["status"]).startswith(("Finished", "+")),
            })
        time.sleep(0.2)
    return pd.DataFrame(rows)


def zeichne_saisonverlauf(ax, verlauf: pd.DataFrame) -> None:
    je_saison = verlauf.groupby("season")["sieg"].mean() * 100
    ax.plot(je_saison.index, je_saison.to_numpy(), color=SERIEN[0], lw=1.6,
           marker="o", ms=3.5)
    for start, _ende, _name in ERAS[1:]:
        ax.axvline(start - 0.5, color=MUTED, lw=1, ls=":")
    ax.set_ylabel("Pole -> Sieg [%]")
    ax.set_xlabel("Saison")
    ax.set_title("Pole-to-Win-Konversionsrate je Saison", loc="left",
                color=FG, fontsize=13, pad=10)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(alpha=0.3, linewidth=0.8, color=GRID)
    ax.set_axisbelow(True)


def zeichne_eras(ax, verlauf: pd.DataFrame) -> None:
    je_era = verlauf.groupby("era")["sieg"].agg(["mean", "count"])
    je_era = je_era.reindex([e[2] for e in ERAS])
    quote = je_era["mean"] * 100
    farben = [SERIEN[0] if v == quote.max() else MUTED for v in quote]
    ax.bar(quote.index, quote.to_numpy(), color=farben, width=0.6)
    for i, (_name, row) in enumerate(je_era.iterrows()):
        ax.text(i, row["mean"] * 100 + 1.5, f"n={int(row['count'])}",
               ha="center", color=MUTED, fontsize=9)
    ax.set_ylabel("Pole -> Sieg [%]")
    ax.set_ylim(0, max(quote.max() + 10, 60))
    ax.set_title("Nach Regelaera", loc="left", color=FG, fontsize=13, pad=10)
    ax.tick_params(axis="x", labelrotation=12)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="y", alpha=0.3, linewidth=0.8, color=GRID)
    ax.set_axisbelow(True)


def main():
    f1lab.enable_cache()
    plt.rcParams.update(matplotlib_stil())
    erg = Ergast(result_type="pandas", auto_cast=True)

    print(f"[1/3] Saisonverlauf {ERSTE_SAISON}-{LETZTE_SAISON} laden "
         "(je Saison ein Aufruf mit grid_position=1, VORGEHEN 1) ...")
    verlauf = saison_verlauf(erg)
    verlauf["era"] = verlauf["season"].apply(era_von)
    print(f"      {len(verlauf)} Rennen mit bekanntem Startplatz-1-Fahrer "
         f"ueber {verlauf['season'].nunique()} Saisons")

    print("\n[2/3] Gesamtquote und Statistik (VORGEHEN 2) ...")
    n = len(verlauf)
    siege = int(verlauf["sieg"].sum())
    test = binomtest(siege, n, 0.5)
    print(f"      {siege}/{n} = {siege / n:.1%} - Binomialtest gegen 50%: "
         f"p={test.pvalue:.2e}")

    print("\n[3/3] Aufschluesselung nach Regelaera (VORGEHEN 3-4) ...")
    for start, ende, name in ERAS:
        teil = verlauf[verlauf["era"] == name]
        if teil.empty:
            continue
        quote = teil["sieg"].mean()
        print(f"      {name} ({start}-{ende}): {int(teil['sieg'].sum())}/"
             f"{len(teil)} = {quote:.1%}")

    # der steigende trend ueber die vier aeren koennte zufall sein - punkt-
    # biseriale korrelation (era als ordinale 1-4-stufe gegen das binaere
    # sieg-ergebnis je rennen, nicht nur die vier aggregierten quoten) prueft
    # das direkt auf ebene der einzelnen rennen.
    era_index = {name: i + 1 for i, (_s, _e, name) in enumerate(ERAS)}
    era_ordinal = verlauf["era"].map(era_index).to_numpy(dtype=float)
    trend = pearsonr(era_ordinal, verlauf["sieg"].to_numpy(dtype=float))
    print(f"\n      Trendtest (Aera-Stufe gegen Sieg, je Rennen): "
         f"r={trend.statistic:.3f}, p={trend.pvalue:.2e}")

    # AUSBAUSTUFE: haengt die konversionsrate an der ausfallquote des
    # start-p1-fahrers, oder verliert er die fuehrung meist im rennen selbst?
    dnf_quote = verlauf["dnf"].mean()
    verloren_ohne_dnf = verlauf[~verlauf["dnf"] & ~verlauf["sieg"]]
    print(f"\n      AUSBAUSTUFE: Start-P1 faellt in {dnf_quote:.1%} der "
         f"Rennen aus. In {len(verloren_ohne_dnf)}/{n} Rennen "
         f"({len(verloren_ohne_dnf) / n:.1%}) beendet er das Rennen, "
         "gewinnt aber nicht - verliert die Fuehrung also im Rennen selbst, "
         "nicht durch einen Ausfall.")

    print("\nGrafik ...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.5),
                                   gridspec_kw={"width_ratios": [3, 2]})
    zeichne_saisonverlauf(ax1, verlauf)
    zeichne_eras(ax2, verlauf)
    fig.suptitle("Pole-to-Win-Konversionsrate seit 1994", x=0.06, ha="left",
                fontsize=16, color=FG, y=1.02)
    plt.tight_layout()
    path = OUT / "pole_to_win.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"      -> {path}")


if __name__ == "__main__":
    main()
