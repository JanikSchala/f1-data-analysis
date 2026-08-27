"""wird der sport zuverlaessiger - technischer ausfall gegen unfall, ueber die vier regelaeren seit 1994"""
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
from scipy.stats import pearsonr

import f1lab
from f1lab.design import FG, GRID, MUTED, SERIEN, matplotlib_stil

warnings.filterwarnings("ignore")
fastf1.set_log_level("ERROR")

OUT = Path(__file__).parent / "out"
OUT.mkdir(exist_ok=True)

ERSTE_SAISON = 1994    # dieselbe grenze wie P46 - Ergast/jolpica davor duenn strukturiert
LETZTE_SAISON = 2026
GRANULAR_ENDE = 2022   # ab 2023 meldet Ergast/jolpica die meisten Ausfaelle nur
                       # noch generisch als "Retired" statt mit konkretem Grund
                       # (Motor/Getriebe/Unfall/...) - siehe VORGEHEN 1 unten.
                       # technisch/unfall sind nur bis hierhin sauber trennbar.

# dieselben vier regelaeren wie P46, fuer direkte vergleichbarkeit der beiden projekte.
ERAS = [
    (1994, 2008, "Refueling-Aera"),
    (2009, 2013, "Kein Refueling, V8"),
    (2014, 2021, "V6-Hybrid-Turbo"),
    (2022, LETZTE_SAISON, "Ground Effect"),
]

# unfall-/ausschluss-status per schluesselwort erkannt, alles andere gilt als
# technischer ausfall (motor/getriebe/elektrik/hydraulik/... - die liste
# moeglicher teileausfaelle ueber 33 jahre ist zu lang fuer eine feste
# positivliste, deshalb als default-fall statt enumeriert).
UNFALL_SCHLUESSEL = ("accident", "collision", "spun off", "damage")
AUSSCHLUSS_SCHLUESSEL = ("disqualified", "excluded", "did not qualify",
                        "did not prequalify", "did not start", "withdrew")


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


def alle_ergebnisse(erg: Ergast, saison: int) -> pd.DataFrame:
    """paginiert vollstaendig durch eine saison. der server deckelt eine
    einzelne antwort auf rund 100 ergebniszeilen - offset zaehlt zeilen,
    nicht rennen, ein einzelnes rennen kann also mitten in der antwort
    ueber zwei seiten verteilt sein. jede zeile wird deshalb einzeln mit
    ihrer eigenen runde getaggt statt ganze content-bloecke anzunehmen."""
    off, total, teile = 0, None, []
    while total is None or off < total:
        res = _mit_wiederholung(erg.get_race_results, season=saison,
                                limit=100, offset=off)
        if res is None or not res.content:
            break
        total = res.total_results
        for beschreibung, df in zip(res.description.itertuples(), res.content):
            if df.empty:
                continue
            teil = df[["driverId", "status"]].copy()
            teil["round"] = int(beschreibung.round)
            teile.append(teil)
        off += sum(len(df) for df in res.content)
        time.sleep(0.2)
    if not teile:
        return pd.DataFrame()
    ges = pd.concat(teile, ignore_index=True)
    ges["season"] = saison
    return ges


def sammle_saisons(erg: Ergast, erste: int, letzte: int) -> pd.DataFrame:
    rows = [alle_ergebnisse(erg, s) for s in range(erste, letzte + 1)]
    return pd.concat([r for r in rows if not r.empty], ignore_index=True)


def kategorie(status: str) -> str:
    s = str(status).lower()
    if s == "finished" or s.startswith("+") or s == "lapped":
        return "gewertet"
    if any(k in s for k in AUSSCHLUSS_SCHLUESSEL):
        return "ausschluss"
    if any(k in s for k in UNFALL_SCHLUESSEL):
        return "unfall"
    return "technisch"


def era_von(saison: int) -> str:
    for start, ende, name in ERAS:
        if start <= saison <= ende:
            return name
    return "unbekannt"


def zeichne_verlauf(ax, verlauf: pd.DataFrame) -> None:
    """dritte linie (generisch) macht den Bruch in der Datenqualitaet ab 2023
    sichtbar statt ihn zu verstecken - technisch/unfall sind erst ab dort
    nicht mehr sauber trennbar, weil "Retired" ohne konkreten Grund den
    grossteil der technisch-linie ab da ausmacht (siehe VORGEHEN 1)."""
    je_saison = (verlauf.groupby("season")[["technisch", "unfall", "generisch"]]
                .mean() * 100)
    ax.plot(je_saison.index, je_saison["technisch"], color=SERIEN[0], lw=1.6,
           marker="o", ms=3.5, label="Technischer Ausfall")
    ax.plot(je_saison.index, je_saison["unfall"], color=SERIEN[1], lw=1.6,
           marker="o", ms=3.5, label="Unfall")
    ax.plot(je_saison.index, je_saison["generisch"], color=MUTED, lw=1.4,
           ls="--", marker="o", ms=3, label="davon nur \"Retired\" (kein Grund)")
    ax.axvline(GRANULAR_ENDE + 0.5, color=MUTED, lw=1, ls=":")
    ax.text(GRANULAR_ENDE + 0.7, 20, " ab hier Grund\n meist unbekannt",
           fontsize=8, color=MUTED, va="top")
    ax.set_ylabel("Anteil an allen Fahrer-Rennen [%]")
    ax.set_xlabel("Saison")
    ax.set_title("Ausfallquote je Saison", loc="left", color=FG, fontsize=13,
                pad=10)
    ax.legend(loc="upper right", frameon=False, labelcolor=FG, fontsize=8.5)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(alpha=0.3, linewidth=0.8, color=GRID)
    ax.set_axisbelow(True)


def zeichne_eras(ax, verlauf: pd.DataFrame) -> None:
    """nur saisons bis GRANULAR_ENDE - 2023-2026 wuerden die "Ground Effect"-
    saeule sonst mit ueberwiegend unbekannten Ausfallgruenden verwaessern."""
    granular = verlauf[verlauf["season"] <= GRANULAR_ENDE]
    je_era = granular.groupby("era")[["technisch", "unfall"]].mean() * 100
    je_era = je_era.reindex([e[2] for e in ERAS]).dropna()
    x = range(len(je_era))
    w = 0.35
    ax.bar([i - w / 2 for i in x], je_era["technisch"], width=w,
          color=SERIEN[0], label="Technisch")
    ax.bar([i + w / 2 for i in x], je_era["unfall"], width=w,
          color=SERIEN[1], label="Unfall")
    labels = [f"{n}\n(nur {GRANULAR_ENDE})" if n == "Ground Effect" else n
             for n in je_era.index]
    ax.set_xticks(list(x), labels)
    ax.set_ylabel("Ausfallquote [%]")
    ax.set_title(f"Nach Regelaera (bis {GRANULAR_ENDE})", loc="left", color=FG,
                fontsize=13, pad=10)
    ax.tick_params(axis="x", labelrotation=12)
    ax.legend(loc="upper right", frameon=False, labelcolor=FG, fontsize=9)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="y", alpha=0.3, linewidth=0.8, color=GRID)
    ax.set_axisbelow(True)


def main():
    f1lab.enable_cache()
    plt.rcParams.update(matplotlib_stil())
    erg = Ergast(result_type="pandas", auto_cast=True)

    print(f"[1/4] Rennergebnisse {ERSTE_SAISON}-{LETZTE_SAISON} laden, je "
         "Saison vollstaendig paginiert (VORGEHEN 1) ...")
    daten = sammle_saisons(erg, ERSTE_SAISON, LETZTE_SAISON)
    daten["kategorie"] = daten["status"].apply(kategorie)
    daten = daten[daten["kategorie"] != "ausschluss"].copy()
    daten["technisch"] = daten["kategorie"] == "technisch"
    daten["unfall"] = daten["kategorie"] == "unfall"
    daten["generisch"] = daten["status"] == "Retired"
    daten["era"] = daten["season"].apply(era_von)
    print(f"      {len(daten)} Fahrer-Rennen ueber {daten['season'].nunique()} "
         "Saisons (Disqualifikationen/Nichtantritte ausgeschlossen)")

    print("\n[2/4] Datenqualitaet pruefen: wie oft steht nur 'Retired' ohne "
         "konkreten Grund da? ...")
    je_saison_generisch = (daten[daten["technisch"]].groupby("season")["generisch"]
                           .mean())
    print(f"      Anteil generisch an allen technischen Ausfaellen, letzte "
         f"10 Saisons:\n{je_saison_generisch.tail(10).round(3).to_string()}")
    if je_saison_generisch.loc[GRANULAR_ENDE + 1:].min() > 0.9:
        print(f"      -> ab {GRANULAR_ENDE + 1} praktisch nur noch generisch "
             f"(Ergast/jolpica liefert seither meist keinen Ausfallgrund "
             f"mehr) - technisch/unfall unten deshalb nur bis "
             f"{GRANULAR_ENDE} verglichen.")

    granular = daten[daten["season"] <= GRANULAR_ENDE]
    print(f"\n[3/4] Trendtest {ERSTE_SAISON}-{GRANULAR_ENDE}, je Fahrer-Rennen "
         "(VORGEHEN 2) ...")
    seasons = granular["season"].to_numpy(dtype=float)
    r_tech, p_tech = pearsonr(seasons, granular["technisch"].to_numpy(dtype=float))
    r_unfall, p_unfall = pearsonr(seasons, granular["unfall"].to_numpy(dtype=float))
    print(f"      Technisch: r={r_tech:.3f}, p={p_tech:.2e}")
    print(f"      Unfall:    r={r_unfall:.3f}, p={p_unfall:.2e}")

    print("\n[4/4] AUSBAUSTUFE: Aufschluesselung nach Regelaera ...")
    for start, ende, name in ERAS:
        teil = granular[granular["era"] == name]
        if teil.empty:
            continue
        bis = min(ende, GRANULAR_ENDE)
        print(f"      {name} ({start}-{bis}): technisch {teil['technisch'].mean():.1%}, "
             f"unfall {teil['unfall'].mean():.1%} (n={len(teil)})")

    print("\nGrafik ...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.5),
                                   gridspec_kw={"width_ratios": [3, 2]})
    zeichne_verlauf(ax1, daten)
    zeichne_eras(ax2, daten)
    fig.suptitle(f"Wird der Sport zuverlaessiger? {ERSTE_SAISON}-{LETZTE_SAISON}",
                x=0.06, ha="left", fontsize=15, color=FG, y=1.02)
    plt.tight_layout()
    path = OUT / "zuverlaessigkeit.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"      -> {path}")


if __name__ == "__main__":
    main()
