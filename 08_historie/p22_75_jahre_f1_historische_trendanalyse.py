"""analysiert 75 jahre f1-saisons aus der ergast-api auf dominanzphasen, dnf-rate, nationenkonzentration und streckenwandel"""
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

import f1lab
from f1lab.design import FG, GRID, MUTED, SERIEN, matplotlib_stil

warnings.filterwarnings("ignore")
fastf1.set_log_level("ERROR")

OUT = Path(__file__).parent / "out"
OUT.mkdir(exist_ok=True)

ERSTE_KONSTRUKTEURS_WM = 1958
DNF_AB = 1970
STRECKEN = {"monza": "Monza", "spa": "Spa-Francorchamps", "silverstone": "Silverstone"}
SCHRITT_STRECKEN = 4   # nur jedes 4. Jahr abfragen, sonst zu viele Calls

plt.rcParams.update(matplotlib_stil())



def dominanz_und_nationen(erg: Ergast) -> pd.DataFrame:
    """nutzt dieselben Constructor-Standings-Calls auch für den Nationen-Trend, keine zusätzlichen Anfragen nötig."""
    rows = []
    for year in range(ERSTE_KONSTRUKTEURS_WM, 2025):
        cs = f1lab.ergast_retry(erg.get_constructor_standings, season=year,
                                leer_bei_fehlschlag=True)
        if cs is None or not cs.content or cs.content[0].empty:
            continue
        cs = cs.content[0]
        total = cs["points"].sum()
        if total == 0:
            continue
        nat_anteil = cs["constructorNationality"].value_counts(
            normalize=True).iloc[0]
        rows.append({
            "season": year, "champion": cs.iloc[0]["constructorName"],
            "anteil": cs.iloc[0]["points"] / total, "teams": len(cs),
            "haeufigste_nation": cs["constructorNationality"].mode().iloc[0],
            "nation_anteil": nat_anteil,
        })
        time.sleep(0.15)
    return pd.DataFrame(rows)


def dnf_rate(erg: Ergast) -> pd.DataFrame:
    rows = []
    for year in range(DNF_AB, 2025):
        st = f1lab.ergast_retry(erg.get_finishing_status, season=year,
                                leer_bei_fehlschlag=True)
        if st is None or st.empty:
            continue
        total = st["count"].sum()
        finished = st.loc[st["status"].str.contains(
            "Finished|Lap", na=False), "count"].sum()
        rows.append({"season": year, "dnf_rate": 1 - finished / total})
        time.sleep(0.15)
    return pd.DataFrame(rows)


def kalendergroesse(erg: Ergast) -> pd.DataFrame:
    rows = []
    for year in range(ERSTE_KONSTRUKTEURS_WM, 2025):
        sched = f1lab.ergast_retry(erg.get_race_schedule, season=year,
                                   leer_bei_fehlschlag=True)
        if sched is None or sched.empty:
            continue
        rows.append({"season": year, "rennen": len(sched),
                    "laender": sched["country"].nunique()})
        time.sleep(0.15)
    return pd.DataFrame(rows)


def rundenzeit_je_strecke(erg: Ergast) -> pd.DataFrame:
    """Rundenzeit des Siegers, nur alle SCHRITT_STRECKEN Jahre um die Zahl der Calls zu begrenzen."""
    rows = []
    for circuit_id, name in STRECKEN.items():
        for year in range(1950, 2025, SCHRITT_STRECKEN):
            res = f1lab.ergast_retry(erg.get_race_results, season=year,
                                     circuit=circuit_id,
                                     leer_bei_fehlschlag=True)
            if res is None or not res.content or res.content[0].empty:
                continue
            df = res.content[0]
            sieger = df[df["position"] == 1]
            if sieger.empty:
                continue
            sieger = sieger.iloc[0]
            if pd.isna(sieger["totalRaceTime"]) or not sieger["laps"]:
                continue
            sekunden = sieger["totalRaceTime"].total_seconds() / sieger["laps"]
            # Ergasts totalRaceTime ist für einzelne sehr alte Rennen kaputt.
            # 30s ist auf keiner F1-Strecke je eine echte Rundenzeit.
            if sekunden < 30:
                continue
            rows.append({"strecke": name, "season": year, "rundenzeit_s": sekunden})
            time.sleep(0.15)
    return pd.DataFrame(rows)


def zeichne_dominanz(ax, dom: pd.DataFrame) -> None:
    ax.bar(dom["season"], dom["anteil"] * 100, color=SERIEN[0], width=0.8)
    ax.set_ylabel("Punkteanteil Champion [%]")
    ax.set_title("Dominanzphasen: Konstrukteurs-WM", loc="left", color=FG,
                fontsize=12, pad=10)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="y", alpha=0.3, linewidth=0.8, color=GRID)
    ax.set_axisbelow(True)


def zeichne_dnf(ax, dnf: pd.DataFrame) -> None:
    ax.plot(dnf["season"], dnf["dnf_rate"] * 100, color=SERIEN[1], lw=1.8)
    ax.set_ylabel("DNF-Rate [%]")
    ax.set_title("Zuverlaessigkeit ueber die Zeit", loc="left", color=FG,
                fontsize=12, pad=10)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(alpha=0.3, linewidth=0.8, color=GRID)
    ax.set_axisbelow(True)


def zeichne_nationen(ax, dom: pd.DataFrame) -> None:
    ax.plot(dom["season"], dom["nation_anteil"] * 100, color=SERIEN[2], lw=1.8)
    ax.set_ylabel("Anteil haeufigste Nation [%]")
    ax.set_title("Nationen: Konzentration im Konstrukteursfeld", loc="left",
                color=FG, fontsize=12, pad=10)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(alpha=0.3, linewidth=0.8, color=GRID)
    ax.set_axisbelow(True)


def zeichne_kalender(ax, kal: pd.DataFrame) -> None:
    ax.plot(kal["season"], kal["rennen"], color=SERIEN[0], lw=1.8, label="Rennen")
    ax.plot(kal["season"], kal["laender"], color=MUTED, lw=1.4, ls="--",
           label="Laender")
    ax.set_ylabel("Anzahl")
    ax.set_title("Streckenwandel: Kalendergroesse", loc="left", color=FG,
                fontsize=12, pad=10)
    ax.legend(loc="upper left", frameon=False, labelcolor=FG, fontsize=9)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(alpha=0.3, linewidth=0.8, color=GRID)
    ax.set_axisbelow(True)


def zeichne_rundenzeiten(ax, rz: pd.DataFrame) -> None:
    for i, (strecke, g) in enumerate(rz.groupby("strecke")):
        g = g.sort_values("season")
        ax.plot(g["season"], g["rundenzeit_s"], marker="o", ms=4, lw=1.6,
               color=SERIEN[i % len(SERIEN)], label=strecke)
    ax.axvline(1979, color=MUTED, lw=1, ls=":")
    ax.text(1979, ax.get_ylim()[1], " Spa fehlt 1971-82,\n kehrt verkuerzt zurueck",
           fontsize=8, color=MUTED, va="top")
    ax.set_ylabel("Rundenzeit Sieger [s]")
    ax.set_xlabel("Saison")
    ax.set_title("AUSBAUSTUFE: Rundenzeit - Vorsicht bei Streckenaenderungen",
                loc="left", color=FG, fontsize=12, pad=10)
    ax.legend(loc="upper right", frameon=False, labelcolor=FG, fontsize=9)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(alpha=0.3, linewidth=0.8, color=GRID)
    ax.set_axisbelow(True)


def main():
    f1lab.enable_cache()
    erg = Ergast(result_type="pandas", auto_cast=True)

    print(f"[1/4] Konstrukteurs-Standings {ERSTE_KONSTRUKTEURS_WM}-2024 "
         f"(VORGEHEN 1-2, Nationen) ...")
    dom = dominanz_und_nationen(erg)
    print(f"      {len(dom)} Saisons geladen. Dominanteste Titel:")
    print(dom.nlargest(5, "anteil")[["season", "champion", "anteil"]]
         .to_string(index=False))
    print("\n      Ausgeglichenste Titel:")
    print(dom.nsmallest(5, "anteil")[["season", "champion", "anteil"]]
         .to_string(index=False))

    print(f"\n[2/4] DNF-Rate {DNF_AB}-2024 (VORGEHEN 3) ...")
    dnf = dnf_rate(erg)
    print(f"      {len(dnf)} Saisons. Hoechste/niedrigste DNF-Rate:")
    print(f"      {dnf.loc[dnf['dnf_rate'].idxmax(), 'season']}: "
         f"{dnf['dnf_rate'].max():.1%}  /  "
         f"{dnf.loc[dnf['dnf_rate'].idxmin(), 'season']}: "
         f"{dnf['dnf_rate'].min():.1%}")

    print(f"\n[3/4] Kalendergroesse {ERSTE_KONSTRUKTEURS_WM}-2024 "
         f"(VORGEHEN 4, Streckenwandel) ...")
    kal = kalendergroesse(erg)
    print(f"      {kal.iloc[0]['season']}: {kal.iloc[0]['rennen']} Rennen "
         f"-> {kal.iloc[-1]['season']}: {kal.iloc[-1]['rennen']} Rennen")

    print(f"\n[4/4] AUSBAUSTUFE: Rundenzeiten {', '.join(STRECKEN.values())} "
         f"(alle {SCHRITT_STRECKEN} Jahre) ...")
    rz = rundenzeit_je_strecke(erg)
    print(f"      {len(rz)} Datenpunkte")
    for strecke, g in rz.groupby("strecke"):
        g = g.sort_values("season")
        print(f"      {strecke}: {g.iloc[0]['season']} "
             f"{g.iloc[0]['rundenzeit_s']:.1f}s -> {g.iloc[-1]['season']} "
             f"{g.iloc[-1]['rundenzeit_s']:.1f}s")

    print("\nGrafik ...")
    fig, ax = plt.subplots(2, 2, figsize=(15, 10))
    zeichne_dominanz(ax[0, 0], dom)
    zeichne_dnf(ax[0, 1], dnf)
    zeichne_nationen(ax[1, 0], dom)
    zeichne_kalender(ax[1, 1], kal)
    fig.suptitle("75 Jahre F1: Historische Trends", x=0.09, ha="left",
                fontsize=16, color=FG, y=1.0)
    plt.tight_layout()
    path = OUT / "historische_trends.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"      -> {path}")

    fig2, ax2 = plt.subplots(figsize=(13, 6))
    zeichne_rundenzeiten(ax2, rz)
    plt.tight_layout()
    path2 = OUT / "rundenzeiten_ausbaustufe.png"
    fig2.savefig(path2, dpi=130, bbox_inches="tight")
    plt.close(fig2)
    print(f"      -> {path2}")


if __name__ == "__main__":
    main()
