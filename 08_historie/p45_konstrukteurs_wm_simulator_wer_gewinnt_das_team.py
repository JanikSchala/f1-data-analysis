"""simuliert konstrukteurs-wm-titelchancen für die restsaison per monte-carlo - dieselbe mechanik wie P21, aber auf teamebene mit zwei autos je event"""
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
import numpy as np
import pandas as pd
from fastf1.ergast import Ergast

import f1lab
from f1lab.design import FG, GRID, MUTED, SERIEN, matplotlib_stil

warnings.filterwarnings("ignore")
fastf1.set_log_level("ERROR")

OUT = Path(__file__).parent / "out"
OUT.mkdir(exist_ok=True)

YEAR = 2026
PTS_RENNEN = {1: 25, 2: 18, 3: 15, 4: 12, 5: 10, 6: 8, 7: 6, 8: 4, 9: 2, 10: 1}
PTS_SPRINT = {1: 8, 2: 7, 3: 6, 4: 5, 5: 4, 6: 3, 7: 2, 8: 1}
N_SIM = 50_000



def punkte_array(tabelle: dict) -> np.ndarray:
    arr = np.zeros(25)
    for platz, pkt in tabelle.items():
        arr[platz] = pkt
    return arr


def saison_verlauf(erg: Ergast, jahr: int, runden: int
                   ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """laedt alle bisher gefahrenen Rennen und Sprints, je Zeile ein Auto
    (nicht ein Team) - dieselbe Rohform wie P21, hier nach constructorId statt
    driverId gruppiert."""
    race_frames, sprint_frames = [], []
    for r in range(1, runden + 1):
        res = f1lab.ergast_retry(erg.get_race_results, season=jahr, round=r).content
        if res:
            race_frames.append(res[0])
        time.sleep(0.2)
        sp = f1lab.ergast_retry(erg.get_sprint_results, season=jahr, round=r).content
        if sp:
            sprint_frames.append(sp[0])
        time.sleep(0.2)
    races = pd.concat(race_frames, ignore_index=True)
    sprints = (pd.concat(sprint_frames, ignore_index=True)
              if sprint_frames else pd.DataFrame(columns=races.columns))
    races["dnf"] = ~races["status"].isin(["Finished", "Lapped"])
    return races, sprints


def monte_carlo_team(base_points: dict, teams: list[str],
                     pos_hist: dict, sprint_hist: dict, sprint_flags: list[bool],
                     dnf_rate: dict | None, rng: np.random.Generator) -> pd.DataFrame:
    """wie P21s monte_carlo(), aber ZWEI unabhaengige Positionszuege je Team
    und Event statt einem - ein Konstrukteur bringt zwei Autos an den Start,
    beide zaehlen in die Teampunkte. Die Positions-Stichprobe je Team ist ueber
    beide Autos gepoolt (keine Modellierung von 'Auto 1 vs. Auto 2'-Asymmetrie
    oder Team-interner Korrelation - dieselbe Vereinfachung wie P21s Verzicht
    auf eine Fahrer-Fahrer-Kopplung)."""
    pts_r, pts_s = punkte_array(PTS_RENNEN), punkte_array(PTS_SPRINT)
    totals = {t: np.full(N_SIM, base_points.get(t, 0.0)) for t in teams}

    for ist_sprint in sprint_flags:
        for t in teams:
            hist = pos_hist.get(t)
            if hist is not None and len(hist) > 0:
                for _auto in range(2):
                    pos = np.clip(rng.choice(hist, size=N_SIM).astype(int), 0, 24)
                    pts = pts_r[pos]
                    if dnf_rate is not None:
                        ausfall = rng.random(N_SIM) < dnf_rate.get(t, 0.0)
                        pts = np.where(ausfall, 0.0, pts)
                    totals[t] += pts
            if ist_sprint:
                shist = sprint_hist.get(t)
                if shist is not None and len(shist) > 0:
                    for _auto in range(2):
                        pos = np.clip(rng.choice(shist, size=N_SIM).astype(int), 0, 24)
                        pts = pts_s[pos]
                        if dnf_rate is not None:
                            ausfall = rng.random(N_SIM) < dnf_rate.get(t, 0.0)
                            pts = np.where(ausfall, 0.0, pts)
                        totals[t] += pts

    mat = np.vstack([totals[t] for t in teams])
    return pd.DataFrame(mat.T, columns=teams)


def zeichne_standings(ax, standings: pd.DataFrame, gefahren: int) -> None:
    top = standings.head(10).iloc[::-1]
    farben = [SERIEN[0] if i == len(top) - 1 else MUTED for i in range(len(top))]
    ax.barh(top["constructorName"], top["points"], color=farben, height=0.6)
    ax.set_xlabel("Punkte")
    ax.set_title(f"Konstrukteurs-WM {YEAR} nach {gefahren} Rennen", loc="left",
                color=FG, fontsize=12, pad=10)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="x", alpha=0.3, linewidth=0.8, color=GRID)
    ax.set_axisbelow(True)


def zeichne_titelchance(ax, chance: pd.Series) -> None:
    top = chance[chance > 0.01].head(6)
    farben = [SERIEN[0] if i == 0 else MUTED for i in range(len(top))]
    ax.bar(top.index, top.to_numpy(), color=farben, width=0.55)
    ax.set_ylabel("Titelchance [%] (Monte-Carlo)")
    ax.set_title("Konstrukteurs-Titelchance, Restsaison simuliert", loc="left",
                color=FG, fontsize=12, pad=10)
    for i, v in enumerate(top.to_numpy()):
        ax.text(i, v + 1, f"{v:.1f}%", ha="center", color=FG, fontsize=9)
    ax.set_ylim(0, 108)
    ax.tick_params(axis="x", labelrotation=12)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="y", alpha=0.3, linewidth=0.8, color=GRID)
    ax.set_axisbelow(True)


def zeichne_dnf_raten(ax, dnf_rate: dict, name_map: dict) -> None:
    s = pd.Series({name_map[t]: r * 100 for t, r in dnf_rate.items() if r > 0}
                 ).sort_values()
    farben = [SERIEN[1] if v == s.max() else MUTED for v in s.to_numpy()]
    ax.barh(s.index, s.to_numpy(), color=farben, height=0.6)
    ax.set_xlabel("Ausfallquote je Auto, bisherige Saison [%]")
    ax.set_title("Ausfallquote je Konstrukteur (beide Autos gepoolt)",
                loc="left", color=FG, fontsize=12, pad=10)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="x", alpha=0.3, linewidth=0.8, color=GRID)
    ax.set_axisbelow(True)


def zeichne_vergleich(ax, vgl: pd.DataFrame) -> None:
    v = vgl.head(6)
    x = np.arange(len(v))
    w = 0.35
    ax.bar(x - w / 2, v["ohne_dnf"], width=w, color=MUTED, label="ohne DNF-Modell")
    ax.bar(x + w / 2, v["mit_dnf"], width=w, color=SERIEN[1], label="mit DNF-Modell")
    ax.set_xticks(x, v.index, rotation=12)
    ax.set_ylabel("P(Top 3 Team-Endstand) [%]")
    ax.set_title("AUSBAUSTUFE: Ausfallmodell gegen reine Positions-Stichprobe",
                loc="left", color=FG, fontsize=12, pad=10)
    ax.legend(loc="upper right", frameon=False, labelcolor=FG, fontsize=9)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="y", alpha=0.3, linewidth=0.8, color=GRID)
    ax.set_axisbelow(True)


def main():
    f1lab.enable_cache()
    plt.rcParams.update(matplotlib_stil())
    erg = Ergast(result_type="pandas", auto_cast=True)

    print(f"[1/4] Konstrukteurs-Stand {YEAR} laden (VORGEHEN 1) ...")
    stand_antwort = f1lab.ergast_retry(erg.get_constructor_standings, season=YEAR)
    if not stand_antwort.content or stand_antwort.content[0].empty:
        print(f"      Saison {YEAR} hat noch nicht begonnen, kein WM-Stand "
              "vorhanden.")
        return
    standings = stand_antwort.content[0]
    print(standings[["position", "points", "wins", "constructorName"]]
         .head(10).to_string(index=False))

    print("\n[2/4] Verbleibende Events (VORGEHEN 2) ...")
    remaining = fastf1.get_events_remaining(include_testing=False)
    sprint_flags = remaining["EventFormat"].str.contains("sprint", case=False).tolist()
    n_races, n_sprints = len(remaining), sum(sprint_flags)
    # ein team bringt zwei autos - platz 1+2 statt nur platz 1 pro event.
    max_points = n_races * (25 + 18) + n_sprints * (8 + 7)
    print(f"      {n_races} Rennen, {n_sprints} Sprints verbleibend -> "
         f"max. {max_points} Punkte fuer ein einzelnes Team (beide Autos "
         f"vorne)")

    print("\n[3/4] Rechnerische Titelchance (VORGEHEN 3) ...")
    leader = standings["points"].max()
    standings["moeglich"] = standings["points"] + max_points >= leader
    print(f"      Rechnerisch noch im Titelrennen: "
         f"{standings['moeglich'].sum()}/{len(standings)} Teams")

    gefahrene_runden = 23 - n_races
    print(f"\n[4/4] Saisonverlauf laden ({gefahrene_runden} Runden) fuer "
         f"Monte-Carlo (VORGEHEN 4) ...")
    races, sprints = saison_verlauf(erg, YEAR, gefahrene_runden)
    base_points = standings.set_index("constructorId")["points"].to_dict()
    name_map = standings.set_index("constructorId")["constructorName"].to_dict()
    teams = list(base_points.keys())

    pos_hist_alle = (races.groupby("constructorId")["position"]
                     .apply(lambda s: s.dropna().to_numpy()).to_dict())
    pos_hist_ohne_dnf = (races[~races["dnf"]].groupby("constructorId")["position"]
                         .apply(lambda s: s.dropna().to_numpy()).to_dict())
    dnf_rate = races.groupby("constructorId")["dnf"].mean().to_dict()
    sprint_hist = (sprints.groupby("constructorId")["position"]
                  .apply(lambda s: s.dropna().to_numpy()).to_dict()
                  if not sprints.empty else {})

    rng = np.random.default_rng(7)
    sim_basis = monte_carlo_team(base_points, teams, pos_hist_alle, sprint_hist,
                                 sprint_flags, None, rng)
    chance = pd.Series(
        {name_map[t]: (sim_basis.idxmax(axis=1) == t).mean() * 100
         for t in teams}).sort_values(ascending=False)
    print("\n      Konstrukteurs-Titelchance [%] (Monte-Carlo, Basis-Version):")
    print(chance[chance > 0.05].round(2).to_string())

    rng2 = np.random.default_rng(7)
    sim_dnf = monte_carlo_team(base_points, teams, pos_hist_ohne_dnf, sprint_hist,
                               sprint_flags, dnf_rate, rng2)

    def top3_quote(sim: pd.DataFrame) -> pd.Series:
        rang = (-sim).rank(axis=1, method="first")
        return pd.Series({name_map[t]: (rang[t] <= 3).mean() * 100
                          for t in teams})

    vgl = pd.DataFrame({"ohne_dnf": top3_quote(sim_basis),
                        "mit_dnf": top3_quote(sim_dnf)})
    vgl = vgl.sort_values("mit_dnf", ascending=False)
    print("\n      P(Top 3 im Team-Endstand) [%], ohne/mit DNF-Modell:")
    print(vgl[(vgl["ohne_dnf"] > 0.1) | (vgl["mit_dnf"] > 0.1)]
         .round(2).to_string())

    print("\nGrafik ...")
    fig, ax = plt.subplots(2, 2, figsize=(15, 10))
    zeichne_standings(ax[0, 0], standings, gefahrene_runden)
    zeichne_titelchance(ax[0, 1], chance)
    zeichne_dnf_raten(ax[1, 0], dnf_rate, name_map)
    zeichne_vergleich(ax[1, 1], vgl)

    fig.suptitle(f"Konstrukteurs-WM-Simulator {YEAR}", x=0.09, ha="left",
                fontsize=16, color=FG, y=1.0)
    plt.tight_layout()
    path = OUT / "konstrukteurs_wm_simulator.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"\n      -> {path}")


if __name__ == "__main__":
    main()
