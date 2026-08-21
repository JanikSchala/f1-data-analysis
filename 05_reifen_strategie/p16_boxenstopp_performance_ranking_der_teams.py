"""rankt boxenstoppzeiten der teams über eine saison aus der ergast/jolpica-api und prüft den einfluss von druck und boxenfenstern"""
from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")                      # kein fenster, nur dateien

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from fastf1.ergast import Ergast

import f1lab
from f1lab.design import FG, GRID, MUTED, SERIEN, matplotlib_stil

warnings.filterwarnings("ignore")

OUT = Path(__file__).parent / "out"
OUT.mkdir(exist_ok=True)

YEAR = 2024
DAUER_MIN, DAUER_MAX = 15.0, 35.0     # volle boxengassen-zeit, nicht die reine standzeit

plt.rcParams.update(matplotlib_stil())


def _mit_wiederholung(fn, *args, versuche: int = 5, **kwargs):
    """ergast/jolpica limitiert die anfragerate. exponentielles backoff statt abbruch beim ersten 429."""
    for i in range(versuche):
        try:
            return fn(*args, **kwargs)
        except Exception:
            if i == versuche - 1:
                raise
            time.sleep(3 * (i + 1))


def hole_saison(erg: Ergast, jahr: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """pitstops und rennergebnisse aller runden, mit paging und retry."""
    sched = _mit_wiederholung(erg.get_race_schedule, season=jahr)

    pit_frames, res_frames = [], []
    for _, race in sched.iterrows():
        rnd = int(race["round"])
        stops = _mit_wiederholung(erg.get_pit_stops, season=jahr, round=rnd,
                                  limit=200).content
        res = _mit_wiederholung(erg.get_race_results, season=jahr,
                                round=rnd).content
        if stops and not stops[0].empty:
            s = stops[0].copy()
            s["round"] = rnd
            pit_frames.append(s)
        if res and not res[0].empty:
            r = res[0].copy()
            r["round"] = rnd
            res_frames.append(r)
        time.sleep(0.4)     # bewusst gebremst statt in jede runde ein 429

    return pd.concat(pit_frames, ignore_index=True), \
        pd.concat(res_frames, ignore_index=True)


def zeichne_ranking(ax, rank: pd.DataFrame) -> None:
    """median je team als punkt gegen den schnellsten einzelstopp als strich. nur die drei schnellsten mediane sind hervorgehoben weil zehn kategorien mehr sind als MAX_SERIEN=3 hausfarben hergeben.

    kein fehlerbalken mit dem vollen IQR weil der teils über 5 s reicht und
    die enge spannweite der mediane auf dieser achse unlesbar machen würde.
    der IQR steht stattdessen in der gedruckten tabelle.
    """
    r = rank.sort_values("Median", ascending=False)
    y = np.arange(len(r))
    farben = [SERIEN[0] if v <= rank["Median"].nsmallest(3).max() else MUTED
             for v in r["Median"]]
    ax.hlines(y, r["Bester"], r["Median"], color=GRID, lw=2, zorder=1)
    ax.scatter(r["Bester"], y, marker="|", s=160, color=MUTED, zorder=2,
              linewidths=1.6)
    ax.scatter(r["Median"], y, s=90, color=farben, zorder=3)
    ax.set_yticks(y, r.index)
    ax.set_xlabel("Boxengassen-Zeit [s] - Punkt = Median, Strich = bester Stopp")
    ax.set_title(f"{len(r)} Teams, Saison {YEAR} - Median-Spannweite nur "
                f"{r['Median'].max() - r['Median'].min():.1f} s", loc="left",
                color=FG, fontsize=13, pad=10)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="x", alpha=0.3, linewidth=0.8, color=GRID)
    ax.set_axisbelow(True)


def zeichne_druck(ax, pit: pd.DataFrame) -> None:
    """stopps in einem vollen boxenfenster (>=4 autos, gleiche runde) gegen isolierte einzelstopps."""
    stapel = pit.groupby(["round", "lap"]).size().rename("n_stops")
    p = pit.merge(stapel, on=["round", "lap"])
    gruppen = {
        "isoliert\n(1 Stopp)": p.loc[p["n_stops"] == 1, "dur"],
        "Boxenfenster\n(>=4 gleichzeitig)": p.loc[p["n_stops"] >= 4, "dur"],
    }
    ax.boxplot(gruppen.values(), tick_labels=gruppen.keys(), widths=0.5,
              patch_artist=True,
              boxprops={"facecolor": SERIEN[0], "edgecolor": FG, "alpha": 0.85},
              medianprops={"color": FG, "linewidth": 1.6},
              whiskerprops={"color": MUTED}, capprops={"color": MUTED},
              flierprops={"markeredgecolor": MUTED, "markersize": 3})
    delta = gruppen["Boxenfenster\n(>=4 gleichzeitig)"].median() - \
        gruppen["isoliert\n(1 Stopp)"].median()
    ax.set_ylabel("Boxengassen-Zeit [s]")
    ax.set_title(f"Volles Boxenfenster kostet +{delta:.1f} s Median", loc="left",
                color=FG, fontsize=13, pad=10)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="y", alpha=0.3, linewidth=0.8, color=GRID)
    ax.set_axisbelow(True)


def zeichne_position(ax, pit: pd.DataFrame) -> None:
    """stoppzeit gegen ergebnis-position, mit regressionsgerade."""
    ok = pit.dropna(subset=["position"])
    slope, inter = np.polyfit(ok["position"], ok["dur"], 1)
    korr = np.corrcoef(ok["position"], ok["dur"])[0, 1]

    ax.scatter(ok["position"], ok["dur"], s=14, color=MUTED, alpha=0.5,
              edgecolors="none")
    xs = np.linspace(ok["position"].min(), ok["position"].max(), 50)
    ax.plot(xs, slope * xs + inter, color=SERIEN[1], lw=2)
    ax.set_xlabel("Zielposition")
    ax.set_ylabel("Boxengassen-Zeit [s]")
    ax.set_title(f"Position im Ergebnis vs. Stoppzeit - r={korr:+.2f}",
                loc="left", color=FG, fontsize=13, pad=10)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(alpha=0.3, linewidth=0.8, color=GRID)
    ax.set_axisbelow(True)


def main():
    f1lab.enable_cache()
    print(f"[1/3] Ergast-Saison {YEAR}: Pitstops + Ergebnisse je Runde "
         f"(gebremst wegen Rate-Limit, dauert etwas) ...")
    erg = Ergast(result_type="pandas", auto_cast=True)
    pit_raw, res = hole_saison(erg, YEAR)
    print(f"      {len(pit_raw)} Boxenstopps, {len(res)} Ergebniszeilen "
         f"ueber {pit_raw['round'].nunique()} Runden")

    print(f"\n[2/3] Filter {DAUER_MIN:.0f}-{DAUER_MAX:.0f}s (volle "
         f"Boxengassen-Zeit, siehe Docstring) und Konstrukteur-Join ...")
    pit_raw["dur"] = pd.to_timedelta(
        pit_raw["duration"], errors="coerce").dt.total_seconds()
    pit = pit_raw[pit_raw["dur"].between(DAUER_MIN, DAUER_MAX)].copy()
    pit = pit.merge(res[["round", "driverId", "constructorName", "position"]],
                    on=["round", "driverId"], how="left")
    print(f"      {len(pit)}/{len(pit_raw)} Stopps behalten "
         f"({len(pit_raw) - len(pit)} Reparaturen/Strafen verworfen)")

    rank = (pit.groupby("constructorName")["dur"]
            .agg(Median="median", Bester="min",
                 IQR=lambda s: s.quantile(0.75) - s.quantile(0.25),
                 Stopps="count").sort_values("Median").round(3))
    print("\n      Ranking (Median, s):")
    print(rank.to_string())
    print("\n      Schnellste 5 Stopps der Saison:")
    print(pit.nsmallest(5, "dur")[
        ["round", "driverId", "constructorName", "dur"]].to_string(index=False))

    print("\n[3/3] AUSBAUSTUFE: Position und Boxenfenster ...")
    ok = pit.dropna(subset=["position"])
    korr = np.corrcoef(ok["position"], ok["dur"])[0, 1]
    print(f"      corr(Position, Stoppzeit) = {korr:+.3f} (n={len(ok)})")

    stapel = pit.groupby(["round", "lap"]).size().rename("n_stops")
    p = pit.merge(stapel, on=["round", "lap"])
    iso = p.loc[p["n_stops"] == 1, "dur"]
    voll = p.loc[p["n_stops"] >= 4, "dur"]
    print(f"      isoliert: n={len(iso)} median={iso.median():.2f}s   "
         f"Boxenfenster (>=4): n={len(voll)} median={voll.median():.2f}s   "
         f"delta={voll.median() - iso.median():+.2f}s")

    print("\nGrafik ...")
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.1, 1], hspace=0.45, wspace=0.3)
    zeichne_ranking(fig.add_subplot(gs[0, :]), rank)
    zeichne_druck(fig.add_subplot(gs[1, 0]), pit)
    zeichne_position(fig.add_subplot(gs[1, 1]), pit)
    fig.suptitle(f"Boxenstopp-Performance-Ranking {YEAR}", x=0.09, ha="left",
                fontsize=16, color=FG, y=0.995)
    path = OUT / "boxenstopp_ranking.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"\n      -> {path}")


if __name__ == "__main__":
    main()
