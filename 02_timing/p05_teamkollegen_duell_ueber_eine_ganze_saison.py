"""scannt eine saison auf teamkollegen-duelle und berechnet daraus ein
elo-rating je fahrer"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")                      # nur dateien statt fenster

import fastf1
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

import f1lab
from f1lab.design import FG, GRID, MUTED, RAMPE, SERIEN, matplotlib_stil

warnings.filterwarnings("ignore")
# der saison-scan laedt 48 sessions am stueck. fastf1s hintergrundabgleich mit
# dem ergast-spiegel laeuft dabei staendig ins rate-limit und faellt auf den
# eigenen cache zurueck. unschaedlich aber sehr laut ohne diese zeile.
fastf1.set_log_level("ERROR")

OUT = Path(__file__).parent / "out"
OUT.mkdir(exist_ok=True)

YEAR = 2024
START_ELO = 1500.0
K = 24.0
RAMPE_CMAP = LinearSegmentedColormap.from_list("rampe", RAMPE)

plt.rcParams.update(matplotlib_stil())


def saison_scannen(year: int) -> pd.DataFrame:
    """iteriert den ganzen kalender und ueberspringt fehlende sessions robust.
    die duell-extraktion je session steckt in f1lab.teammate_duels()."""
    schedule = fastf1.get_event_schedule(year, include_testing=False)
    rows = []
    for _, ev in schedule.iterrows():
        rnd = int(ev["RoundNumber"])
        for ident in ("Q", "R"):
            try:
                ses = f1lab.load(year, rnd, ident)
                duelle = f1lab.teammate_duels(ses)
            except Exception as exc:
                print(f"      R{rnd:>2} {ident} uebersprungen: "
                     f"{type(exc).__name__}")
                continue
            for d in duelle:
                rows.append({"round": rnd, "event": ev["EventName"],
                            "session": ident, **d})
    return pd.DataFrame(rows)


def elo_verlauf(duelle: pd.DataFrame, k: float = K,
                start: float = START_ELO) -> pd.DataFrame:
    """elo-rating nach jedem rennwochenende ueber quali- und rennduelle."""
    rating: dict[str, float] = {}
    verlauf = []
    for rnd, gruppe in duelle.sort_values(["round", "session"]).groupby("round"):
        for _, d in gruppe.iterrows():
            ra = rating.get(d["a"], start)
            rb = rating.get(d["b"], start)
            rating[d["a"]], rating[d["b"]] = f1lab.elo_update(
                ra, rb, d["score_a"], k=k)
        for drv, elo in rating.items():
            verlauf.append({"round": rnd, "driver": drv, "elo": elo})
    return pd.DataFrame(verlauf)


def zeichne_heatmap(ax, quali: pd.DataFrame) -> None:
    """team x runde, farbe = wie deutlich der schnellere teamkollege vorn lag.
    sortiert nach mittlerem rueckstand. auf dem 90.-perzentil gekappt damit
    keine einzelne ausreisser-session die ganze farbskala dominiert."""
    pivot = quali.pivot_table(index="team", columns="round",
                              values="delta_pct", aggfunc="first")
    pivot = pivot.reindex(pivot.mean(axis=1).sort_values().index)

    im = ax.imshow(pivot.to_numpy(), aspect="auto", cmap=RAMPE_CMAP,
                   vmin=0, vmax=np.nanpercentile(pivot.to_numpy(), 90))
    ax.set_yticks(range(len(pivot.index)), pivot.index, fontsize=9)
    ax.set_xticks(range(0, len(pivot.columns), 2),
                  pivot.columns[::2], fontsize=8)
    ax.set_xlabel("Rennwoche")
    ax.set_title("Quali-Dominanz je Team und Rennen", loc="left", color=FG,
                fontsize=13, pad=12)
    for side in ax.spines.values():
        side.set_visible(False)

    cb = ax.figure.colorbar(im, ax=ax, pad=0.02)
    cb.set_label("Rueckstand des langsameren Teamkollegen [%]", color=MUTED,
                fontsize=9)
    cb.ax.tick_params(colors=MUTED, length=0)
    cb.outline.set_visible(False)


def zeichne_elo(ax, verlauf: pd.DataFrame, top_n: int = 3) -> None:
    """elo-verlauf aller fahrer. die top n am saisonende sind farbig
    hervorgehoben, der rest duenn und grau im hintergrund."""
    pivot = verlauf.pivot(index="round", columns="driver", values="elo")
    top = pivot.iloc[-1].dropna().sort_values(ascending=False).index[:top_n]

    for drv in pivot.columns:
        if drv not in top:
            ax.plot(pivot.index, pivot[drv], color=GRID, linewidth=1.0,
                   alpha=0.8, zorder=1)

    for i, drv in enumerate(top):
        serie = pivot[drv].dropna()
        ax.plot(serie.index, serie, color=SERIEN[i], linewidth=2.2, zorder=3)
        ax.text(serie.index[-1] + 0.3, serie.iloc[-1], drv, color=SERIEN[i],
               fontsize=10, va="center", fontweight="bold")

    ax.axhline(START_ELO, color=MUTED, linewidth=0.8, linestyle=":")
    ax.set_xlabel("Rennwoche")
    ax.set_ylabel("Elo-Rating")
    ax.set_title(f"Teamkollegen-Elo, Top {top_n} am Saisonende", loc="left",
                color=FG, fontsize=13, pad=12)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(alpha=0.3, linewidth=0.8, color=GRID)
    ax.set_axisbelow(True)


def main():
    f1lab.enable_cache()

    print(f"[1/3] {YEAR} scannen (Quali + Rennen, aus dem Cache) ...")
    duelle = saison_scannen(YEAR)
    print(f"      {len(duelle)} Duelle aus {duelle['round'].nunique()} "
         f"Wochenenden ({(duelle['session'] == 'Q').sum()} Quali, "
         f"{(duelle['session'] == 'R').sum()} Rennen)")

    print("[2/3] Elo-Verlauf ...")
    verlauf = elo_verlauf(duelle)
    endstand = (verlauf.sort_values(["driver", "round"])
               .groupby("driver")["elo"].last().sort_values(ascending=False))
    print("\nEndstand (Top 10):")
    print(endstand.head(10).round(1).to_string())

    quali_siege = (duelle[duelle["session"] == "Q"].groupby("a").size()
                  .rename("Quali-Siege").sort_values(ascending=False))
    print("\nMeiste Quali-Siege gegen den Teamkollegen:")
    print(quali_siege.head(5).to_string())

    print("[3/3] Grafik ...")
    fig, ax = plt.subplots(1, 2, figsize=(15, 8),
                           gridspec_kw={"width_ratios": [1, 1.15]})
    zeichne_heatmap(ax[0], duelle[duelle["session"] == "Q"])
    zeichne_elo(ax[1], verlauf)
    fig.suptitle(f"{YEAR} - Teamkollegen-Duell ueber die Saison", x=0.125,
                ha="left", fontsize=16, color=FG, y=1.02)
    plt.tight_layout()
    path = OUT / "teamkollegen_duell.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"\n      -> {path}")


if __name__ == "__main__":
    main()
