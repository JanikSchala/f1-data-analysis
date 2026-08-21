"""Grafiken fuer den Wochenend-Report.

selber Hausstil wie die 01-12-Skripte, ueber f1lab.design. damit sehen
PDF-Report und App gleich aus.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

from f1lab.design import FG, GRID, MUTED, SERIEN, matplotlib_stil

plt.rcParams.update(matplotlib_stil())


def plot_pace(pace_df: pd.DataFrame, titel: str) -> plt.Figure:
    """bereinigte Race Pace als Balken, Delta zum Schnellsten."""
    fig, ax = plt.subplots(figsize=(10, max(4, 0.32 * len(pace_df))))
    df = pace_df.sort_values("delta_s")
    farben = [SERIEN[0] if i < 3 else MUTED for i in range(len(df))]
    ax.barh(df["driver"], df["delta_s"], color=farben, height=0.6,
           xerr=[df["delta_s"] - df["ci_lo"], df["ci_hi"] - df["delta_s"]],
           error_kw={"ecolor": MUTED, "elinewidth": 1, "capsize": 2})
    ax.invert_yaxis()
    ax.set_xlabel("Delta zum Schnellsten [s]")
    ax.set_title(titel, loc="left", color=FG, fontsize=13, pad=10)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="x", alpha=0.3, linewidth=0.8, color=GRID)
    ax.set_axisbelow(True)
    fig.tight_layout()
    return fig


def plot_strategy(stints_df: pd.DataFrame, order: list[str], titel: str) -> plt.Figure:
    """strategie-Gantt-Chart, dieselbe Bauweise wie P14."""
    from f1lab.design import COMPOUND

    fig, ax = plt.subplots(figsize=(10, max(4, 0.32 * len(order))))
    for drv in order:
        prev = 0
        for s in stints_df[stints_df["Driver"] == drv].sort_values("stint").itertuples():
            farbe = COMPOUND.get(str(s.Compound).upper(), MUTED)
            ax.barh(drv, s.laps, left=prev, color=farbe, edgecolor="#00000055",
                   height=0.7)
            prev += s.laps
    ax.invert_yaxis()
    ax.set_xlabel("Runde")
    ax.set_title(titel, loc="left", color=FG, fontsize=13, pad=10)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="x", alpha=0.3, linewidth=0.8, color=GRID)
    ax.set_axisbelow(True)
    fig.tight_layout()
    return fig


def build_pdf(race_session, quali_session, pace_df: pd.DataFrame,
              stints_df: pd.DataFrame, path: Path) -> None:
    """alle Grafiken eines Wochenendes in ein PDF, eine Seite je Grafik
    plus eine Titelseite."""
    order = (race_session.results.sort_values("Position")["Abbreviation"]
            .loc[lambda s: s.isin(stints_df["Driver"].unique())].tolist())

    with PdfPages(path) as pdf:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.axis("off")
        ax.text(0.5, 0.6, race_session.event["EventName"], ha="center",
               fontsize=22, color=FG, transform=ax.transAxes)
        ax.text(0.5, 0.5, str(race_session.event.year), ha="center",
               fontsize=14, color=MUTED, transform=ax.transAxes)
        fig.patch.set_facecolor("#15151e")
        pdf.savefig(fig)
        plt.close(fig)

        pdf.savefig(plot_pace(pace_df, f"{race_session.event['EventName']} - "
                              "Race Pace"))
        plt.close("all")
        pdf.savefig(plot_strategy(stints_df, order,
                                  f"{race_session.event['EventName']} - "
                                  "Reifenstrategie"))
        plt.close("all")
