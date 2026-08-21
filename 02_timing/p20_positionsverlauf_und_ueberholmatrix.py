"""zeichnet den positionsverlauf und die ueberholmatrix und trennt versuchte
von erfolgreichen ueberholmanoevern"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")                      # nur dateien statt fenster

import fastf1.plotting as f1plt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import f1lab
from f1lab.design import FG, GRID, MUTED, POSITIV, SERIEN, matplotlib_stil

warnings.filterwarnings("ignore")

OUT = Path(__file__).parent / "out"
OUT.mkdir(exist_ok=True)

SEASON, EVENT, IDENT = 2024, "Monza", "R"
ANNAEHERUNG_M = 3.0

plt.rcParams.update(matplotlib_stil())
f1plt.setup_mpl(mpl_timedelta_support=False, color_scheme="fastf1")


def naeherungen(laps_gruen: pd.DataFrame, pos: pd.DataFrame,
                nummer_zu_code: dict[str, str]) -> pd.DataFrame:
    """minimaler abstand zum direkten rivalen je runde und fahrer. rohmaterial
    fuer die versuch/erfolg-klassifikation."""
    beobachtungen = []
    for drv in laps_gruen["Driver"].unique():
        d_laps = laps_gruen[laps_gruen["Driver"] == drv].sort_values("LapStartTime")
        if d_laps.empty:
            continue
        tel = d_laps.get_telemetry().add_driver_ahead().sort_values("SessionTime")
        tel = tel[tel["DriverAhead"] != ""]
        if tel.empty:
            continue
        tel = tel.copy()
        tel["ZielCode"] = tel["DriverAhead"].map(nummer_zu_code)
        zug = pd.merge_asof(
            tel, d_laps[["LapNumber", "LapStartTime"]]
            .rename(columns={"LapStartTime": "SessionTime"}),
            on="SessionTime", direction="backward")

        for lap, g in zug.groupby("LapNumber"):
            if lap < 2 or lap not in pos.index or (lap - 1) not in pos.index \
                    or drv not in pos.columns:
                continue
            p_prev = pos.loc[lap - 1, drv]
            if pd.isna(p_prev):
                continue
            rivalen = pos.loc[lap - 1][pos.loc[lap - 1] == p_prev - 1]
            if rivalen.empty:
                continue
            rivale = rivalen.index[0]
            sub = g[g["ZielCode"] == rivale]
            if sub.empty:
                continue
            beobachtungen.append({
                "chaser": drv, "rivale": rivale, "lap": int(lap),
                "min_dist": float(sub["DistanceToDriverAhead"].min())})
    return pd.DataFrame(beobachtungen)


def klassifiziere(naeh: pd.DataFrame, pos: pd.DataFrame,
                  schwelle: float = ANNAEHERUNG_M) -> pd.DataFrame:
    """markiert versuche unter der annaeherungs-schwelle als erfolgreich wenn
    der angreifer die runde vor dem rivalen beendet."""
    versuche = naeh[naeh["min_dist"] < schwelle].copy()

    def erfolgreich(row) -> bool:
        lap = row["lap"]
        if lap not in pos.index:
            return False
        p_a = pos.loc[lap].get(row["chaser"], np.nan)
        p_b = pos.loc[lap].get(row["rivale"], np.nan)
        return bool(pd.notna(p_a) and pd.notna(p_b) and p_a < p_b)

    versuche["erfolgreich"] = versuche.apply(erfolgreich, axis=1)
    return versuche


def zeichne_position(ax, pos: pd.DataFrame, ses) -> None:
    for drv in pos.columns:
        style = f1plt.get_driver_style(drv, ["color", "linestyle"], session=ses)
        ax.plot(pos.index, pos[drv], label=drv, lw=2, **style)
    ax.set_ylim(20.5, 0.5)
    ax.set_yticks(range(1, 21))
    ax.set_xlabel("Runde")
    ax.set_ylabel("Position")
    ax.set_title("Positionsverlauf", loc="left", color=FG, fontsize=13, pad=12)
    ax.grid(alpha=0.25, linewidth=0.8, color=GRID)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    f1plt.add_sorted_driver_legend(ax, ses)


def zeichne_ueberholungen(ax, mat: pd.DataFrame) -> None:
    gesamt = mat.sum(axis=1).sort_values()
    gesamt = gesamt[gesamt > 0]
    ax.barh(gesamt.index, gesamt.to_numpy(), color=SERIEN[0], height=0.65)
    ax.set_xlabel("Ueberholungen (ohne Boxenstopp-Effekt)")
    ax.set_title("Ueberholungen gesamt je Fahrer", loc="left", color=FG,
                fontsize=13, pad=12)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="x", alpha=0.35, linewidth=0.8, color=GRID)
    ax.set_axisbelow(True)


def zeichne_versuche(ax, versuche: pd.DataFrame) -> None:
    je_fahrer = (versuche.groupby(["chaser", "erfolgreich"]).size()
                .unstack(fill_value=0))
    for spalte in (True, False):
        if spalte not in je_fahrer.columns:
            je_fahrer[spalte] = 0
    je_fahrer = je_fahrer.loc[je_fahrer.sum(axis=1).sort_values().index]

    ax.barh(je_fahrer.index, je_fahrer[True], color=POSITIV, height=0.65,
           label="erfolgreich")
    ax.barh(je_fahrer.index, je_fahrer[False], left=je_fahrer[True],
           color=MUTED, height=0.65, label="ohne Erfolg")
    ax.set_xlabel(f"Versuche (Annaeherung < {ANNAEHERUNG_M:.0f} m)")
    ax.set_title("Versuchte gegen erfolgreiche Ueberholmanoever", loc="left",
                color=FG, fontsize=13, pad=12)
    ax.legend(loc="lower right", frameon=False, labelcolor=FG, fontsize=9)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="x", alpha=0.35, linewidth=0.8, color=GRID)
    ax.set_axisbelow(True)


def main():
    f1lab.enable_cache()

    print(f"[1/4] {EVENT} {SEASON} {IDENT} laden (mit Telemetrie) ...")
    ses = f1lab.load(SEASON, EVENT, IDENT, telemetry=True)
    laps_alle = ses.laps
    laps_gruen = laps_alle.pick_track_status("1").pick_wo_box()
    nummer_zu_code = (laps_alle.drop_duplicates("DriverNumber")
                      .set_index("DriverNumber")["Driver"].to_dict())

    print("[2/4] Positionsverlauf und Ueberholmatrix ...")
    pos = f1lab.position_progression(ses)
    mat = f1lab.overtakes_matrix(ses)
    print("Ueberholungen gesamt je Fahrer:")
    print(mat.sum(axis=1).sort_values(ascending=False).head(8).to_string())

    print(f"\n[3/4] Annaeherungen an den direkten Rivalen "
         f"(< {ANNAEHERUNG_M:.0f} m) ...")
    naeh = naeherungen(laps_gruen, pos, nummer_zu_code)
    versuche = klassifiziere(naeh, pos)
    quote = versuche["erfolgreich"].mean() if len(versuche) else float("nan")
    print(f"      {len(versuche)} Versuche, {int(versuche['erfolgreich'].sum())} "
         f"erfolgreich ({100 * quote:.0f}%)")
    if len(versuche):
        knapp = versuche[~versuche["erfolgreich"]].nsmallest(3, "min_dist")
        print("\nKnappste erfolglose Versuche:")
        print(knapp[["chaser", "rivale", "lap", "min_dist"]].to_string(index=False))

    print("\n[4/4] Grafik ...")
    fig = plt.figure(figsize=(15, 13))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.5, 1], hspace=0.32, wspace=0.28)
    zeichne_position(fig.add_subplot(gs[0, :]), pos, ses)
    zeichne_ueberholungen(fig.add_subplot(gs[1, 0]), mat)
    zeichne_versuche(fig.add_subplot(gs[1, 1]), versuche)

    fig.suptitle(f"{EVENT} {SEASON} {IDENT} - Positionsverlauf und "
                f"Ueberholmatrix", x=0.09, ha="left", fontsize=16, color=FG,
                y=0.995)
    path = OUT / "positionsverlauf_ueberholmatrix.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"\n      -> {path}")


if __name__ == "__main__":
    main()
