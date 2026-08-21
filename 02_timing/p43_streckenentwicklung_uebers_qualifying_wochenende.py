"""prueft per paarweisem fahrervergleich ob die strecke von Q1 zu Q3
schneller wird und ob temperatur oder sprint-format das aendern"""
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
from scipy.stats import pearsonr, wilcoxon

import f1lab
from f1lab.design import FG, GRID, MUTED, SERIEN, matplotlib_stil

warnings.filterwarnings("ignore")

OUT = Path(__file__).parent / "out"
OUT.mkdir(exist_ok=True)

SAISON = 2024
SANITY_EVENT = (2024, "Spain", "Q")

plt.rcParams.update(matplotlib_stil())


def ist_nass(session) -> bool:
    """regen ueberdeckt den reinen gummi-effekt und wird deshalb ausgeschlossen."""
    return session.laps["Compound"].isin(["INTERMEDIATE", "WET"]).any()


def saison_scan(saison: int) -> tuple[pd.DataFrame, list[str]]:
    """sammelt alle trockenen qualifyings einer saison."""
    schedule = f1lab.event_dimension([saison])
    alle, nass = [], []
    for _, row in schedule.iterrows():
        try:
            ses = f1lab.load(saison, int(row["round"]), "Q", telemetry=False)
        except Exception:
            continue
        if ist_nass(ses):
            nass.append(row["event_name"])
            continue
        d = f1lab.qualifying_track_evolution(ses)
        if not d.empty:
            d = d.copy()
            d["gp"] = row["event_name"]
            alle.append(d)
    return (pd.concat(alle, ignore_index=True) if alle else pd.DataFrame(),
            nass)


SPRINT_QUALI_IDENT = {2023: "SS", 2024: "SQ"}  # fia-umbenennung: SS -> SQ


def sprint_quali_scan(saisons) -> tuple[pd.DataFrame, list[str]]:
    """dieselbe frage an sprint qualifying/sprint shootout. der
    session-identifier wechselt zwischen den saisons (SS 2023, SQ 2024)."""
    alle, nass = [], []
    for saison in saisons:
        ident = SPRINT_QUALI_IDENT[saison]
        sched_roh = fastf1.get_event_schedule(saison, include_testing=False)
        sprint_namen = sched_roh[sched_roh["EventFormat"]
                                 .str.contains("sprint", case=False,
                                              na=False)]["EventName"].tolist()
        for gp in sprint_namen:
            try:
                ses = f1lab.load(saison, gp, ident, telemetry=False)
            except Exception:
                continue
            if ist_nass(ses):
                nass.append(f"{saison} {gp}")
                continue
            d = f1lab.qualifying_track_evolution(ses)
            if not d.empty:
                d = d.copy()
                d["gp"] = f"{saison} {gp}"
                alle.append(d)
    return (pd.concat(alle, ignore_index=True) if alle else pd.DataFrame(),
            nass)


def temperatur_confound(saison: int) -> pd.DataFrame:
    """TrackTemp-trend gegen pace-delta je rennen. eigener scan mit
    weather=True statt saison_scan(), das keine wetterdaten laedt."""
    schedule = f1lab.event_dimension([saison])
    zeilen = []
    for _, row in schedule.iterrows():
        try:
            ses = f1lab.load(saison, int(row["round"]), "Q", telemetry=False,
                             weather=True)
        except Exception:
            continue
        if ist_nass(ses):
            continue
        laps = ses.laps
        try:
            q1, q2, q3 = laps.split_qualifying_sessions()
        except Exception:
            continue
        if q1 is None or q2 is None or q3 is None or q1.empty or q2.empty or q3.empty:
            continue

        def bestzeit(q):
            gueltig = q.dropna(subset=["LapTime"])
            return (gueltig.groupby("Driver")["LapTime"].min()
                   if not gueltig.empty else pd.Series(dtype="timedelta64[ns]"))

        def temp(q):
            try:
                return q.get_weather_data()["TrackTemp"].mean()
            except Exception:
                return np.nan

        b1, b2, b3 = bestzeit(q1), bestzeit(q2), bestzeit(q3)
        t1, t2, t3 = temp(q1), temp(q2), temp(q3)
        for segment, a, b, ta, tb in (("Q1->Q2", b1, b2, t1, t2),
                                      ("Q2->Q3", b2, b3, t2, t3)):
            gemeinsam = a.index.intersection(b.index)
            if len(gemeinsam) == 0 or pd.isna(ta) or pd.isna(tb):
                continue
            zeilen.append({
                "gp": row["event_name"], "segment": segment,
                "pace_delta_s": (a[gemeinsam] - b[gemeinsam]).dt.total_seconds().median(),
                "temp_delta_c": ta - tb})  # positiv = strecke kuehlt ab
    return pd.DataFrame(zeilen)


def zeichne_temperatur(ax, temp_df: pd.DataFrame) -> None:
    """streudiagramm von temperatur-delta gegen pace-delta."""
    for seg, farbe, marker in (("Q1->Q2", SERIEN[0], "o"),
                               ("Q2->Q3", SERIEN[1], "s")):
        sub = temp_df[temp_df["segment"] == seg]
        ax.scatter(sub["temp_delta_c"], sub["pace_delta_s"], color=farbe,
                  marker=marker, s=60, label=seg, alpha=0.85,
                  edgecolors=FG, linewidths=0.5)
    ax.axvline(0, color=MUTED, lw=1, ls="--")
    ax.axhline(0, color=MUTED, lw=1, ls="--")
    ax.set_xlabel("Temperatur-Delta [°C], positiv = Strecke kuehlt ab")
    ax.set_ylabel("Pace-Delta [s], positiv = spaeteres Segment schneller")
    ax.set_title("ZWEITE AUSBAUSTUFE: Kuehlt die Strecke ab, oder gummiert "
                "sie ein?", loc="left", color=FG, fontsize=13, pad=10)
    ax.legend(loc="upper right", frameon=False, labelcolor=FG, fontsize=9)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(alpha=0.3, linewidth=0.8, color=GRID)
    ax.set_axisbelow(True)


def zeichne_verteilung(ax, deltas: pd.DataFrame) -> None:
    """verteilung der paarweisen deltas je segment."""
    segmente = ["Q1->Q2", "Q2->Q3"]
    daten = [deltas.loc[deltas["segment"] == s, "delta_s"] for s in segmente]
    bp = ax.boxplot(daten, tick_labels=segmente, patch_artist=True,
                    widths=0.5, showfliers=False)
    for patch, farbe in zip(bp["boxes"], SERIEN):
        patch.set_facecolor(farbe)
        patch.set_alpha(0.6)
    for element in ("whiskers", "caps", "medians"):
        for line in bp[element]:
            line.set_color(FG)
    ax.axhline(0, color=MUTED, lw=1, ls="--")
    for i, s in enumerate(segmente):
        med = deltas.loc[deltas["segment"] == s, "delta_s"].median()
        pos = (deltas.loc[deltas["segment"] == s, "delta_s"] > 0).mean()
        ax.text(i + 1, ax.get_ylim()[1] * 0.9,
               f"Median {med:+.3f}s\n{pos:.0%} positiv",
               ha="center", fontsize=9, color=FG)
    ax.set_ylabel("Delta [s], positiv = spaeteres Segment schneller")
    ax.set_title(f"Fahrer-Paarvergleich je Segment, Saison {SAISON} "
                f"(trockene Qualifyings)", loc="left", color=FG, fontsize=13,
                pad=10)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="y", alpha=0.3, linewidth=0.8, color=GRID)
    ax.set_axisbelow(True)


def zeichne_konsistenz(ax, je_rennen: pd.DataFrame) -> None:
    """zeigt ob die richtung in jedem einzelnen rennen haelt."""
    e = je_rennen.sort_values("median_q1q2")
    x = np.arange(len(e))
    breite = 0.38
    ax.bar(x - breite / 2, e["median_q1q2"], width=breite, color=SERIEN[0],
          label="Q1->Q2")
    ax.bar(x + breite / 2, e["median_q2q3"], width=breite, color=SERIEN[1],
          label="Q2->Q3")
    ax.axhline(0, color=MUTED, lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [g.replace(" Grand Prix", "") for g in e["gp"]], rotation=60,
        ha="right", fontsize=7)
    ax.set_ylabel("Median-Delta je Rennen [s]")
    ax.set_title("AUSBAUSTUFE-Vorstufe: Konsistenz ueber alle Rennen der "
                f"Saison ({len(e)} trockene Qualifyings)", loc="left",
                color=FG, fontsize=13, pad=10)
    ax.legend(loc="upper left", frameon=False, labelcolor=FG, fontsize=9)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="y", alpha=0.3, linewidth=0.8, color=GRID)
    ax.set_axisbelow(True)


def main():
    f1lab.enable_cache()

    print(f"[1/5] Sanity-Check: {SANITY_EVENT[1]} {SANITY_EVENT[0]} "
         f"{SANITY_EVENT[2]} (VORGEHEN 1) ...")
    ses = f1lab.load(*SANITY_EVENT, telemetry=False)
    einzel = f1lab.qualifying_track_evolution(ses)
    for seg, g in einzel.groupby("segment"):
        print(f"      {seg}: n={len(g)}, Median={g['delta_s'].median():+.3f}s")

    print(f"\n[2-3/5] Saison-Scan {SAISON}, nasse Sessions raus "
         "(VORGEHEN 2-3) ...")
    deltas, nass = saison_scan(SAISON)
    print(f"      {deltas['gp'].nunique()} trockene Qualifyings, "
         f"{len(nass)} wegen Regen ausgeschlossen: {nass}")

    print("\n[4/5] Statistik ueber alle Fahrer-Paare (VORGEHEN 4) ...")
    for seg in ("Q1->Q2", "Q2->Q3"):
        werte = deltas.loc[deltas["segment"] == seg, "delta_s"]
        test = wilcoxon(werte)
        print(f"      {seg}: n={len(werte)}, Median={werte.median():+.3f}s, "
             f"Anteil positiv={( werte > 0).mean():.1%}, "
             f"Wilcoxon p={test.pvalue:.2e}")

    print("\n[5/5] Je-Rennen-Konsistenz (VORGEHEN 5) ...")
    je_rennen = (deltas.groupby(["gp", "segment"])["delta_s"].median()
                       .unstack("segment").reset_index()
                       .rename(columns={"Q1->Q2": "median_q1q2",
                                        "Q2->Q3": "median_q2q3"}))
    pos12 = (je_rennen["median_q1q2"] > 0).sum()
    pos23 = (je_rennen["median_q2q3"] > 0).sum()
    print(f"      Q1->Q2 positiv in {pos12}/{len(je_rennen)} Rennen")
    print(f"      Q2->Q3 positiv in {pos23}/{len(je_rennen)} Rennen")

    print("\nAUSBAUSTUFE: Cross-Saison-Check 2023 ...")
    deltas_2023, nass_2023 = saison_scan(2023)
    for seg in ("Q1->Q2", "Q2->Q3"):
        werte = deltas_2023.loc[deltas_2023["segment"] == seg, "delta_s"]
        test = wilcoxon(werte)
        je_r = (deltas_2023.loc[deltas_2023["segment"] == seg]
                          .groupby("gp")["delta_s"].median())
        print(f"      {seg}: n={len(werte)}, Median={werte.median():+.3f}s, "
             f"Anteil positiv={( werte > 0).mean():.1%}, p={test.pvalue:.2e}, "
             f"Rennen positiv={(je_r > 0).sum()}/{len(je_r)}")
    print(f"      ({len(nass_2023)} 2023er Qualifyings wegen Regen raus: "
         f"{nass_2023})")

    print("\nZWEITE AUSBAUSTUFE: Streckentemperatur als moeglicher "
         "Confound ...")
    temp_df = temperatur_confound(SAISON)
    for seg in ("Q1->Q2", "Q2->Q3"):
        sub = temp_df[temp_df["segment"] == seg]
        r, p = pearsonr(sub["temp_delta_c"], sub["pace_delta_s"])
        kuehlt_ab = (sub["temp_delta_c"] > 0).sum()
        erwaermt_aber_schneller = ((sub["temp_delta_c"] < 0)
                                   & (sub["pace_delta_s"] > 0)).sum()
        erwaermt = (sub["temp_delta_c"] < 0).sum()
        print(f"      {seg}: n={len(sub)} Rennen, Strecke kuehlt ab in "
             f"{kuehlt_ab}/{len(sub)}, Pearson r={r:+.3f} (p={p:.3f})")
        print(f"        Rennen mit Erwaermung trotzdem schneller: "
             f"{erwaermt_aber_schneller}/{erwaermt}")

    print("\nDRITTE AUSBAUSTUFE: gilt das auch fuer Sprint Qualifying "
         "(SS 2023 / SQ 2024)? ...")
    sq_deltas, sq_nass = sprint_quali_scan(SPRINT_QUALI_IDENT.keys())
    print(f"      {sq_deltas['gp'].nunique()} trockene Sprint-Qualifyings, "
         f"{len(sq_nass)} wegen Regen raus: {sq_nass}")
    for seg in ("Q1->Q2", "Q2->Q3"):
        sub = sq_deltas[sq_deltas["segment"] == seg]
        w = wilcoxon(sub["delta_s"])
        je_r = sub.groupby("gp")["delta_s"].median()
        print(f"      {seg}: n={len(sub)}, median={sub['delta_s'].median():+.3f}s, "
             f"positiv={(sub['delta_s'] > 0).mean():.1%}, p={w.pvalue:.2e}, "
             f"Rennen positiv={(je_r > 0).sum()}/{len(je_r)}")

    print("\nGrafiken speichern ...")
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(11, 17),
                                        gridspec_kw={"height_ratios":
                                                     [1, 1.3, 1]})
    fig.patch.set_facecolor(plt.rcParams["figure.facecolor"])
    zeichne_verteilung(ax1, deltas)
    zeichne_konsistenz(ax2, je_rennen)
    zeichne_temperatur(ax3, temp_df)
    fig.tight_layout()
    fig.savefig(OUT / "p43_streckenentwicklung.png", dpi=140)
    plt.close(fig)
    print(f"      {OUT / 'p43_streckenentwicklung.png'}")


if __name__ == "__main__":
    main()
