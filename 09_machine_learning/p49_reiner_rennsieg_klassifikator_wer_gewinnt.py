"""sagt die Sieg-Wahrscheinlichkeit (nicht nur Podium) per Gradient Boosting aus Startplatz, Form und DNF-Quote vorher und vergleicht die Feature-Importance mit P36s Podium-Modell"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")                      # kein Fenster, nur Dateien

import fastf1
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit

import f1lab
from f1lab.design import FG, GRID, MUTED, SERIEN, matplotlib_stil

warnings.filterwarnings("ignore")
fastf1.set_log_level("ERROR")

OUT = Path(__file__).parent / "out"
OUT.mkdir(exist_ok=True)

JAHRE = (2022, 2023, 2024)
FORM_FENSTER = 5

plt.rcParams.update(matplotlib_stil())


def sammle_rennen(jahre: range) -> pd.DataFrame:
    """Grid/Ziel/Status je Fahrer und Rennen, direkt aus der FastF1-Session
    (Session.results) - dieselbe Sammlung wie P36, hier zusaetzlich mit der
    Sieg-Spalte."""
    rows = []
    for jahr in jahre:
        sched = fastf1.get_event_schedule(jahr, include_testing=False)
        for _, event in sched.iterrows():
            rnd = int(event["RoundNumber"])
            try:
                ses = f1lab.load(jahr, rnd, "R", telemetry=False, weather=False,
                                 messages=False)
            except Exception:
                continue
            res = ses.results
            if res.empty:
                continue
            for _, r in res.iterrows():
                rows.append({
                    "season": jahr, "round": rnd, "event": event["EventName"],
                    "driver": r["Abbreviation"], "team": r["TeamName"],
                    "grid": r["GridPosition"], "position": r["Position"],
                    "status": r["Status"],
                })
    df = pd.DataFrame(rows)
    df["dnf"] = ~df["status"].isin(["Finished", "Lapped"])
    df["sieg"] = (df["position"] == 1) & ~df["dnf"]
    return df


def form_anhaengen(races: pd.DataFrame, fenster: int = FORM_FENSTER) -> pd.DataFrame:
    """rollierende Fahrer-/Team-Form (mittlere Zielposition) und rollierende
    DNF-Quote je Fahrer, alle um ein Rennen verschoben (shift(1)) - identisch
    zu P36s form_anhaengen(), damit beide Modelle auf denselben Features
    stehen und vergleichbar sind."""
    races = races.sort_values(["season", "round"]).copy()

    races["driver_form"] = (races.groupby(["season", "driver"])["position"]
                            .transform(lambda s: s.shift(1).rolling(
                                fenster, min_periods=1).mean()))
    races["dnf_rate"] = (races.groupby(["season", "driver"])["dnf"]
                         .transform(lambda s: s.shift(1).rolling(
                             fenster, min_periods=1).mean().astype(float)))

    je_team_rennen = (races.groupby(["season", "round", "team"])["position"]
                      .mean().reset_index())
    je_team_rennen["team_form"] = (
        je_team_rennen.groupby(["season", "team"])["position"]
        .transform(lambda s: s.shift(1).rolling(fenster, min_periods=1).mean()))
    return races.merge(je_team_rennen[["season", "round", "team", "team_form"]],
                       on=["season", "round", "team"], how="left")


def kalibrierung(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10
                 ) -> pd.DataFrame:
    beob, vorh = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy="uniform")
    return pd.DataFrame({"vorhergesagt": vorh, "beobachtet": beob})


def zeichne_kalibrierung(ax, kal: pd.DataFrame, brier: float,
                         brier_baseline: float) -> None:
    ax.plot([0, 1], [0, 1], color=MUTED, lw=1, ls="--", label="perfekt kalibriert")
    ax.plot(kal["vorhergesagt"], kal["beobachtet"], marker="o", ms=6,
           color=SERIEN[1], lw=2, label="Modell")
    ax.set_xlabel("Vorhergesagte Sieg-Wahrscheinlichkeit")
    ax.set_ylabel("Beobachtete Sieg-Quote")
    ax.set_title(f"Kalibrierung: Brier {brier:.3f} (Baseline {brier_baseline:.3f})",
                loc="left", color=FG, fontsize=12, pad=10)
    ax.legend(loc="upper left", frameon=False, labelcolor=FG, fontsize=9)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(alpha=0.3, linewidth=0.8, color=GRID)
    ax.set_axisbelow(True)


def zeichne_importance_vergleich(ax, sieg_imp: pd.Series, podium_imp: pd.Series
                                 ) -> None:
    """stellt die Feature Importance des Sieg-Modells der des Podium-Modells
    gegenueber (dieselben vier Features, andere Zielgroesse)."""
    feats = sieg_imp.index
    x = np.arange(len(feats))
    w = 0.35
    ax.barh(x - w / 2, sieg_imp.to_numpy(), height=w, color=SERIEN[0],
           label="Sieg (P49)")
    ax.barh(x + w / 2, podium_imp.reindex(feats).to_numpy(), height=w,
           color=MUTED, label="Podium (P36)")
    ax.set_yticks(x, feats)
    ax.set_xlabel("Permutation Importance (AUC-Abfall)")
    ax.set_title("Feature Importance: Sieg gegen Podium", loc="left",
                color=FG, fontsize=12, pad=10)
    ax.legend(loc="lower right", frameon=False, labelcolor=FG, fontsize=9)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="x", alpha=0.3, linewidth=0.8, color=GRID)
    ax.set_axisbelow(True)


def zeichne_auc_je_fold(ax, auc_scores: list[float], basisrate: float) -> None:
    x = np.arange(1, len(auc_scores) + 1)
    ax.bar(x, auc_scores, color=SERIEN[0], width=0.5)
    ax.axhline(0.5, color=MUTED, lw=1, ls="--", label="Zufall (AUC 0.5)")
    ax.set_xticks(x)
    ax.set_xlabel("TimeSeriesSplit-Fold")
    ax.set_ylabel("ROC-AUC")
    ax.set_ylim(0.5, 1.0)
    ax.set_title(f"Sieg-Klassifikation je Fold (Basisrate {basisrate:.1%})",
                loc="left", color=FG, fontsize=12, pad=10)
    ax.legend(loc="lower right", frameon=False, labelcolor=FG, fontsize=9)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="y", alpha=0.3, linewidth=0.8, color=GRID)
    ax.set_axisbelow(True)


def main():
    f1lab.enable_cache()

    print(f"[1/4] Rennergebnisse {JAHRE[0]}-{JAHRE[-1]} sammeln, dieselbe "
         "Quelle wie P36 (VORGEHEN 1) ...")
    races = sammle_rennen(range(JAHRE[0], JAHRE[-1] + 1))
    print(f"      {len(races)} Fahrer-Rennen, "
         f"{races[['season', 'round']].drop_duplicates().shape[0]} Rennen, "
         f"{races['sieg'].mean():.1%} Sieg-Quote (P36s Podium-Quote lag "
         "bei ~15%, ein Sieg ist ein deutlich seltenerer Zielwert)")

    print("\n[2/4] Form- und Zuverlaessigkeits-Features (VORGEHEN 2) ...")
    races = form_anhaengen(races)
    races = races.dropna(subset=["grid", "position"])
    print(f"      {len(races)} auswertbare Zeilen")

    feat = ["grid", "driver_form", "team_form", "dnf_rate"]
    races = races.sort_values(["season", "round"]).reset_index(drop=True)
    X = races[feat]
    y = races["sieg"].astype(int)

    print("\n[3/4] TimeSeriesSplit-Validierung: Sieg-Klassifikation "
         "(VORGEHEN 3-4) ...")
    cv = TimeSeriesSplit(n_splits=5)
    auc_scores, brier_scores, letzter_test_idx = [], [], None
    alle_y_true, alle_y_prob = [], []
    for tr, te in cv.split(X):
        m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06)
        m.fit(X.iloc[tr], y.iloc[tr])
        prob = m.predict_proba(X.iloc[te])[:, 1]
        auc_scores.append(roc_auc_score(y.iloc[te], prob))
        brier_scores.append(brier_score_loss(y.iloc[te], prob))
        alle_y_true.append(y.iloc[te].to_numpy())
        alle_y_prob.append(prob)
        letzter_test_idx = te

    y_true_ges = np.concatenate(alle_y_true)
    y_prob_ges = np.concatenate(alle_y_prob)
    basisrate = y.mean()
    brier_baseline = brier_score_loss(y_true_ges, np.full(len(y_true_ges), basisrate))
    baseline_pred_acc = ((races["grid"] == 1).astype(int) == y).mean()

    print(f"      ROC-AUC: {np.mean(auc_scores):.3f} (je Fold: "
         f"{[round(v, 3) for v in auc_scores]})")
    print(f"      Brier Score: {np.mean(brier_scores):.3f} gegen Baseline "
         f"(immer Basisrate {basisrate:.1%}): {brier_baseline:.3f}")
    print(f"      Baseline 'Grid==1 gewinnt': Accuracy {baseline_pred_acc:.3f}")

    print("\n[4/4] AUSBAUSTUFE: Kalibrierung, Feature Importance, Vergleich "
         "mit P36s Podium-Modell ...")
    kal = kalibrierung(y_true_ges, y_prob_ges)
    print(kal.round(3).to_string(index=False))

    m_final = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06)
    tr_final = np.setdiff1d(np.arange(len(X)), letzter_test_idx)
    m_final.fit(X.iloc[tr_final], y.iloc[tr_final])
    imp = permutation_importance(m_final, X.iloc[letzter_test_idx],
                                 y.iloc[letzter_test_idx],
                                 scoring="roc_auc", n_repeats=20, random_state=7)
    sieg_imp = pd.Series(imp.importances_mean, index=feat).sort_values()
    print("\n      Permutation Importance Sieg-Modell (AUC-Abfall):")
    print(sieg_imp.round(3).to_string())

    # P36s Podium-Modell zum Vergleich auf denselben Daten neu trainiert
    # (nicht importiert - siehe Modul-Docstring, jedes Skript laeuft
    # eigenstaendig), identisches Vorgehen, nur die Zielspalte ist anders.
    y_podium = ((races["position"] <= 3) & ~races["dnf"]).astype(int)
    m_podium = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06)
    m_podium.fit(X.iloc[tr_final], y_podium.iloc[tr_final])
    imp_podium = permutation_importance(m_podium, X.iloc[letzter_test_idx],
                                        y_podium.iloc[letzter_test_idx],
                                        scoring="roc_auc", n_repeats=20,
                                        random_state=7)
    podium_imp = pd.Series(imp_podium.importances_mean, index=feat)
    print("\n      Permutation Importance Podium-Modell, zum Vergleich:")
    print(podium_imp.round(3).to_string())

    print("\nGrafik ...")
    fig, ax = plt.subplots(1, 3, figsize=(19, 6))
    zeichne_auc_je_fold(ax[0], auc_scores, basisrate)
    zeichne_kalibrierung(ax[1], kal, np.mean(brier_scores), brier_baseline)
    zeichne_importance_vergleich(ax[2], sieg_imp, podium_imp)
    fig.suptitle("Reiner Rennsieg-Klassifikator", x=0.06, ha="left",
                fontsize=15, color=FG, y=1.03)
    plt.tight_layout()
    path = OUT / "rennsieg_vorhersage.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"\n      -> {path}")


if __name__ == "__main__":
    main()
