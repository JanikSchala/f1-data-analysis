"""
P36 - Rennergebnis- und Podium-Wahrscheinlichkeit vorhersagen
===============================================================

Aus Startplatz, Form und Zuverlaessigkeit: wie wahrscheinlich ist ein Podium, und auf welcher Position landet ein Fahrer voraussichtlich?

Kategorie:   Machine Learning
Niveau:      Profi
Aufwand:     6-7 h
Schwerpunkt: Datenanalyse
Zusaetzliche Pakete: scikit-learn

WARUM DAS LOHNT
P23 sagt die Quali-Position voraus - dieses Projekt geht den Schritt weiter
zum Rennergebnis selbst, mit einer echten Wahrscheinlichkeitsangabe statt
nur einer Punktschaetzung ("wie wahrscheinlich Podium", nicht nur "welcher
Platz"). Startplatz ist in der Formel 1 ein extrem starker Praediktor - der
eigentliche Test ist, ob Form und Zuverlaessigkeit ueber den reinen
Startplatz hinaus etwas beitragen, und ob die Wahrscheinlichkeitsangabe
kalibriert ist (eine Vorhersage "70% Podium" muss auch ueber viele Faelle
in etwa 70% der Zeit eintreffen, sonst ist sie keine echte Wahrscheinlichkeit).

VORGEHEN
  1. Rennergebnisse ueber mehrere Saisons sammeln (Grid, Zielposition, Status)
  2. Features: Grid, rollierende Fahrer-/Team-Form, rollierende DNF-Quote
     (alle um eine Runde verschoben, damit das aktuelle Rennen nicht in die
     eigene Formkurve einfliesst - derselbe Kunstgriff wie in P23)
  3. Zwei Modelle: Podium-Wahrscheinlichkeit (Klassifikation) und
     Zielposition (Regression, nur klassifizierte/gefinishte Ergebnisse)
  4. TimeSeriesSplit-Validierung gegen die Baseline "Startplatz bleibt
     Zielplatz"
  5. Kalibrierung der Podium-Wahrscheinlichkeit pruefen (Reliability-Kurve)

GENUTZTE FASTF1-BAUSTEINE
  - Session.results (GridPosition/Position/Status/Points), konsequent aus
    der FastF1-Session statt Ergast - wie in P23 begruendet
  - sklearn HistGradientBoostingClassifier/Regressor, calibration_curve

AUSBAUSTUFE  [umgesetzt]
Kalibrierung der Podium-Wahrscheinlichkeit: die Vorhersagen in Bins sortiert
(0-10%, 10-20%, ...) und je Bin die tatsaechliche Podium-Quote gegen die
mittlere vorhergesagte Wahrscheinlichkeit verglichen (Brier Score als
Gesamtmass). Ergebnis siehe Docstring-Ergaenzung nach dem ersten echten Lauf
weiter unten - Zahlen werden erst nach der Verifikation gegen echte Daten
eingetragen, nicht vorher behauptet.

ERGEBNIS (Saisons 2022-2024, 1359 Fahrer-Rennen/68 Rennen, TimeSeriesSplit,
5 Folds; Basisrate Podium 15.0%, DNF-Quote 20.7%):

Podium-Klassifikation: ROC-AUC 0.905 - deutlich ueber 0.5 (Zufall), das
Modell trennt Podium/Nicht-Podium klar. Brier Score 0.093 gegen 0.127 fuer
die Baseline (immer die Basisrate 15%) - eine echte Verbesserung. Die
Grid<=3-Heuristik als zweite, haerter geschnittene Baseline kommt auf 0.884
Accuracy - hoch, aber nicht direkt mit dem ROC-AUC vergleichbar (andere
Metrik, kein gemeinsamer Schwellenwert); der eigentliche Mehrwert des
Modells ist die abgestufte Wahrscheinlichkeit, nicht ein hartes Ja/Nein.

Permutation Importance ueberrascht gegenueber der Erwartung "Grid dominiert
alles": grid (0.101) fuehrt knapp vor team_form (0.075) - die aktuelle
Team-/Auto-Form traegt fast so viel zur Podium-Vorhersage bei wie der
Startplatz selbst. driver_form (0.006) und dnf_rate (-0.004, im Rahmen des
Rauschens) tragen dagegen praktisch nichts bei - plausibel, weil grid
bereits die aktuelle Form des Fahrer-Auto-Pakets fuer DIESES Wochenende
einpreist, waehrend driver_form nur die (schwaechere) individuelle
Fahrerkomponente separat erfasst.

Kalibrierung ist der ehrlichste Teil des Befunds: das Modell ist NICHT
gleichmaessig gut kalibriert, sondern systematisch ueberkonfident im oberen
Wahrscheinlichkeitsbereich - bei vorhergesagten 45-95% Podium-Chance liegt
die tatsaechliche Quote durchgaengig 10-30 Prozentpunkte darunter (z.B.
vorhergesagt 85.5%, beobachtet nur 53.6%). Nur am unteren Ende (<15%
vorhergesagt) ist die Kalibrierung sauber. Mit ~270 Testfaellen je Bin-Satz
sind einzelne Bins duenn besetzt, aber der Trend ist ueber alle oberen Bins
konsistent genug, um kein reines Rauschen zu sein: das Modell tut sich
schwer, "sehr wahrscheinlich Podium" von "ziemlich wahrscheinlich Podium"
zuverlaessig zu unterscheiden - fuer eine ehrliche Wahrscheinlichkeitsangabe
braeuchte es entweder mehr Daten oder eine explizite Kalibrierungsstufe
(Platt-Scaling/Isotonic Regression), die hier bewusst nicht nachtraeglich
draufgesetzt wird, um das ungeschoente Rohergebnis zu zeigen.

Positions-Regression (nur gefinishte Ergebnisse): MAE 2.42 Positionen gegen
Baseline (Grid=Ziel) 2.87 - eine spuerbare, nicht nur kosmetische
Verbesserung (~16% relativ). Anders als bei der urspruenglichen Annahme
"Startplatz erklaert fast alles" tragen Form und Team-Staerke hier
sichtbar zur Genauigkeit bei, nicht nur zur Feinabstimmung.
"""
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
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)
from sklearn.inspection import permutation_importance
from sklearn.metrics import brier_score_loss, mean_absolute_error, roc_auc_score
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
    """VORGEHEN 1: Grid/Ziel/Status je Fahrer und Rennen, direkt aus der
    FastF1-Session (Session.results) - wie in P23 begruendet, konsequent
    ohne Ergast fuer Daten, die FastF1 selbst schon liefert."""
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
                    "status": r["Status"], "points": r["Points"],
                })
    df = pd.DataFrame(rows)
    df["dnf"] = ~df["status"].isin(["Finished", "Lapped"])
    df["podium"] = (df["position"] <= 3) & ~df["dnf"]
    return df


def form_anhaengen(races: pd.DataFrame, fenster: int = FORM_FENSTER) -> pd.DataFrame:
    """VORGEHEN 2: rollierende Fahrer-/Team-Form (mittlere Zielposition) und
    rollierende DNF-Quote je Fahrer, alle um ein Rennen verschoben (shift(1)) -
    derselbe Kunstgriff wie P23s team_form, damit das aktuelle Rennen nicht
    in die eigene Formkurve einfliesst."""
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
    """VORGEHEN 5/AUSBAUSTUFE: Reliability-Kurve - vorhergesagte gegen
    tatsaechliche Podium-Quote je Wahrscheinlichkeits-Bin."""
    beob, vorh = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy="uniform")
    return pd.DataFrame({"vorhergesagt": vorh, "beobachtet": beob})


def zeichne_modellvergleich(ax, mae: float, mae_baseline: float,
                            auc: float, acc_baseline: float) -> None:
    """VORGEHEN 4."""
    labels = ["Position\n(Modell)", "Position\n(Grid=Ziel)"]
    werte = [mae, mae_baseline]
    ax.bar(labels, werte, color=[SERIEN[0], MUTED], width=0.5)
    for i, v in enumerate(werte):
        ax.text(i, v + 0.05, f"{v:.2f}", ha="center", color=FG, fontsize=9)
    ax.set_ylabel("MAE [Positionen]")
    ax.set_title(f"Positions-Regression: ROC-AUC Podium {auc:.3f} "
                f"(Baseline-Accuracy {acc_baseline:.3f})", loc="left",
                color=FG, fontsize=12, pad=10)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="y", alpha=0.3, linewidth=0.8, color=GRID)
    ax.set_axisbelow(True)


def zeichne_kalibrierung(ax, kal: pd.DataFrame, brier: float,
                         brier_baseline: float) -> None:
    """AUSBAUSTUFE."""
    ax.plot([0, 1], [0, 1], color=MUTED, lw=1, ls="--", label="perfekt kalibriert")
    ax.plot(kal["vorhergesagt"], kal["beobachtet"], marker="o", ms=6,
           color=SERIEN[1], lw=2, label="Modell")
    ax.set_xlabel("Vorhergesagte Podium-Wahrscheinlichkeit")
    ax.set_ylabel("Beobachtete Podium-Quote")
    ax.set_title(f"Kalibrierung: Brier {brier:.3f} (Baseline {brier_baseline:.3f})",
                loc="left", color=FG, fontsize=12, pad=10)
    ax.legend(loc="upper left", frameon=False, labelcolor=FG, fontsize=9)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(alpha=0.3, linewidth=0.8, color=GRID)
    ax.set_axisbelow(True)


def zeichne_importance(ax, imp_s: pd.Series) -> None:
    """VORGEHEN 4/5."""
    farben = [SERIEN[0] if v == imp_s.max() else MUTED for v in imp_s.to_numpy()]
    ax.barh(imp_s.index, imp_s.to_numpy(), color=farben, height=0.6)
    ax.set_xlabel("Permutation Importance (AUC-Abfall)")
    ax.set_title("Feature Importance: Podium-Klassifikation", loc="left",
                color=FG, fontsize=12, pad=10)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="x", alpha=0.3, linewidth=0.8, color=GRID)
    ax.set_axisbelow(True)


def main():
    f1lab.enable_cache()

    print(f"[1/5] Rennergebnisse {JAHRE[0]}-{JAHRE[-1]} sammeln (VORGEHEN 1) ...")
    races = sammle_rennen(range(JAHRE[0], JAHRE[-1] + 1))
    print(f"      {len(races)} Fahrer-Rennen, "
         f"{races[['season', 'round']].drop_duplicates().shape[0]} Rennen, "
         f"{races['dnf'].mean():.1%} DNF-Quote, {races['podium'].mean():.1%} "
         f"Podium-Quote")

    print("\n[2/5] Form- und Zuverlaessigkeits-Features (VORGEHEN 2) ...")
    races = form_anhaengen(races)
    races = races.dropna(subset=["grid", "position"])
    print(f"      {len(races)} auswertbare Zeilen nach Filter auf gueltigen "
         f"Grid/Zielwert")

    feat = ["grid", "driver_form", "team_form", "dnf_rate"]
    races = races.sort_values(["season", "round"]).reset_index(drop=True)
    X = races[feat]
    y_podium = races["podium"].astype(int)

    print("\n[3/5] TimeSeriesSplit-Validierung: Podium-Klassifikation "
         "(VORGEHEN 3-4) ...")
    cv = TimeSeriesSplit(n_splits=5)
    auc_scores, brier_scores, letzter_test_idx = [], [], None
    alle_y_true, alle_y_prob = [], []
    for tr, te in cv.split(X):
        m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06)
        m.fit(X.iloc[tr], y_podium.iloc[tr])
        prob = m.predict_proba(X.iloc[te])[:, 1]
        auc_scores.append(roc_auc_score(y_podium.iloc[te], prob))
        brier_scores.append(brier_score_loss(y_podium.iloc[te], prob))
        alle_y_true.append(y_podium.iloc[te].to_numpy())
        alle_y_prob.append(prob)
        letzter_test_idx = te

    y_true_ges = np.concatenate(alle_y_true)
    y_prob_ges = np.concatenate(alle_y_prob)
    basisrate = y_podium.mean()
    brier_baseline = brier_score_loss(y_true_ges, np.full(len(y_true_ges), basisrate))
    baseline_pred = (races["grid"] <= 3).astype(int)
    acc_baseline = (baseline_pred == y_podium).mean()

    print(f"      ROC-AUC: {np.mean(auc_scores):.3f} (je Fold: "
         f"{[round(v, 3) for v in auc_scores]})")
    print(f"      Brier Score: {np.mean(brier_scores):.3f} gegen Baseline "
         f"(immer Basisrate {basisrate:.1%}): {brier_baseline:.3f}")
    print(f"      Baseline 'Grid<=3': Accuracy {acc_baseline:.3f}")

    print("\n[4/5] TimeSeriesSplit-Validierung: Positions-Regression "
         "(nur gefinishte Ergebnisse) ...")
    fertig = races[~races["dnf"]].copy()
    Xf = fertig[feat]
    yf = fertig["position"]
    mae_scores, mae_baseline_scores = [], []
    for tr, te in cv.split(Xf):
        m = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.06)
        m.fit(Xf.iloc[tr], yf.iloc[tr])
        pred = m.predict(Xf.iloc[te])
        mae_scores.append(mean_absolute_error(yf.iloc[te], pred))
        mae_baseline_scores.append(mean_absolute_error(
            yf.iloc[te], fertig["grid"].iloc[te]))
    print(f"      MAE Modell: {np.mean(mae_scores):.2f} Positionen")
    print(f"      MAE Baseline (Grid=Ziel): {np.mean(mae_baseline_scores):.2f} "
         "Positionen")

    print("\n[5/5] AUSBAUSTUFE: Kalibrierung und Feature Importance ...")
    kal = kalibrierung(y_true_ges, y_prob_ges)
    print(kal.round(3).to_string(index=False))

    m_final = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06)
    tr_final = np.setdiff1d(np.arange(len(X)), letzter_test_idx)
    m_final.fit(X.iloc[tr_final], y_podium.iloc[tr_final])
    imp = permutation_importance(m_final, X.iloc[letzter_test_idx],
                                 y_podium.iloc[letzter_test_idx],
                                 scoring="roc_auc", n_repeats=20, random_state=7)
    imp_s = pd.Series(imp.importances_mean, index=feat).sort_values()
    print("\n      Permutation Importance (AUC-Abfall):")
    print(imp_s.round(3).to_string())

    print("\nGrafik ...")
    fig, ax = plt.subplots(1, 3, figsize=(19, 6))
    zeichne_modellvergleich(ax[0], np.mean(mae_scores), np.mean(mae_baseline_scores),
                            np.mean(auc_scores), acc_baseline)
    zeichne_kalibrierung(ax[1], kal, np.mean(brier_scores), brier_baseline)
    zeichne_importance(ax[2], imp_s)
    fig.suptitle("Rennergebnis- und Podium-Vorhersage", x=0.06, ha="left",
                fontsize=15, color=FG, y=1.03)
    plt.tight_layout()
    path = OUT / "renn_podium_vorhersage.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"\n      -> {path}")


if __name__ == "__main__":
    main()
