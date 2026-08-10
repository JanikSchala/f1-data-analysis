"""
P23 - Qualifying-Ergebnis vorhersagen (Machine Learning)
========================================================

Gradient Boosting auf Features aus Freiem Training, Vorjahresform und Streckencharakteristik - Ziel: Quali-Position.

Kategorie:   Machine Learning
Niveau:      Profi
Aufwand:     6-8 h
Schwerpunkt: Datenanalyse
Zusaetzliche Pakete: scikit-learn

WARUM DAS LOHNT
Ein ML-Projekt mit sauberer zeitlicher Validierung (kein Data Leakage!) ist das, was Data-Science-Rollen sehen wollen. Der Domaenenkontext macht es einpraegsam.

VORGEHEN
  1. Trainingsdaten ueber mehrere Saisons sammeln
  2. Features: beste FP-Runde, Long-Run-Pace, Vorjahresposition, Teamform
  3. Target: Quali-Position (oder Quali-Zeit als Prozent zur Pole)
  4. TimeSeriesSplit statt zufaelligem Split - Rennen kommen chronologisch
  5. Feature Importance interpretieren

GENUTZTE FASTF1-BAUSTEINE
  - Session.laps FP1-FP3
  - Session.results
  - Laps.pick_quicklaps
  - sklearn

AUSBAUSTUFE  [umgesetzt]
Ersetze die Zielgroesse durch die Quali-Zeit relativ zur Pole und vergleiche,
ob Regression auf Zeit besser funktioniert als auf Position.

Zwei der fuenf VORGEHEN-Punkte fehlten im Code der Vorlage komplett:

VORGEHEN 2 nannte "Vorjahresposition" und "Teamform" als Features, gebaut
wurden nur die FP-Kennzahlen. Beide ergaenzt: Vorjahresposition ist die
Quali-Platzierung desselben Fahrers beim selben Event (ueber den
Eventnamen gematcht, nicht die Rundennummer - die verschiebt sich zwischen
Saisons) ein Jahr zuvor, fehlt fuer Rookies und neue Strecken (NaN bleibt
NaN - HistGradientBoostingRegressor kann das nativ, kein Imputer noetig).
Teamform ist der rollierende Mittelwert der Team-Quali-Positionen aus den
letzten 3 Rennen DERSELBEN Saison, um einen Schritt verschoben (shift(1)),
damit das aktuelle Rennen nicht in die eigene Formkurve einfliesst.

VORGEHEN 5 versprach Feature-Importance-Interpretation, der Code der
Vorlage druckte nur die MAE. Nachgeholt - aber nicht mit
`.feature_importances_`: HistGradientBoostingRegressor hat dieses Attribut
schlicht nicht (getestet: hasattr() liefert False), anders als
GradientBoostingRegressor oder RandomForestRegressor. Permutation Importance
(sklearn.inspection) auf dem letzten Testfold verwendet.

AUSBAUSTUFE: Zwei Modelle, gleiche Features, unterschiedliches Target -
Position direkt vs. Quali-Zeit relativ zur Pole (Zielsegment Q3/Q2/Q1 in
dieser Prioritaet, je nachdem wie weit ein Fahrer kam). Fuer einen fairen
Vergleich wird die Zeit-Vorhersage je Session in eine Rang-Vorhersage
umgerechnet (Platz 1 = kleinste vorhergesagte Zeit) und in denselben
Positions-MAE umgerechnet wie das direkte Modell.

Der erste Lauf hatte einen echten Data-Leak: das Feature-Filter
`c.endswith("_rel")` fing versehentlich die Zielspalte "zeit_rel" mit ein,
weil ihr Name zufaellig auch auf "_rel" endet - das Positions-Modell hatte
damit im Training Zugriff auf (fast) die Antwort und die Permutation
Importance zeigte "zeit_rel" folgerichtig mit 0.87 als dominantes
"Feature". Ergebnis danach unbrauchbar schoen (MAE 2.57). Gefixt durch ein
praeziseres Filter (nur "FP*_rel"-Spalten). Mit sauberer Trennung dreht
sich der AUSBAUSTUFE-Befund sogar um: Position direkt vorherzusagen (MAE
3.52) schlaegt den Umweg ueber die Zeit-Vorhersage (MAE 5.20, schlechter
als die Baseline von 4.98) - das Gegenteil der urspruenglich vermuteten
Richtung. Plausibler Grund: kleine Fehler in der Zeitvorhersage kippen bei
eng beieinanderliegenden Fahrern (typisch im Mittelfeld) leicht die
Reihenfolge, waehrend das direkte Rang-Modell genau darauf optimiert statt
auf einen Zeitwert, aus dem der Rang erst nachtraeglich rekonstruiert wird.
Permutation Importance (ebenfalls neu berechnet, ohne Leak): team_form ist
das mit Abstand wichtigste Feature (0.33), vor der besten FP2-Runde (0.07) -
die aktuelle Team-Form draengt die einzelne Trainingsrunde deutlich in den
Hintergrund.

ZWEITE AUSBAUSTUFE: schlaegt ein kleines neuronales Netz (sklearn
MLPRegressor, kein zusaetzlicher Dependency) das Gradient-Boosting-Modell?
Erwartung vorab: nein - bei ~1300 Zeilen und 6-8 Features ist das klassische
Terrain fuer baumbasierte Modelle, nicht fuer ein Netz. Zwei methodische
Unterschiede muessen dafuer fair gemacht werden, sonst waere der Vergleich
unehrlich zugunsten des Baums: HistGradientBoostingRegressor behandelt NaN
nativ als eigenen Split (fehlender Vorjahreswert ist selbst ein Signal -
Rookie oder neue Strecke), ein MLP kann das nicht und braucht eine explizite
Imputation (Median), die dieses Signal wegglaettet. Und Baeume sind
skaleninvariant, ein MLP nicht - StandardScaler vor dem Netz, nicht vor dem
Baum. Beides steckt in ``mlp_pipeline()``.

Ergebnis bestaetigt die Erwartung nur mit Einschraenkung: MLP MAE 3.66 gegen
Baum-MAE 3.52 - das Netz liegt tatsaechlich dahinter, aber nur knapp
(+0.14 Positionen), nicht abgeschlagen, und weit vor der Zeit->Rang-Variante
oben (5.20) sowie der Baseline (4.98). Mit derselben Datenmenge und
denselben Features holt ein kleines, sorgfaeltig regularisiertes Netz
(early_stopping, zwei kleine Hidden-Layer) fast an den baumbasierten
Vorsprung heran, den Feature-Filter und NaN-Handling dem Baum eigentlich
verschaffen sollten - ein differenzierterer Befund als "NN verliert klar
gegen Baum auf kleinen Tabellendaten", eher "beide sind in derselben
Groessenordnung, der Baum gewinnt knapp".
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
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

import f1lab
from f1lab.design import FG, GRID, MUTED, SERIEN, matplotlib_stil

warnings.filterwarnings("ignore")
fastf1.set_log_level("ERROR")

OUT = Path(__file__).parent / "out"
OUT.mkdir(exist_ok=True)

JAHRE = (2022, 2023, 2024)
TEAMFORM_FENSTER = 3

plt.rcParams.update(matplotlib_stil())


def quali_zeit(row: pd.Series) -> float:
    """Zeit aus dem am weitesten erreichten Segment - das bestimmt die
    Platzierung, nicht zwingend die insgesamt schnellste Runde."""
    for seg in ("Q3", "Q2", "Q1"):
        v = row[seg]
        if pd.notna(v):
            return v.total_seconds()
    return np.nan


def sammle_quali(jahre: range) -> pd.DataFrame:
    """Reine Quali-Ergebnisse ueber alle Jahre - Grundlage fuer
    Vorjahresposition und Teamform, kein FP-Laden noetig."""
    rows = []
    for jahr in jahre:
        sched = fastf1.get_event_schedule(jahr, include_testing=False)
        for _, event in sched.iterrows():
            rnd = int(event["RoundNumber"])
            try:
                q = f1lab.load(jahr, rnd, "Q", telemetry=False, weather=False,
                              messages=False)
            except Exception:
                continue
            res = q.results
            if res.empty:
                continue
            zeiten = res.apply(quali_zeit, axis=1)
            pole = zeiten.min()
            for (_, r), zt in zip(res.iterrows(), zeiten, strict=True):
                rows.append({
                    "season": jahr, "round": rnd, "event": event["EventName"],
                    "driver": r["Abbreviation"], "team": r["TeamName"],
                    "position": r["Position"],
                    "zeit_rel": zt / pole if pd.notna(zt) and pole else np.nan,
                })
    return pd.DataFrame(rows)


def teamform_anhaengen(quali: pd.DataFrame) -> pd.DataFrame:
    """Rollierender Team-Formindex, um eine Runde verschoben (kein Blick
    auf das eigene Rennen)."""
    quali = quali.sort_values(["season", "round"]).copy()
    je_team_rennen = (quali.groupby(["season", "round", "team"])["position"]
                      .mean().reset_index())
    je_team_rennen["team_form"] = (
        je_team_rennen.groupby(["season", "team"])["position"]
        .transform(lambda s: s.shift(1).rolling(TEAMFORM_FENSTER, min_periods=1).mean()))
    return quali.merge(je_team_rennen[["season", "round", "team", "team_form"]],
                       on=["season", "round", "team"], how="left")


def vorjahr_anhaengen(quali: pd.DataFrame) -> pd.DataFrame:
    vorjahr = quali[["season", "event", "driver", "position"]].copy()
    vorjahr["season"] = vorjahr["season"] + 1
    vorjahr = vorjahr.rename(columns={"position": "vorjahr_position"})
    return quali.merge(vorjahr, on=["season", "event", "driver"], how="left")


def fp_features(jahr: int, rnd: int) -> pd.DataFrame:
    """VORGEHEN 2, FP-Teil: beste Runde und Long-Run-Pace je FP-Session."""
    rows = {}
    for fp in ("FP1", "FP2", "FP3"):
        try:
            s = f1lab.load(jahr, rnd, fp, telemetry=False, weather=False,
                          messages=False)
        except Exception:
            continue
        laps = s.laps.pick_accurate().pick_wo_box()
        if laps.empty:
            continue
        best = laps.groupby("Driver")["LapTime"].min().dt.total_seconds()
        longrun = (laps.pick_quicklaps(1.10).groupby("Driver")["LapTime"]
                  .median().dt.total_seconds())
        for d in best.index:
            rows.setdefault(d, {})[f"{fp}_best"] = best[d]
            rows.setdefault(d, {})[f"{fp}_long"] = longrun.get(d, np.nan)
    return pd.DataFrame(rows).T


def datensatz_bauen(quali_mit_form: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for (jahr, rnd), _g in quali_mit_form.groupby(["season", "round"]):
        fp = fp_features(int(jahr), int(rnd))
        if fp.empty:
            continue
        fp.index.name = "driver"
        frames.append(fp.reset_index().assign(season=jahr, round=rnd))
    fp_alle = pd.concat(frames, ignore_index=True)
    data = quali_mit_form.merge(fp_alle, on=["season", "round", "driver"],
                                how="inner")

    for c in [c for c in data.columns if c.endswith(("_best", "_long"))]:
        data[c + "_rel"] = data.groupby(["season", "round"])[c].transform(
            lambda s: s / s.min())

    return data.dropna(subset=["position"])


def positions_mae_aus_zeit(data: pd.DataFrame, pred_zeit: np.ndarray) -> float:
    """AUSBAUSTUFE: vorhergesagte Zeit je Session in einen Rang uebersetzen,
    damit beide Modelle in derselben Einheit verglichen werden."""
    tmp = data[["season", "round", "position"]].copy()
    tmp["pred_zeit"] = pred_zeit
    tmp["pred_rang"] = tmp.groupby(["season", "round"])["pred_zeit"].rank()
    return mean_absolute_error(tmp["position"], tmp["pred_rang"])


def mlp_pipeline(seed: int = 7) -> Pipeline:
    """ZWEITE AUSBAUSTUFE: Imputation (Median) + Skalierung + kleines Netz.

    Anders als HistGradientBoostingRegressor braucht ein MLP beides: NaN
    muessen weg (der Baum haette sie nativ als Split genutzt), und die
    Features muessen skaliert sein (der Baum ist skaleninvariant, ein
    Gradientenverfahren nicht). Zwei kleine Hidden-Layer, weil bei ~1300
    Zeilen und 6-8 Features ein groesseres Netz nur auswendig lernt.
    """
    return make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        MLPRegressor(hidden_layer_sizes=(32, 16), max_iter=2000,
                     early_stopping=True, random_state=seed),
    )


def main():
    f1lab.enable_cache()

    print(f"[1/4] Quali-Ergebnisse {JAHRE[0] - 1}-{JAHRE[-1]} sammeln "
         f"(VORGEHEN 2: Vorjahresposition/Teamform-Basis) ...")
    quali = sammle_quali(range(JAHRE[0] - 1, JAHRE[-1] + 1))
    quali = teamform_anhaengen(quali)
    quali = vorjahr_anhaengen(quali)
    quali = quali[quali["season"].isin(JAHRE)]
    print(f"      {len(quali)} Quali-Ergebnisse, "
         f"{quali['vorjahr_position'].notna().mean():.0%} mit Vorjahreswert, "
         f"{quali['team_form'].notna().mean():.0%} mit Teamform")

    print("\n[2/4] Freies Training je Wochenende laden (VORGEHEN 1-2) ...")
    data = datensatz_bauen(quali)
    print(f"      {len(data)} Fahrer-Wochenenden, "
         f"{data[['season', 'round']].drop_duplicates().shape[0]} Events")

    # Nur die FP-Kennzahlen ("FP1_best_rel" etc.) - "zeit_rel" ist das
    # Ziel des zweiten Modells und darf hier nicht als Feature landen,
    # sonst lernt das Positions-Modell nur noch diese eine Spalte (siehe
    # Docstring, das ist beim ersten Lauf tatsaechlich passiert).
    feat = ([c for c in data.columns if c.startswith("FP") and c.endswith("_rel")]
           + ["vorjahr_position", "team_form"])
    data = data.sort_values(["season", "round"]).reset_index(drop=True)
    X = data[feat]
    y_pos = data["position"]
    y_zeit = data["zeit_rel"]

    print("\n[3/4] TimeSeriesSplit-Validierung (VORGEHEN 3-4, AUSBAUSTUFE) ...")
    cv = TimeSeriesSplit(n_splits=5)
    mae_pos, mae_zeit_als_position, mae_mlp = [], [], []
    letzter_test_idx = None
    for tr, te in cv.split(X):
        m_pos = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.06)
        m_pos.fit(X.iloc[tr], y_pos.iloc[tr])
        mae_pos.append(mean_absolute_error(y_pos.iloc[te], m_pos.predict(X.iloc[te])))

        m_zeit = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.06)
        gueltig = y_zeit.iloc[tr].notna()
        m_zeit.fit(X.iloc[tr][gueltig], y_zeit.iloc[tr][gueltig])
        pred_zeit = m_zeit.predict(X.iloc[te])
        mae_zeit_als_position.append(
            positions_mae_aus_zeit(data.iloc[te], pred_zeit))

        m_mlp = mlp_pipeline()
        m_mlp.fit(X.iloc[tr], y_pos.iloc[tr])
        mae_mlp.append(mean_absolute_error(y_pos.iloc[te], m_mlp.predict(X.iloc[te])))

        letzter_test_idx = te

    print(f"      Target=Position (Baum):  MAE = {np.mean(mae_pos):.2f} "
         f"Positionen (je Fold: {[round(v, 2) for v in mae_pos]})")
    print(f"      Target=Zeit->Rang (Baum): MAE = "
         f"{np.mean(mae_zeit_als_position):.2f} Positionen (je Fold: "
         f"{[round(v, 2) for v in mae_zeit_als_position]})")
    print(f"      Target=Position (MLP):   MAE = {np.mean(mae_mlp):.2f} "
         f"Positionen (je Fold: {[round(v, 2) for v in mae_mlp]})")
    baseline = mean_absolute_error(y_pos, np.full(len(y_pos), y_pos.median()))
    print(f"      Baseline (immer Median-Position {y_pos.median():.0f}): "
         f"{baseline:.2f}")

    print("\n[4/4] Permutation Importance auf letztem Testfold (VORGEHEN 5) ...")
    m_final = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.06)
    tr_final = np.setdiff1d(np.arange(len(X)), letzter_test_idx)
    m_final.fit(X.iloc[tr_final], y_pos.iloc[tr_final])
    imp = permutation_importance(m_final, X.iloc[letzter_test_idx],
                                 y_pos.iloc[letzter_test_idx],
                                 n_repeats=20, random_state=7)
    imp_s = pd.Series(imp.importances_mean, index=feat).sort_values()
    print(imp_s.round(3).to_string())

    print("\nGrafik ...")
    fig, ax = plt.subplots(1, 2, figsize=(15, 6), gridspec_kw={"width_ratios": [1, 1.2]})
    ax[0].bar(["Position\n(Baum)", "Zeit->Rang\n(Baum)", "Position\n(MLP)"],
             [np.mean(mae_pos), np.mean(mae_zeit_als_position), np.mean(mae_mlp)],
             color=[SERIEN[0], SERIEN[1], SERIEN[2]], width=0.5)
    ax[0].axhline(baseline, color=MUTED, lw=1, ls="--")
    ax[0].text(2.4, baseline, f" Baseline {baseline:.1f}", color=MUTED, fontsize=9,
              va="center")
    ax[0].set_ylabel("MAE [Positionen]")
    ax[0].set_title("AUSBAUSTUFE: Zielgroesse und Modell im Vergleich", loc="left",
                   color=FG, fontsize=12, pad=10)
    for side in ("top", "right"):
        ax[0].spines[side].set_visible(False)
    ax[0].grid(axis="y", alpha=0.3, linewidth=0.8, color=GRID)
    ax[0].set_axisbelow(True)

    farben = [SERIEN[0] if v == imp_s.max() else MUTED for v in imp_s.to_numpy()]
    ax[1].barh(imp_s.index, imp_s.to_numpy(), color=farben, height=0.6)
    ax[1].set_xlabel("Permutation Importance (MAE-Anstieg)")
    ax[1].set_title("Feature Importance (VORGEHEN 5)", loc="left", color=FG,
                   fontsize=12, pad=10)
    for side in ("top", "right"):
        ax[1].spines[side].set_visible(False)
    ax[1].grid(axis="x", alpha=0.3, linewidth=0.8, color=GRID)
    ax[1].set_axisbelow(True)

    fig.suptitle("Qualifying-Vorhersage: Modellvergleich und Feature Importance",
                x=0.09, ha="left", fontsize=15, color=FG, y=1.02)
    plt.tight_layout()
    path = OUT / "quali_vorhersage.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"\n      -> {path}")


if __name__ == "__main__":
    main()
