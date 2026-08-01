"""
P23 - Qualifying-Ergebnis vorhersagen (Machine Learning)
========================================================

Gradient Boosting auf Features aus Freiem Training, Vorjahresform und Streckencharakteristik - Ziel: Quali-Position.

Kategorie:   Machine Learning
Niveau:      Profi
Aufwand:     6-8 h
Rollen:      DS
Zusaetzliche Pakete: scikit-learn

WARUM DAS ZAEHLT
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

AUSBAUSTUFE
Ersetze die Zielgroesse durch die Quali-Zeit relativ zur Pole und vergleiche, ob Regression auf Zeit besser funktioniert als auf Position.
"""

import numpy as np
import pandas as pd
import fastf1
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error

fastf1.Cache.enable_cache("~/f1_cache")


def weekend_features(year, rnd):
    rows = {}
    for fp in ("FP1", "FP2", "FP3"):
        try:
            s = fastf1.get_session(year, rnd, fp)
            s.load(telemetry=False, weather=False, messages=False)
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

    try:
        q = fastf1.get_session(year, rnd, "Q")
        q.load(telemetry=False, weather=False, messages=False)
    except Exception:
        return pd.DataFrame()

    res = q.results.set_index("Abbreviation")
    df = pd.DataFrame(rows).T
    df["Target"] = res["Position"]
    df["Team"] = res["TeamName"]
    df["Season"], df["Round"] = year, rnd
    return df.dropna(subset=["Target"])


frames = []
for year in (2022, 2023, 2024):
    sched = fastf1.get_event_schedule(year, include_testing=False)
    for rnd in sched["RoundNumber"]:
        f = weekend_features(year, int(rnd))
        if not f.empty:
            frames.append(f)

data = pd.concat(frames).reset_index(names="Driver")

# Relative Features statt absoluter Zeiten -> streckenunabhaengig
for c in [c for c in data.columns if c.endswith(("_best", "_long"))]:
    data[c + "_rel"] = data.groupby(["Season", "Round"])[c].transform(
        lambda s: s / s.min())

feat = [c for c in data.columns if c.endswith("_rel")]
data = data.sort_values(["Season", "Round"])
X, y = data[feat], data["Target"]

cv = TimeSeriesSplit(n_splits=5)
scores = []
for tr, te in cv.split(X):
    m = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.06)
    m.fit(X.iloc[tr], y.iloc[tr])
    pred = m.predict(X.iloc[te])
    scores.append(mean_absolute_error(y.iloc[te], pred))

print(f"MAE ueber {len(scores)} Folds: {np.mean(scores):.2f} Positionen")
print("Baseline (immer P10):",
      round(mean_absolute_error(y, np.full(len(y), 10.0)), 2))
