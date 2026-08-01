"""
P24 - Fahrstil-Clustering: Wer faehrt wie?
==========================================

Unsupervised Learning auf Telemetrie-Features: Bremsverhalten, Gaspedal-Modulation, Lenkaggressivitaet.

Kategorie:   Machine Learning
Niveau:      Profi
Aufwand:     5-6 h
Rollen:      DS
Zusaetzliche Pakete: scikit-learn

WARUM DAS ZAEHLT
Unsupervised auf Zeitreihen ist anspruchsvoll und selten im Portfolio. Das Ergebnis - Fahrertypen - ist inhaltlich sofort diskutierbar.

VORGEHEN
  1. Fuer viele Runden vieler Fahrer Telemetrie-Features aggregieren
  2. Features: Vollgasanteil, Bremsanteil, Anzahl Gaspedal-Modulationen, Coasting
  3. Standardisieren, PCA auf 2 Dimensionen
  4. k-Means und Interpretation der Cluster

GENUTZTE FASTF1-BAUSTEINE
  - Lap.get_car_data
  - Telemetry Throttle/Brake/Speed/nGear
  - sklearn PCA/KMeans

AUSBAUSTUFE
Nutze Dynamic Time Warping auf den rohen Speed-Traces statt aggregierter Features - naeher am tatsaechlichen Fahrverhalten.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import fastf1
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

fastf1.Cache.enable_cache("~/f1_cache")


def style_features(tel):
    t = tel.copy()
    t["dt"] = t["Time"].diff().dt.total_seconds().fillna(0)
    total = t["dt"].sum()
    full = t.loc[t["Throttle"] > 98, "dt"].sum()
    brake = t.loc[t["Brake"].astype(bool), "dt"].sum()
    coast = t.loc[(t["Throttle"] < 5) & (~t["Brake"].astype(bool)), "dt"].sum()
    thr_mod = np.abs(np.diff(t["Throttle"].to_numpy())).sum() / max(total, 1e-6)
    overlap = t.loc[(t["Brake"].astype(bool)) & (t["Throttle"] > 10), "dt"].sum()
    return {
        "Vollgas_%": 100 * full / total,
        "Bremsen_%": 100 * brake / total,
        "Coasting_%": 100 * coast / total,
        "Gas_Modulation": thr_mod,
        "Overlap_%": 100 * overlap / total,
        "Schaltungen": int((t["nGear"].diff() != 0).sum()),
    }


rows = []
for gp in ("Bahrain", "Spain", "Monza", "Suzuka"):
    ses = fastf1.get_session(2024, gp, "Q")
    ses.load()
    for drv in ses.drivers:
        abb = ses.get_driver(drv)["Abbreviation"]
        try:
            tel = ses.laps.pick_drivers(drv).pick_fastest().get_car_data()
        except Exception:
            continue
        if tel is None or tel.empty:
            continue
        f = style_features(tel)
        f.update({"Driver": abb, "GP": gp})
        rows.append(f)

df = pd.DataFrame(rows)
agg = df.groupby("Driver").mean(numeric_only=True)
print(agg.round(2).to_string())

X = StandardScaler().fit_transform(agg)
pcs = PCA(n_components=2).fit_transform(X)
labels = KMeans(n_clusters=3, n_init=20, random_state=0).fit_predict(X)

fig, ax = plt.subplots(figsize=(10, 8))
ax.scatter(pcs[:, 0], pcs[:, 1], c=labels, cmap="Set1", s=140)
for i, name in enumerate(agg.index):
    ax.annotate(name, (pcs[i, 0], pcs[i, 1]),
                xytext=(6, 4), textcoords="offset points")
ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
ax.set_title("Fahrstil-Cluster aus Telemetrie-Features")
plt.tight_layout()
plt.show()

agg["Cluster"] = labels
print("\nCluster-Profile:")
print(agg.groupby("Cluster").mean(numeric_only=True).round(2).to_string())
