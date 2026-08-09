"""
P24 - Fahrstil-Clustering: Wer faehrt wie?
==========================================

Unsupervised Learning auf Telemetrie-Features: Bremsverhalten, Gaspedal-Modulation, Lenkaggressivitaet.

Kategorie:   Machine Learning
Niveau:      Profi
Aufwand:     5-6 h
Schwerpunkt: Datenanalyse
Zusaetzliche Pakete: scikit-learn

WARUM DAS LOHNT
Unsupervised auf Zeitreihen ist anspruchsvoll und die Ergebnisse sind nie eindeutig. Aber Fahrertypen allein aus Telemetrie abzuleiten, ohne dem Modell zu sagen wonach es suchen soll, hat etwas.

VORGEHEN
  1. Fuer viele Runden vieler Fahrer Telemetrie-Features aggregieren
  2. Features: Vollgasanteil, Bremsanteil, Anzahl Gaspedal-Modulationen, Coasting
  3. Standardisieren, PCA auf 2 Dimensionen
  4. k-Means und Interpretation der Cluster

GENUTZTE FASTF1-BAUSTEINE
  - Lap.get_car_data
  - Telemetry Throttle/Brake/Speed/nGear
  - sklearn PCA/KMeans

AUSBAUSTUFE  [umgesetzt]
Nutze Dynamic Time Warping auf den rohen Speed-Traces statt aggregierter
Features - naeher am tatsaechlichen Fahrverhalten.

VORGEHEN 1 sagt "viele Runden", die Vorlage nahm pro Fahrer und Strecke nur
die schnellste - eine einzelne Runde ist ein verrauschter Schaetzer fuer
"Fahrstil". Jetzt gehen alle Runden innerhalb der 107%-Schwelle ein
(im Schnitt 4-5 je Fahrer und Session statt einer), bevor auf Fahrerebene
gemittelt wird.

VORGEHEN 4 nannte "Interpretation der Cluster", die Vorlage druckte nur die
Cluster-Mittelwerte. Ergaenzt: eine Silhouette-Auswahl von k (2-5) statt der
angenommenen 3, und eine Einordnung, welches Feature ein Cluster tatsaechlich
von den anderen trennt (per-Feature-Differenz zum Gesamtmittel, nicht nur
die rohen Cluster-Mittel nebeneinander). Ehrlicher Nebenbefund: die
Silhouette-Werte liegen fuer alle getesteten k zwischen 0.21 und 0.24, dicht
beieinander - es gibt keine natuerliche, klar abgesetzte Clusterstruktur in
den Pedal-Features, nur die am wenigsten schlechte Einteilung.

VORGEHEN 2s "Vollgas_%" (Throttle > 98) hatte einen echten Kalibrierungsbug:
Norris' Throttle-Kanal erreicht in Bahrain 2024 Q auf allen vier Runden nie
mehr als 98-99 (jeder andere Fahrer im Feld: 99-100), die fixe Schwelle
maass ihm dadurch 0% Vollgasanteil zu - eindeutig falsch, seine Rundenzeiten
liegen im normalen Feld. Umgestellt auf eine rundeneigene Schwelle
(Throttle innerhalb 2 Prozentpunkten des Rundenmaximums), robust gegen
Kanaele, die aus welchem Grund auch immer nie exakt 100 erreichen.

AUSBAUSTUFE: Dynamic Time Warping (klassische DP, keine Zusatzbibliothek -
weder dtaidistance noch fastdtw stehen in requirements.txt) mit
Sakoe-Chiba-Band auf die Speed-Spur der schnellsten Runde in Suzuka - der
Strecke mit dem groessten Wechsel aus Highspeed- und Technisch-Passagen im
Sample, am ehesten geeignet, Fahrstil in der reinen Geschwindigkeitsspur
sichtbar zu machen. Die per DTW gefundenen Cluster (hierarchisch, da
k-Means keine vorberechnete Distanzmatrix nimmt) stimmen mit den
Feature-basierten Clustern praktisch nicht ueberein (Adjusted Rand Index
nahe 0, siehe Ausgabe - Zufallsniveau, nicht "teilweise") - ein ehrlicher
Befund: rohe Geschwindigkeitsform und aggregierte Pedal-Statistik erfassen
nicht dasselbe, und keines der beiden Verfahren liefert hier eine
sonderlich robuste Struktur (siehe Silhouette-Werte oben).
"""
from __future__ import annotations

import sys
import warnings
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")                      # kein Fenster, nur Dateien

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import MDS
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler

import f1lab
from f1lab.design import FG, GRID, MUTED, SERIEN, matplotlib_stil

warnings.filterwarnings("ignore")

OUT = Path(__file__).parent / "out"
OUT.mkdir(exist_ok=True)

SEASON = 2024
STRECKEN = ("Bahrain", "Spain", "Monza", "Suzuka")
DTW_STRECKE = "Suzuka"
QUICKLAP_SCHWELLE = 1.07
DTW_PUNKTE, DTW_BAND = 150, 20

plt.rcParams.update(matplotlib_stil())


def style_features(tel: pd.DataFrame) -> dict:
    """VORGEHEN 2."""
    t = tel.copy()
    t["dt"] = t["Time"].diff().dt.total_seconds().fillna(0)
    total = t["dt"].sum()
    # Peak-relativ statt fixer Schwelle (">98"): manche Autos/Sessions
    # erreichen im Throttle-Kanal nie 100, ohne dass der Fahrer weniger
    # Vollgas gefahren waere - siehe Docstring (Norris, Bahrain 2024, Peak
    # 98 statt 100 auf allen vier Runden, mit fixer Schwelle 0% "Vollgas").
    full = t.loc[t["Throttle"] >= t["Throttle"].max() - 2, "dt"].sum()
    brake = t.loc[t["Brake"].astype(bool), "dt"].sum()
    coast = t.loc[(t["Throttle"] < 5) & (~t["Brake"].astype(bool)), "dt"].sum()
    thr_mod = np.abs(np.diff(t["Throttle"].to_numpy())).sum() / max(total, 1e-6)
    overlap = t.loc[(t["Brake"].astype(bool)) & (t["Throttle"] > 10), "dt"].sum()
    return {
        "vollgas_pct": 100 * full / total, "bremsen_pct": 100 * brake / total,
        "coasting_pct": 100 * coast / total, "gas_modulation": thr_mod,
        "overlap_pct": 100 * overlap / total,
        "schaltungen": int((t["nGear"].diff() != 0).sum()),
    }


def sammle_features() -> pd.DataFrame:
    """VORGEHEN 1: alle Quicklaps (nicht nur die schnellste) je Fahrer und
    Strecke, danach auf Fahrerebene gemittelt."""
    rows = []
    for gp in STRECKEN:
        ses = f1lab.load(SEASON, gp, "Q", telemetry=True)
        laps = ses.laps.pick_accurate().pick_quicklaps(QUICKLAP_SCHWELLE)
        for lap in laps.itertuples():
            try:
                tel = ses.laps.loc[[lap.Index]].get_car_data()
            except Exception:
                continue
            if tel is None or tel.empty:
                continue
            f = style_features(tel)
            f.update({"driver": lap.Driver, "gp": gp})
            rows.append(f)
    return pd.DataFrame(rows)


def dtw_distanz(a: np.ndarray, b: np.ndarray, band: int = DTW_BAND) -> float:
    """Klassisches DTW mit Sakoe-Chiba-Band (keine Zusatzbibliothek noetig,
    siehe Docstring)."""
    n, m = len(a), len(b)
    d = np.full((n + 1, m + 1), np.inf)
    d[0, 0] = 0.0
    for i in range(1, n + 1):
        j_lo, j_hi = max(1, i - band), min(m, i + band)
        for j in range(j_lo, j_hi + 1):
            kosten = abs(a[i - 1] - b[j - 1])
            d[i, j] = kosten + min(d[i - 1, j], d[i, j - 1], d[i - 1, j - 1])
    return float(d[n, m])


def dtw_matrix(ses) -> pd.DataFrame:
    """AUSBAUSTUFE: Speed-Spuren der schnellsten Runde je Fahrer, auf ein
    gemeinsames Punkteraster gebracht (DTW warpt in der Zeit, braucht aber
    trotzdem endliche, vergleichbar aufgeloeste Folgen)."""
    spuren = {}
    for drv in ses.drivers:
        abb = ses.get_driver(drv)["Abbreviation"]
        try:
            lap = ses.laps.pick_drivers(drv).pick_fastest()
            tel = lap.get_telemetry().add_distance()
        except Exception:
            continue
        if tel is None or tel.empty:
            continue
        grid = np.linspace(0, tel["Distance"].max(), DTW_PUNKTE)
        spuren[abb] = np.interp(grid, tel["Distance"], tel["Speed"])

    fahrer = list(spuren)
    n = len(fahrer)
    dist = np.zeros((n, n))
    for i, j in combinations(range(n), 2):
        d = dtw_distanz(spuren[fahrer[i]], spuren[fahrer[j]])
        dist[i, j] = dist[j, i] = d
    return pd.DataFrame(dist, index=fahrer, columns=fahrer)


def beste_clusteranzahl(X: np.ndarray, kandidaten=range(2, 6)) -> tuple[int, dict]:
    """VORGEHEN 4: k ueber Silhouette waehlen statt anzunehmen."""
    scores = {}
    for k in kandidaten:
        labels = KMeans(n_clusters=k, n_init=20, random_state=0).fit_predict(X)
        scores[k] = silhouette_score(X, labels)
    bestes_k = max(scores, key=scores.get)
    return bestes_k, scores


def cluster_interpretation(agg: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    """VORGEHEN 4: nicht nur Cluster-Mittel, sondern Abweichung vom
    Gesamtmittel - das zeigt, WAS ein Cluster tatsaechlich auszeichnet."""
    z = (agg - agg.mean()) / agg.std()
    z["cluster"] = labels
    return z.groupby("cluster").mean().round(2)


# Cluster sind hier Ergebnis, nicht von vornherein feste Kategorien - bei
# k>MAX_SERIEN reichen die drei Hausfarben nicht; ein qualitatives Colormap
# plus unterschiedliche Marker (Redundanz fuer Farbsehschwaeche) uebernimmt
# dann, dieselbe Ausnahme wie bei Gaengen/DRS-Codes in P09.
MARKER = ["o", "^", "s", "D", "P", "X", "v"]


def cluster_farben(labels: np.ndarray) -> dict:
    eindeutig = sorted(set(labels))
    if len(eindeutig) <= len(SERIEN):
        return {k: SERIEN[i] for i, k in enumerate(eindeutig)}
    cmap = plt.get_cmap("tab10")
    return {k: cmap(i % 10) for i, k in enumerate(eindeutig)}


def zeichne_pca(ax, pcs: np.ndarray, agg: pd.DataFrame, labels: np.ndarray) -> None:
    farben = cluster_farben(labels)
    for i, k in enumerate(sorted(set(labels))):
        m = labels == k
        ax.scatter(pcs[m, 0], pcs[m, 1], s=140, color=farben[k],
                  marker=MARKER[i % len(MARKER)], label=f"Cluster {k}", zorder=3)
    for i, name in enumerate(agg.index):
        ax.annotate(name, (pcs[i, 0], pcs[i, 1]), xytext=(6, 4),
                   textcoords="offset points", fontsize=8, color=MUTED)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("Fahrstil-Cluster aus Pedal-/Schaltstatistik", loc="left",
                color=FG, fontsize=12, pad=10)
    ax.legend(loc="best", frameon=False, labelcolor=FG, fontsize=9)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(alpha=0.3, linewidth=0.8, color=GRID)
    ax.set_axisbelow(True)


def zeichne_silhouette(ax, scores: dict, bestes_k: int) -> None:
    farben = [SERIEN[0] if k == bestes_k else MUTED for k in scores]
    ax.bar([str(k) for k in scores], list(scores.values()), color=farben,
          width=0.5)
    ax.set_xlabel("k")
    ax.set_ylabel("Silhouette Score")
    ax.set_title(f"Clusteranzahl: k={bestes_k} gewaehlt", loc="left",
                color=FG, fontsize=12, pad=10)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="y", alpha=0.3, linewidth=0.8, color=GRID)
    ax.set_axisbelow(True)


def zeichne_dtw(ax, mds_pts: np.ndarray, fahrer: list[str], labels: np.ndarray) -> None:
    farben = cluster_farben(labels)
    for i, k in enumerate(sorted(set(labels))):
        m = labels == k
        ax.scatter(mds_pts[m, 0], mds_pts[m, 1], s=140, color=farben[k],
                  marker=MARKER[i % len(MARKER)], label=f"Cluster {k}", zorder=3)
    for i, name in enumerate(fahrer):
        ax.annotate(name, (mds_pts[i, 0], mds_pts[i, 1]), xytext=(6, 4),
                   textcoords="offset points", fontsize=8, color=MUTED)
    ax.set_xlabel("MDS 1")
    ax.set_ylabel("MDS 2")
    ax.set_title(f"AUSBAUSTUFE: DTW auf Speed-Spur ({DTW_STRECKE})", loc="left",
                color=FG, fontsize=12, pad=10)
    ax.legend(loc="best", frameon=False, labelcolor=FG, fontsize=9)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(alpha=0.3, linewidth=0.8, color=GRID)
    ax.set_axisbelow(True)


def main():
    f1lab.enable_cache()

    print(f"[1/3] Telemetrie {STRECKEN} {SEASON} Q laden (VORGEHEN 1-2) ...")
    df = sammle_features()
    print(f"      {len(df)} Runden, {df.groupby(['gp', 'driver']).size().mean():.1f} "
         f"Runden je Fahrer und Strecke im Schnitt")
    agg = df.groupby("driver").mean(numeric_only=True)
    print(agg.round(2).to_string())

    print("\n[2/3] Standardisieren, PCA, k-Means (VORGEHEN 3-4) ...")
    X = StandardScaler().fit_transform(agg)
    pcs = PCA(n_components=2).fit_transform(X)
    bestes_k, scores = beste_clusteranzahl(X)
    print(f"      Silhouette je k: "
         f"{ {k: round(v, 3) for k, v in scores.items()} }")
    print(f"      -> k={bestes_k} gewaehlt")
    labels = KMeans(n_clusters=bestes_k, n_init=20, random_state=0).fit_predict(X)

    interpretation = cluster_interpretation(agg, labels)
    print("\n      Cluster-Profile (z-standardisiert, +/- vom Gesamtmittel):")
    print(interpretation.to_string())

    print(f"\n[3/3] AUSBAUSTUFE: DTW auf Speed-Spuren ({DTW_STRECKE}) ...")
    ses_dtw = f1lab.load(SEASON, DTW_STRECKE, "Q", telemetry=True)
    dist = dtw_matrix(ses_dtw)
    fahrer = list(dist.index)
    dtw_labels = AgglomerativeClustering(
        n_clusters=bestes_k, metric="precomputed", linkage="average"
    ).fit_predict(dist.to_numpy())

    gemeinsame = [d for d in fahrer if d in agg.index]
    feat_labels_g = pd.Series(labels, index=agg.index).reindex(gemeinsame)
    dtw_labels_g = pd.Series(dtw_labels, index=fahrer).reindex(gemeinsame)
    rand = adjusted_rand_score(feat_labels_g, dtw_labels_g)
    print(f"      {len(fahrer)} Fahrer, Adjusted Rand Index Feature- vs. "
         f"DTW-Cluster: {rand:.3f} (1.0 = identisch, 0.0 = Zufallsniveau)")

    print("\nGrafik ...")
    fig, ax = plt.subplots(1, 3, figsize=(19, 6.5))
    zeichne_pca(ax[0], pcs, agg, labels)
    zeichne_silhouette(ax[1], scores, bestes_k)
    mds_pts = MDS(n_components=2, dissimilarity="precomputed", random_state=7,
                 normalized_stress="auto").fit_transform(dist.to_numpy())
    zeichne_dtw(ax[2], mds_pts, fahrer, dtw_labels)
    fig.suptitle("Fahrstil-Clustering: Pedal-Statistik vs. DTW auf Speed-Spur",
                x=0.06, ha="left", fontsize=15, color=FG, y=1.03)
    plt.tight_layout()
    path = OUT / "fahrstil_cluster.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"\n      -> {path}")


if __name__ == "__main__":
    main()
