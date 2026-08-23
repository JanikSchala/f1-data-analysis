"""vier ML-bausteine aus P23-P25/P36: quali-vorhersage, fahrstil-clustering,
anomalie-erkennung, rennergebnis-/podium-vorhersage.

alle vier sind auf einen festen datensatz zugeschnitten (mehrere saisons
bzw. eine bestimmte session), nicht auf eine in der seitenleiste waehlbare
session. anders als der rest der app, aber wie
11_Boxenstopps.py/14_Historie.py aus demselben grund: das training braucht
entweder mehrere hundert session-ladevorgaenge oder eine session mit einer
bekannten, interessanten antwort (Baku 2024 mit vier ausfaellen). alle vier
datensaetze liegen dauerhaft im diskcache, nur der erste aufruf dauert.
"""
from __future__ import annotations

import fastf1
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import torch
from common import (
    achse,
    hinweis,
    kein_cache_hinweis,
    namensachse,
    setup,
    tabelle,
    zeige,
)
from scipy.stats import spearmanr
from sklearn.calibration import calibration_curve
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    IsolationForest,
)
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.manifold import MDS
from sklearn.metrics import (
    adjusted_rand_score,
    brier_score_loss,
    mean_absolute_error,
    roc_auc_score,
    silhouette_score,
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn

import f1lab
from f1lab import design as d

pfad = setup("Machine Learning", "Quali-Vorhersage, Fahrstil-Clustering, "
                                 "Anomalie-Erkennung und Rennergebnis-"
                                 "Vorhersage - fest zugeschnittene "
                                 "Datensaetze, dauerhaft im Diskcache.")
if kein_cache_hinweis(pfad):
    st.stop()

tab_quali, tab_stil, tab_anom, tab_renn = st.tabs(
    ["Quali-Vorhersage", "Fahrstil-Clustering", "Anomalie-Erkennung",
     "Renn-/Podium-Vorhersage"])

# ========================================================== quali-vorhersage
JAHRE = (2022, 2023, 2024)
TEAMFORM_FENSTER = 3


def _quali_zeit(row: pd.Series) -> float:
    for seg in ("Q3", "Q2", "Q1"):
        v = row[seg]
        if pd.notna(v):
            return v.total_seconds()
    return np.nan


def _sammle_quali(jahre: range) -> pd.DataFrame:
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
            zeiten = res.apply(_quali_zeit, axis=1)
            pole = zeiten.min()
            for (_, r), zt in zip(res.iterrows(), zeiten, strict=True):
                rows.append({
                    "season": jahr, "round": rnd, "event": event["EventName"],
                    "driver": r["Abbreviation"], "team": r["TeamName"],
                    "position": r["Position"],
                    "zeit_rel": zt / pole if pd.notna(zt) and pole else np.nan,
                })
    return pd.DataFrame(rows)


def _teamform_anhaengen(quali: pd.DataFrame) -> pd.DataFrame:
    quali = quali.sort_values(["season", "round"]).copy()
    je_team_rennen = (quali.groupby(["season", "round", "team"])["position"]
                      .mean().reset_index())
    je_team_rennen["team_form"] = (
        je_team_rennen.groupby(["season", "team"])["position"]
        .transform(lambda s: s.shift(1).rolling(TEAMFORM_FENSTER, min_periods=1).mean()))
    return quali.merge(je_team_rennen[["season", "round", "team", "team_form"]],
                       on=["season", "round", "team"], how="left")


def _vorjahr_anhaengen(quali: pd.DataFrame) -> pd.DataFrame:
    vorjahr = quali[["season", "event", "driver", "position"]].copy()
    vorjahr["season"] = vorjahr["season"] + 1
    vorjahr = vorjahr.rename(columns={"position": "vorjahr_position"})
    return quali.merge(vorjahr, on=["season", "event", "driver"], how="left")


def _fp_features(jahr: int, rnd: int) -> pd.DataFrame:
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
        for drv in best.index:
            rows.setdefault(drv, {})[f"{fp}_best"] = best[drv]
            rows.setdefault(drv, {})[f"{fp}_long"] = longrun.get(drv, np.nan)
    return pd.DataFrame(rows).T


def _positions_mae_aus_zeit(data: pd.DataFrame, pred_zeit: np.ndarray) -> float:
    tmp = data[["season", "round", "position"]].copy()
    tmp["pred_zeit"] = pred_zeit
    tmp["pred_rang"] = tmp.groupby(["season", "round"])["pred_zeit"].rank()
    return mean_absolute_error(tmp["position"], tmp["pred_rang"])


def _mlp_pipeline():
    """imputation (median) und skalierung noetig, weil ein MLP anders als
    der baum weder NaN noch unskalierte features nativ verarbeiten kann
    (siehe P23)."""
    return make_pipeline(
        SimpleImputer(strategy="median"), StandardScaler(),
        MLPRegressor(hidden_layer_sizes=(32, 16), max_iter=2000,
                    early_stopping=True, random_state=7))


@st.cache_data(persist="disk", show_spinner=False)
def _quali_datensatz(jahre: tuple[int, ...]) -> pd.DataFrame:
    quali = _sammle_quali(range(jahre[0] - 1, jahre[-1] + 1))
    quali = _teamform_anhaengen(quali)
    quali = _vorjahr_anhaengen(quali)
    quali = quali[quali["season"].isin(jahre)]

    frames = []
    for (jahr, rnd), _g in quali.groupby(["season", "round"]):
        fp = _fp_features(int(jahr), int(rnd))
        if fp.empty:
            continue
        fp.index.name = "driver"
        frames.append(fp.reset_index().assign(season=jahr, round=rnd))
    if not frames:
        return pd.DataFrame()
    fp_alle = pd.concat(frames, ignore_index=True)
    data = quali.merge(fp_alle, on=["season", "round", "driver"], how="inner")
    for c in [c for c in data.columns if c.endswith(("_best", "_long"))]:
        data[c + "_rel"] = data.groupby(["season", "round"])[c].transform(
            lambda s: s / s.min())
    return data.dropna(subset=["position"])


@st.cache_data(persist="disk", show_spinner=False)
def _quali_ergebnisse(jahre: tuple[int, ...]):
    """datensatz, 5-fold-cv und permutation importance in einem cache.

    vorher lief die cv-schleife (15 modell-fits je aufruf) roh im
    tab-koerper. st.tabs rendert alle tabs bei jedem rerun, das training
    startete also bei jeder interaktion irgendwo auf der seite neu.
    """
    data = _quali_datensatz(jahre)
    if data.empty:
        return {"data": data}

    feat = ([c for c in data.columns
            if c.startswith("FP") and c.endswith("_rel")]
           + ["vorjahr_position", "team_form"])
    data = data.sort_values(["season", "round"]).reset_index(drop=True)
    X, y_pos, y_zeit = data[feat], data["position"], data["zeit_rel"]

    cv = TimeSeriesSplit(n_splits=5)
    mae_pos, mae_zeit, mae_mlp = [], [], []
    letzter_test_idx = None
    for tr, te in cv.split(X):
        m_pos = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.06)
        m_pos.fit(X.iloc[tr], y_pos.iloc[tr])
        mae_pos.append(mean_absolute_error(
            y_pos.iloc[te], m_pos.predict(X.iloc[te])))

        m_zeit = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.06)
        gueltig = y_zeit.iloc[tr].notna()
        m_zeit.fit(X.iloc[tr][gueltig], y_zeit.iloc[tr][gueltig])
        mae_zeit.append(_positions_mae_aus_zeit(
            data.iloc[te], m_zeit.predict(X.iloc[te])))

        m_mlp = _mlp_pipeline()
        m_mlp.fit(X.iloc[tr], y_pos.iloc[tr])
        mae_mlp.append(mean_absolute_error(
            y_pos.iloc[te], m_mlp.predict(X.iloc[te])))

        letzter_test_idx = te
    baseline = mean_absolute_error(y_pos, np.full(len(y_pos), y_pos.median()))

    m_final = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.06)
    tr_final = np.setdiff1d(np.arange(len(X)), letzter_test_idx)
    m_final.fit(X.iloc[tr_final], y_pos.iloc[tr_final])
    imp = permutation_importance(m_final, X.iloc[letzter_test_idx],
                                 y_pos.iloc[letzter_test_idx],
                                 n_repeats=20, random_state=7)
    imp_s = pd.Series(imp.importances_mean, index=feat).sort_values()

    return {"data": data, "mae_pos": mae_pos, "mae_zeit": mae_zeit,
           "mae_mlp": mae_mlp, "baseline": baseline, "imp_s": imp_s}


with tab_quali:
    with st.spinner(f"Quali-Datensatz {JAHRE[0]}-{JAHRE[-1]} und "
                   "5-Fold-Validierung werden gebaut (nur beim ersten "
                   "Aufruf - mehrere hundert Sessions, dauert einige "
                   "Minuten) ..."):
        erg_q = _quali_ergebnisse(JAHRE)
    data = erg_q["data"]

    if data.empty:
        st.info(f"Fuer {JAHRE[0]}-{JAHRE[-1]} liegen nicht genug FP- und "
               "Quali-Sessions im Cache.")
    else:
        mae_pos, mae_zeit, mae_mlp = erg_q["mae_pos"], erg_q["mae_zeit"], erg_q["mae_mlp"]
        baseline, imp_s = erg_q["baseline"], erg_q["imp_s"]

        k = st.columns(3)
        k[0].metric("Fahrer-Wochenenden", len(data))
        k[1].metric("Events", data[["season", "round"]].drop_duplicates().shape[0])
        k[2].metric("Mit Vorjahreswert",
                   f"{data['vorjahr_position'].notna().mean():.0%}")

        links, rechts = st.columns([1, 1.3])
        with links:
            st.markdown("##### Zielgroesse und Modell im Vergleich")
            labels = ["Position\n(Baum)", "Zeit->Rang\n(Baum)",
                     "Position\n(MLP)", "Baseline\n(Median)"]
            werte = [np.mean(mae_pos), np.mean(mae_zeit), np.mean(mae_mlp), baseline]
            farben = [d.SERIEN[0], d.SERIEN[1], d.SERIEN[2], d.MUTED]
            fig = go.Figure(go.Bar(x=labels, y=werte, marker={"color": farben},
                                   text=[f"{v:.2f}" for v in werte],
                                   textposition="outside"))
            zeige(fig, hoehe=380, showlegend=False, xaxis=namensachse(),
                 yaxis=achse("MAE [Positionen]"))
            hinweis("Das direkte Positions-Modell (Baum) schlaegt den Umweg "
                    "ueber die Zeit-Vorhersage deutlich. Das neuronale Netz "
                    "(MLP, Median-Imputation + Skalierung noetig, der Baum "
                    "braucht beides nicht) liegt nur knapp dahinter - kein "
                    "klarer Sieg fuer eine Seite, beide weit vor der "
                    "Baseline (siehe P23).")

        with rechts:
            st.markdown("##### Feature Importance (Permutation, letzter Fold)")
            farben2 = [d.SERIEN[0] if v == imp_s.max() else d.MUTED
                      for v in imp_s.to_numpy()]
            fig2 = go.Figure(go.Bar(x=imp_s.to_numpy(), y=imp_s.index,
                                    orientation="h", marker={"color": farben2}))
            zeige(fig2, hoehe=380, showlegend=False,
                 xaxis=achse("Permutation Importance (MAE-Anstieg)"),
                 yaxis=namensachse())
            hinweis("team_form (rollierender Team-Formindex, um eine Runde "
                    "verschoben) draengt die einzelne FP-Trainingsrunde "
                    "deutlich in den Hintergrund (siehe P23).")

# ======================================================== fahrstil-clustering
STIL_SEASON = 2024
STIL_STRECKEN = ("Bahrain", "Spain", "Monza", "Suzuka")
DTW_STRECKE = "Suzuka"
QUICKLAP_SCHWELLE = 1.07
DTW_PUNKTE, DTW_BAND = 150, 20
MARKER = ["circle", "triangle-up", "square", "diamond", "cross", "x", "triangle-down"]


def _style_features(tel: pd.DataFrame) -> dict:
    t = tel.copy()
    t["dt"] = t["Time"].diff().dt.total_seconds().fillna(0)
    total = t["dt"].sum()
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


def _dtw_distanz(a: np.ndarray, b: np.ndarray, band: int = DTW_BAND) -> float:
    n, m = len(a), len(b)
    dmat = np.full((n + 1, m + 1), np.inf)
    dmat[0, 0] = 0.0
    for i in range(1, n + 1):
        j_lo, j_hi = max(1, i - band), min(m, i + band)
        for j in range(j_lo, j_hi + 1):
            kosten = abs(a[i - 1] - b[j - 1])
            dmat[i, j] = kosten + min(dmat[i - 1, j], dmat[i, j - 1], dmat[i - 1, j - 1])
    return float(dmat[n, m])


@st.cache_data(persist="disk", show_spinner=False)
def _fahrstil_daten():
    rows = []
    for gp in STIL_STRECKEN:
        ses = f1lab.load(STIL_SEASON, gp, "Q", telemetry=True)
        laps = ses.laps.pick_accurate().pick_quicklaps(QUICKLAP_SCHWELLE)
        for lap in laps.itertuples():
            try:
                tel = ses.laps.loc[[lap.Index]].get_car_data()
            except Exception:
                continue
            if tel is None or tel.empty:
                continue
            f = _style_features(tel)
            f.update({"driver": lap.Driver, "gp": gp})
            rows.append(f)
    df = pd.DataFrame(rows)
    if df.empty:
        return df, pd.DataFrame(), [], np.zeros((0, 0))
    agg = df.groupby("driver").mean(numeric_only=True)

    X = StandardScaler().fit_transform(agg)
    scores = {k: silhouette_score(X, KMeans(n_clusters=k, n_init=20,
                                            random_state=0).fit_predict(X))
             for k in range(2, 6)}
    bestes_k = max(scores, key=scores.get)
    labels = KMeans(n_clusters=bestes_k, n_init=20, random_state=0).fit_predict(X)
    pcs = PCA(n_components=2).fit_transform(X)

    ses_dtw = f1lab.load(STIL_SEASON, DTW_STRECKE, "Q", telemetry=True)
    spuren = {}
    for drv in ses_dtw.drivers:
        abb = ses_dtw.get_driver(drv)["Abbreviation"]
        try:
            lap = ses_dtw.laps.pick_drivers(drv).pick_fastest()
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
    for i in range(n):
        for j in range(i + 1, n):
            dv = _dtw_distanz(spuren[fahrer[i]], spuren[fahrer[j]])
            dist[i, j] = dist[j, i] = dv
    dtw_labels = AgglomerativeClustering(
        n_clusters=bestes_k, metric="precomputed", linkage="average"
    ).fit_predict(dist) if n > bestes_k else np.zeros(n, dtype=int)
    mds_pts = (MDS(n_components=2, dissimilarity="precomputed", random_state=7,
                   normalized_stress="auto").fit_transform(dist)
              if n > 2 else np.zeros((n, 2)))

    gemeinsame = [dv for dv in fahrer if dv in agg.index]
    feat_labels_g = pd.Series(labels, index=agg.index).reindex(gemeinsame)
    dtw_labels_g = pd.Series(dtw_labels, index=fahrer).reindex(gemeinsame)
    rand = (adjusted_rand_score(feat_labels_g, dtw_labels_g)
           if len(gemeinsame) > 1 else float("nan"))

    return {
        "agg": agg, "pcs": pcs, "labels": labels, "scores": scores,
        "bestes_k": bestes_k, "fahrer": fahrer, "dtw_labels": dtw_labels,
        "mds_pts": mds_pts, "rand": rand,
    }


with tab_stil:
    with st.spinner(f"Telemetrie {STIL_SEASON} ({', '.join(STIL_STRECKEN)}) "
                   "wird aus dem lokalen Cache gebaut (nur beim ersten "
                   "Aufruf) ..."):
        erg = _fahrstil_daten()

    if not erg or erg.get("agg", pd.DataFrame()).empty:
        st.info(f"Fuer {STIL_SEASON} ({', '.join(STIL_STRECKEN)}) liegt "
               "nicht genug Quali-Telemetrie im Cache.")
    else:
        agg, pcs, labels = erg["agg"], erg["pcs"], erg["labels"]
        k = st.columns(3)
        k[0].metric("Fahrer", len(agg))
        k[1].metric("Cluster (per Silhouette gewaehlt)", erg["bestes_k"])
        k[2].metric("Adjusted Rand Index (Feature- vs. DTW-Cluster)",
                   f"{erg['rand']:.3f}" if pd.notna(erg["rand"]) else "\N{EN DASH}")

        links, rechts = st.columns(2)
        with links:
            st.markdown("##### PCA: Fahrstil-Cluster aus Pedal-/Schaltstatistik")
            fig = go.Figure()
            for i, cl in enumerate(sorted(set(labels))):
                m = labels == cl
                fig.add_trace(go.Scatter(
                    x=pcs[m, 0], y=pcs[m, 1], mode="markers+text",
                    text=list(agg.index[m]), textposition="top center",
                    textfont={"color": d.MUTED, "size": 9},
                    marker={"size": 13, "color": d.SERIEN[i % len(d.SERIEN)],
                           "symbol": MARKER[i % len(MARKER)]},
                    name=f"Cluster {cl}"))
            zeige(fig, hoehe=420, xaxis=achse("PC1"), yaxis=achse("PC2"))

        with rechts:
            st.markdown("##### Clusteranzahl per Silhouette-Score")
            scores = erg["scores"]
            farben = [d.SERIEN[0] if kk == erg["bestes_k"] else d.MUTED
                     for kk in scores]
            fig2 = go.Figure(go.Bar(x=[str(kk) for kk in scores],
                                    y=list(scores.values()),
                                    marker={"color": farben}))
            zeige(fig2, hoehe=420, showlegend=False, xaxis=namensachse("k"),
                 yaxis=achse("Silhouette Score"))
            hinweis("Silhouette-Werte liegen fuer alle getesteten k dicht "
                    "beieinander (0.21-0.24 in P22s Original) - keine "
                    "natuerliche, klar abgesetzte Clusterstruktur.")

        st.markdown(f"##### AUSBAUSTUFE: DTW auf Speed-Spur ({DTW_STRECKE})")
        fig3 = go.Figure()
        for i, cl in enumerate(sorted(set(erg["dtw_labels"]))):
            m = erg["dtw_labels"] == cl
            fig3.add_trace(go.Scatter(
                x=erg["mds_pts"][m, 0], y=erg["mds_pts"][m, 1],
                mode="markers+text",
                text=[fh for fh, mm in zip(erg["fahrer"], m, strict=True) if mm],
                textposition="top center", textfont={"color": d.MUTED, "size": 9},
                marker={"size": 13, "color": d.SERIEN[i % len(d.SERIEN)],
                       "symbol": MARKER[i % len(MARKER)]},
                name=f"Cluster {cl}"))
        zeige(fig3, hoehe=420, xaxis=achse("MDS 1"), yaxis=achse("MDS 2"))
        hinweis("Cluster aus der rohen Geschwindigkeitsspur (dynamic time "
                "warping) stimmen mit den Pedal-Feature-Clustern praktisch "
                "nicht ueberein (Adjusted Rand Index nahe 0, siehe Metrik "
                "oben) - beide Verfahren erfassen etwas anderes (siehe P24).")

# ======================================================== anomalie-erkennung
ANOM_SEASON, ANOM_EVENT, ANOM_IDENT = 2024, "Baku", "R"
CONTAMINATION = 0.04
FEAT_ANOM = ["lap_time", "vmax", "vmean", "rpm_max", "rpm_mean", "throttle_mean",
            "brake_pct", "gear_mean"]
TRACE_KANAELE = ["Speed", "RPM", "Throttle", "Brake"]
TRACE_PUNKTE = 96
AE_BOTTLENECK = 8
AE_EPOCHS = 80


class LapAutoencoder(nn.Module):
    """1D-Conv-Autoencoder fuer telemetriespuren (siehe P25 zweite
    AUSBAUSTUFE). schmaler flaschenhals, damit das netz eine "normale"
    rundenform komprimiert statt sie auswendig zu lernen."""

    def __init__(self, n_channels=None, n_points=TRACE_PUNKTE,
                bottleneck=AE_BOTTLENECK):
        n_channels = len(TRACE_KANAELE) if n_channels is None else n_channels
        super().__init__()
        halb = n_points // 4
        self.halb, self.kanaele_innen = halb, 32
        self.encoder = nn.Sequential(
            nn.Conv1d(n_channels, 16, kernel_size=4, stride=2, padding=1), nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=4, stride=2, padding=1), nn.ReLU())
        self.zu_bottleneck = nn.Linear(32 * halb, bottleneck)
        self.von_bottleneck = nn.Linear(bottleneck, 32 * halb)
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(32, 16, kernel_size=4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose1d(16, n_channels, kernel_size=4, stride=2, padding=1))

    def forward(self, x):
        z = self.encoder(x)
        b = self.zu_bottleneck(z.flatten(1))
        z2 = self.von_bottleneck(b).view(-1, self.kanaele_innen, self.halb)
        return self.decoder(z2)


def _lap_traces(ses, laps_index):
    spuren, index = [], []
    for idx in laps_index:
        try:
            tel = ses.laps.loc[[idx]].get_car_data().add_distance()
        except Exception:
            continue
        if tel is None or len(tel) < 50 or tel["Distance"].max() <= 0:
            continue
        grid = np.linspace(0, tel["Distance"].max(), TRACE_PUNKTE)
        kanaele = [np.interp(grid, tel["Distance"], tel[k].astype(float))
                  for k in TRACE_KANAELE]
        spuren.append(np.stack(kanaele))
        index.append(idx)
    return np.stack(spuren), index


def _autoencoder_scores(spuren, seed: int = 0):
    torch.manual_seed(seed)
    mean = spuren.mean(axis=(0, 2), keepdims=True)
    std = spuren.std(axis=(0, 2), keepdims=True) + 1e-6
    x = torch.tensor((spuren - mean) / std, dtype=torch.float32)

    modell = LapAutoencoder(n_channels=spuren.shape[1], n_points=spuren.shape[2])
    opt = torch.optim.Adam(modell.parameters(), lr=1e-3, weight_decay=1e-5)
    verlust_fn = nn.MSELoss()
    modell.train()
    for _epoch in range(AE_EPOCHS):
        opt.zero_grad()
        verlust = verlust_fn(modell(x), x)
        verlust.backward()
        opt.step()
    modell.eval()
    with torch.no_grad():
        fehler = ((modell(x) - x) ** 2).mean(dim=(1, 2)).numpy()
    return fehler


def _ausfall_analyse(df: pd.DataFrame, anomalien: pd.DataFrame, ses) -> pd.DataFrame:
    ausfaelle = ses.results[ses.results["Status"].isin(["Retired"])]
    zeilen = []
    for r in ausfaelle.itertuples():
        drv = r.Abbreviation
        letzte_runde = df.loc[df["driver"] == drv, "lap"].max()
        eigene = anomalien[anomalien["driver"] == drv].sort_values("lap")
        vor_ausfall = eigene[(eigene["anomalie"] == -1)
                             & (eigene["lap"] < letzte_runde)]
        auf_strecke = vor_ausfall[~vor_ausfall["boxenrunde"]]
        vorlauf = (letzte_runde - auf_strecke["lap"].max()
                  if not auf_strecke.empty else np.nan)
        zeilen.append({
            "driver": drv, "status": r.Status, "letzte_runde": letzte_runde,
            "anomalien_boxenrunden": len(vor_ausfall) - len(auf_strecke),
            "anomalien_auf_strecke": len(auf_strecke),
            "runden_vorlauf_auf_strecke": vorlauf,
        })
    return pd.DataFrame(zeilen)


def _rundenprofile(ses) -> pd.DataFrame:
    phasen = f1lab.track_status_phases(ses)
    neutral_laps = set()
    for p in phasen[phasen["label"] != "gruen"].itertuples():
        neutral_laps.update(range(p.lap_start, p.lap_end + 1))

    rows = []
    for lap in ses.laps.itertuples():
        if pd.isna(lap.LapTime):
            continue
        try:
            tel = ses.laps.loc[[lap.Index]].get_car_data()
        except Exception:
            continue
        if tel is None or len(tel) < 50:
            continue
        rows.append({
            "driver": lap.Driver, "lap": lap.LapNumber,
            "lap_time": lap.LapTime.total_seconds(),
            "vmax": tel["Speed"].max(), "vmean": tel["Speed"].mean(),
            "rpm_max": tel["RPM"].max(), "rpm_mean": tel["RPM"].mean(),
            "throttle_mean": tel["Throttle"].mean(),
            "brake_pct": 100 * tel["Brake"].astype(bool).mean(),
            "gear_mean": tel["nGear"].mean(),
            "boxenrunde": pd.notna(lap.PitInTime) or pd.notna(lap.PitOutTime),
            "neutralisiert": lap.LapNumber in neutral_laps,
            "erste_runde": lap.LapNumber == 1,
        })
    return pd.DataFrame(rows).dropna()


@st.cache_data(persist="disk", show_spinner=False)
def _anomalie_daten():
    ses = f1lab.load(ANOM_SEASON, ANOM_EVENT, ANOM_IDENT, telemetry=True,
                     messages=True)
    df = _rundenprofile(ses)
    if df.empty:
        leer = pd.DataFrame()
        return df, leer, leer, leer, leer, leer, float("nan"), []

    sauber = df[~df["neutralisiert"] & ~df["erste_runde"]].copy()
    z = sauber.groupby("driver")[FEAT_ANOM].transform(
        lambda s: (s - s.mean()) / (s.std() + 1e-9))
    iso = IsolationForest(contamination=CONTAMINATION, random_state=0)
    sauber["anomalie"] = iso.fit_predict(z)
    sauber["score"] = -iso.score_samples(z)

    phasen = f1lab.track_status_phases(ses)
    ausfall_tab = _ausfall_analyse(df, sauber, ses)

    # zweite AUSBAUSTUFE: derselbe vergleich wie im P25-skript
    spuren, index = _lap_traces(ses, sauber.index)
    fehler = _autoencoder_scores(spuren)
    sauber_ae = sauber.loc[index].copy()
    sauber_ae["score"] = fehler
    schwelle = np.quantile(fehler, 1 - CONTAMINATION)
    sauber_ae["anomalie"] = np.where(fehler >= schwelle, -1, 1)
    ausfall_tab_ae = _ausfall_analyse(df, sauber_ae, ses)

    gemeinsam = sauber.index.intersection(sauber_ae.index)
    korr, _p = spearmanr(sauber.loc[gemeinsam, "score"],
                         sauber_ae.loc[gemeinsam, "score"])

    return (df, sauber, phasen, ausfall_tab, sauber_ae, ausfall_tab_ae, korr,
           list(gemeinsam))


with tab_anom:
    with st.spinner(f"{ANOM_EVENT} {ANOM_SEASON} {ANOM_IDENT} wird analysiert, "
                   "inklusive Autoencoder auf rohen Telemetriespuren (nur beim "
                   "ersten Aufruf) ..."):
        (df, anomalien, phasen, ausfall_tab, anomalien_ae, ausfall_tab_ae,
         ae_korr, ae_gemeinsam) = _anomalie_daten()

    if df.empty:
        st.info(f"Fuer {ANOM_EVENT} {ANOM_SEASON} liegt keine auswertbare "
               "Telemetrie im Cache.")
    else:
        geflaggt = int((anomalien["anomalie"] == -1).sum())
        k = st.columns(4)
        k[0].metric("Runden gesamt", len(df))
        k[1].metric("Unter Gelb/VSC/SC", int(df["neutralisiert"].sum()))
        k[2].metric("Sauber ausgewertet", len(anomalien))
        k[3].metric("Geflaggt", f"{geflaggt} ({CONTAMINATION:.0%})")

        st.markdown("##### Anomalie-Score ueber den Rennverlauf")
        fig = go.Figure()
        for p in phasen[(phasen["label"] != "gruen")
                       & (phasen["lap_start"] > 0)].itertuples():
            fig.add_vrect(x0=p.lap_start, x1=max(p.lap_end, p.lap_start + 0.3),
                         fillcolor=d.PHASE.get(p.label, d.MUTED), opacity=0.3,
                         line_width=0, layer="below")
        for _drv, g in anomalien.groupby("driver"):
            g = g.sort_values("lap")
            fig.add_trace(go.Scatter(x=g["lap"], y=g["score"], mode="lines",
                                     line={"color": d.MUTED, "width": 0.8},
                                     opacity=0.35, showlegend=False,
                                     hoverinfo="skip"))
        flags = anomalien[anomalien["anomalie"] == -1]
        fig.add_trace(go.Scatter(x=flags["lap"], y=flags["score"], mode="markers",
                                 marker={"color": d.SERIEN[1], "size": 8},
                                 name="Anomalie (sauber)",
                                 customdata=flags["driver"],
                                 hovertemplate="%{customdata} Runde %{x}<br>"
                                              "Score %{y:.2f}<extra></extra>"))
        zeige(fig, hoehe=420, xaxis=achse("Runde"), yaxis=achse("Anomalie-Score"))
        hinweis("Farbige Baender sind Gelb/VSC/SC-Phasen - Runden darunter "
                "sind vom Training ausgeschlossen, sonst waeren die Top-Flags "
                "fast ausschliesslich Rennleitungs-Artefakte statt "
                "Telemetrie-Befunde (siehe P25).")

        st.markdown("##### AUSBAUSTUFE: Fruehwarnung vor Ausfaellen?")
        if ausfall_tab.empty:
            st.info("Keine Ausfaelle (Status='Retired') in dieser Session.")
        else:
            tabelle(ausfall_tab.rename(columns={
                "driver": "Fahrer", "status": "Status",
                "letzte_runde": "Letzte Runde",
                "anomalien_boxenrunden": "Anomalien (Box)",
                "anomalien_auf_strecke": "Anomalien (Strecke)",
                "runden_vorlauf_auf_strecke": "Runden Vorlauf (Strecke)"}))
            hinweis("Nur Anomalien auf der Strecke (nicht in der Box) zaehlen "
                    "als echte Fruehwarnung - ein planmaessiger Reifenwechsel "
                    "waere sonst faelschlich eine \"Warnung\". Ehrlicher "
                    "Befund aus P25: fuer die meisten Ausfaelle gibt es kaum "
                    "Vorlauf - die auffaelligen Werte erscheinen oft erst in "
                    "der Runde, in der ohnehin schon in die Box gefahren "
                    "wird.")

        st.markdown("##### ZWEITE AUSBAUSTUFE: Autoencoder auf rohen "
                   "Telemetriespuren statt aggregierten Rundenkennzahlen")
        hinweis("1D-Conv-Autoencoder (PyTorch) auf Speed/RPM/Throttle/Brake, "
                "auf 96 Distanzpunkte interpoliert - Rekonstruktionsfehler "
                "als Anomalie-Score, trainiert auf derselben sauberen "
                "Grundgesamtheit wie der IsolationForest oben (siehe P25).")

        k2 = st.columns(2)
        k2[0].metric("Spearman-Korrelation der beiden Scores", f"{ae_korr:.3f}",
                    help="Nahe 0 heisst: die beiden Verfahren flaggen "
                         "weitgehend unterschiedliche Runden.")
        geflaggt_ae = int((anomalien_ae["anomalie"] == -1).sum())
        k2[1].metric("Autoencoder geflaggt", f"{geflaggt_ae} ({CONTAMINATION:.0%})")

        iso_s = anomalien.loc[ae_gemeinsam, "score"]
        ae_s = anomalien_ae.loc[ae_gemeinsam, "score"]
        iso_flag = anomalien.loc[ae_gemeinsam, "anomalie"] == -1
        ae_flag = anomalien_ae.loc[ae_gemeinsam, "anomalie"] == -1
        gruppen = [("keins", ~iso_flag & ~ae_flag, d.MUTED, 5),
                  ("nur IsolationForest", iso_flag & ~ae_flag, d.SERIEN[0], 8),
                  ("nur Autoencoder", ~iso_flag & ae_flag, d.SERIEN[1], 8),
                  ("beide", iso_flag & ae_flag, d.SERIEN[2], 11)]
        fig_ae = go.Figure()
        for label, maske, farbe, groesse in gruppen:
            fig_ae.add_trace(go.Scatter(
                x=iso_s[maske], y=ae_s[maske], mode="markers",
                marker={"color": farbe, "size": groesse}, name=label,
                hovertemplate=f"{label}<br>IsolationForest %{{x:.2f}}<br>"
                              "Autoencoder %{y:.2f}<extra></extra>"))
        zeige(fig_ae, hoehe=420, xaxis=achse("IsolationForest-Score"),
             yaxis=achse("Autoencoder-Rekonstruktionsfehler"))

        st.markdown("###### Ausfallanalyse mit Autoencoder-Flags")
        tabelle(ausfall_tab_ae.rename(columns={
            "driver": "Fahrer", "status": "Status",
            "letzte_runde": "Letzte Runde",
            "anomalien_boxenrunden": "Anomalien (Box)",
            "anomalien_auf_strecke": "Anomalien (Strecke)",
            "runden_vorlauf_auf_strecke": "Runden Vorlauf (Strecke)"}))
        hinweis("Der Autoencoder zeigt hier fuer alle vier Ausfaelle mehr "
                "Vorlauf als der IsolationForest oben - aber Vorsicht mit "
                "dem Schluss \"also besser\": PER und SAI schieden durch "
                "eine Kollision aus, die sich mechanisch nicht ueber Dutzende "
                "Runden ankuendigen kann. Dass ausgerechnet dort die groessten "
                "Vorlaufzeiten stehen, ist eher ein Hinweis, dass die globale "
                "(nicht fahrerweise) Normalisierung des Autoencoders "
                "Strecken-/Verkehrsmuster statt individueller "
                "Fahrzeugabweichungen lernt - eine Lektion ueber "
                "Normalisierung, kein bestaetigter Fund ueber die Autos "
                "(siehe P25).")

# =================================================== renn-/podium-vorhersage
RENN_JAHRE = (2022, 2023, 2024)
RENN_FORM_FENSTER = 5


def _sammle_rennen(jahre) -> pd.DataFrame:
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


def _renn_form_anhaengen(races: pd.DataFrame, fenster: int = RENN_FORM_FENSTER
                         ) -> pd.DataFrame:
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


@st.cache_data(persist="disk", show_spinner=False)
def _renn_datensatz(jahre: tuple[int, ...]) -> pd.DataFrame:
    races = _sammle_rennen(range(jahre[0], jahre[-1] + 1))
    if races.empty:
        return races
    races = _renn_form_anhaengen(races)
    return races.dropna(subset=["grid", "position"])


@st.cache_data(persist="disk", show_spinner=False)
def _renn_ergebnisse(jahre: tuple[int, ...]):
    """datensatz, klassifikation, regression und kalibrierung in einem cache.

    dieselbe umstellung wie bei _quali_ergebnisse: die cv-schleife lief
    vorher roh im tab-koerper und damit bei jedem rerun neu.
    """
    races = _renn_datensatz(jahre)
    if races.empty:
        return {"races": races}

    feat = ["grid", "driver_form", "team_form", "dnf_rate"]
    races = races.sort_values(["season", "round"]).reset_index(drop=True)
    X = races[feat]
    y_podium = races["podium"].astype(int)

    cv = TimeSeriesSplit(n_splits=5)
    auc_scores, brier_scores = [], []
    alle_y_true, alle_y_prob = [], []
    letzter_test_idx = None
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
    brier_baseline = brier_score_loss(
        y_true_ges, np.full(len(y_true_ges), basisrate))
    acc_baseline = ((races["grid"] <= 3).astype(int) == y_podium).mean()

    fertig = races[~races["dnf"]].copy()
    Xf, yf = fertig[feat], fertig["position"]
    mae_scores, mae_baseline_scores = [], []
    for tr, te in cv.split(Xf):
        m2 = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.06)
        m2.fit(Xf.iloc[tr], yf.iloc[tr])
        mae_scores.append(mean_absolute_error(
            yf.iloc[te], m2.predict(Xf.iloc[te])))
        mae_baseline_scores.append(mean_absolute_error(
            yf.iloc[te], fertig["grid"].iloc[te]))

    beob, vorh = calibration_curve(y_true_ges, y_prob_ges, n_bins=10,
                                   strategy="uniform")

    m_final = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06)
    tr_final = np.setdiff1d(np.arange(len(X)), letzter_test_idx)
    m_final.fit(X.iloc[tr_final], y_podium.iloc[tr_final])
    imp = permutation_importance(m_final, X.iloc[letzter_test_idx],
                                 y_podium.iloc[letzter_test_idx],
                                 scoring="roc_auc", n_repeats=20,
                                 random_state=7)
    imp_s = pd.Series(imp.importances_mean, index=feat).sort_values()

    return {"races": races, "auc_scores": auc_scores, "brier_scores": brier_scores,
           "basisrate": basisrate, "brier_baseline": brier_baseline,
           "acc_baseline": acc_baseline, "mae_scores": mae_scores,
           "mae_baseline_scores": mae_baseline_scores, "beob": beob, "vorh": vorh,
           "imp_s": imp_s}


with tab_renn:
    with st.spinner(f"Rennergebnisse {RENN_JAHRE[0]}-{RENN_JAHRE[-1]} und "
                   "5-Fold-Validierung werden gebaut (nur beim ersten "
                   "Aufruf - ein Ladevorgang je Rennen, dauert einige "
                   "Minuten) ..."):
        erg_r = _renn_ergebnisse(RENN_JAHRE)
    races = erg_r["races"]

    if races.empty:
        st.info(f"Fuer {RENN_JAHRE[0]}-{RENN_JAHRE[-1]} liegen nicht genug "
               "Rennsessions im Cache.")
    else:
        auc_scores, brier_scores = erg_r["auc_scores"], erg_r["brier_scores"]
        basisrate, brier_baseline = erg_r["basisrate"], erg_r["brier_baseline"]
        acc_baseline = erg_r["acc_baseline"]
        mae_scores, mae_baseline_scores = erg_r["mae_scores"], erg_r["mae_baseline_scores"]
        beob, vorh, imp_s = erg_r["beob"], erg_r["vorh"], erg_r["imp_s"]

        k = st.columns(4)
        k[0].metric("Fahrer-Rennen", len(races))
        k[1].metric("Rennen", races[["season", "round"]].drop_duplicates().shape[0])
        k[2].metric("DNF-Quote", f"{races['dnf'].mean():.1%}")
        k[3].metric("Podium-Quote", f"{races['podium'].mean():.1%}")

        links, mitte, rechts = st.columns(3)
        with links:
            st.markdown("##### Podium-Klassifikation")
            labels_p = ["Modell\n(ROC-AUC)", "Baseline Grid<=3\n(Accuracy)"]
            werte_p = [np.mean(auc_scores), acc_baseline]
            fig = go.Figure(go.Bar(x=labels_p, y=werte_p,
                                   marker={"color": [d.SERIEN[0], d.MUTED]},
                                   text=[f"{v:.3f}" for v in werte_p],
                                   textposition="outside"))
            zeige(fig, hoehe=340, showlegend=False, xaxis=namensachse(),
                 yaxis=achse("Score", range=[0, 1.05]))
            hinweis(f"Brier Score Modell {np.mean(brier_scores):.3f} gegen "
                    f"Baseline (immer Basisrate {basisrate:.1%}): "
                    f"{brier_baseline:.3f} (siehe P36).")

        with mitte:
            st.markdown("##### Kalibrierung")
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                                      line={"color": d.MUTED, "dash": "dash"},
                                      name="perfekt kalibriert"))
            fig2.add_trace(go.Scatter(x=vorh, y=beob, mode="lines+markers",
                                      line={"color": d.SERIEN[1]},
                                      name="Modell"))
            zeige(fig2, hoehe=340, xaxis=achse("Vorhergesagt"),
                 yaxis=achse("Beobachtet"))
            hinweis("Eine Vorhersage von 70% Podium muss ueber viele Faelle "
                    "auch in etwa 70% der Zeit eintreffen, sonst ist sie "
                    "keine echte Wahrscheinlichkeit (siehe P36).")

        with rechts:
            st.markdown("##### Feature Importance")
            farben = [d.SERIEN[0] if v == imp_s.max() else d.MUTED
                     for v in imp_s.to_numpy()]
            fig3 = go.Figure(go.Bar(x=imp_s.to_numpy(), y=imp_s.index,
                                    orientation="h", marker={"color": farben}))
            zeige(fig3, hoehe=340, showlegend=False,
                 xaxis=achse("Permutation Importance (AUC-Abfall)"),
                 yaxis=namensachse())
            hinweis(f"Positions-Regression (nur gefinishte Ergebnisse): "
                    f"MAE {np.mean(mae_scores):.2f} gegen Baseline "
                    f"(Grid=Ziel) {np.mean(mae_baseline_scores):.2f} - "
                    "Startplatz dominiert, Form/Zuverlaessigkeit liefern nur "
                    "Feinschliff (siehe P36).")
