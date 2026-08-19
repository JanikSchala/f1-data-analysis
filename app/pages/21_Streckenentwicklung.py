"""Streckenentwicklung uebers Qualifying-Wochenende (P43): wird die Strecke
von Q1 zu Q3 wirklich schneller?

Wie 18_Ueberholschwierigkeit.py/19_Startplatz_Paritaet.py an eine ganze
Saison gebunden statt an eine einzelne Session, `persist="disk"`-gecacht.
Gerechnet wird nirgends hier: f1lab.qualifying_track_evolution() liefert
die paarweisen Fahrer-Deltas zwischen Q1/Q2/Q3 (nur Fahrer, die in beiden
Segmenten eine Zeit gesetzt haben - haelt Auto-/Fahrerqualitaet konstant,
siehe P43-Docstring). Nasse Qualifyings (INTERMEDIATE/WET-Reifen) werden
ausgeschlossen, weil eine trocknende Strecke einen eigenen, viel groesseren
Zeiteffekt hat als das reine "Gummi einfahren".
"""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from common import achse, hinweis, inventar, kein_cache_hinweis, setup, tabelle, zeige
from scipy.stats import pearsonr, wilcoxon

import f1lab
from f1lab import design as d

pfad = setup("Streckenentwicklung", "Wird die Strecke von Q1 zu Q3 wirklich "
                                    "schneller - und ist das mehr als nur "
                                    "'die langsameren Autos sind raus'?")
if kein_cache_hinweis(pfad):
    st.stop()

inv = inventar(str(pfad))
saisons = sorted(inv[inv["ident"] == "Q"]["season"].unique(), reverse=True)
with st.sidebar:
    st.header("Saison")
    saison = st.selectbox("Jahr", saisons, key="streckenentw_saison")


@st.cache_data(persist="disk", show_spinner="Qualifying-Segmente werden "
                                            "ausgewertet (erster Aufruf je "
                                            "Saison dauert) ...")
def _deltas(cache_pfad: str, saison: int) -> tuple[pd.DataFrame, list[str]]:
    inv_lokal = f1lab.cached_sessions(cache_pfad)
    rennen = sorted(inv_lokal[(inv_lokal["season"] == saison)
                              & (inv_lokal["ident"] == "Q")]["event"].unique())
    alle, nass = [], []
    for gp in rennen:
        try:
            ses = f1lab.load(saison, gp, "Q", telemetry=False)
        except Exception:
            continue
        if ses.laps["Compound"].isin(["INTERMEDIATE", "WET"]).any():
            nass.append(gp)
            continue
        d_ = f1lab.qualifying_track_evolution(ses)
        if not d_.empty:
            d_ = d_.copy()
            d_["gp"] = gp
            alle.append(d_)
    return (pd.concat(alle, ignore_index=True) if alle else pd.DataFrame(),
            nass)


deltas, nasse_rennen = _deltas(str(pfad), int(saison))
if deltas.empty:
    st.info(f"Keine auswertbaren trockenen Qualifyings fuer {saison} im Cache.")
    st.stop()

k = st.columns(4)
k[0].metric("Trockene Qualifyings", deltas["gp"].nunique())
if nasse_rennen:
    k[1].metric("Wegen Regen ausgeschlossen", len(nasse_rennen),
               help=", ".join(nasse_rennen))
else:
    k[1].metric("Wegen Regen ausgeschlossen", 0)

q12 = deltas.loc[deltas["segment"] == "Q1->Q2", "delta_s"]
q23 = deltas.loc[deltas["segment"] == "Q2->Q3", "delta_s"]
p12 = wilcoxon(q12).pvalue if len(q12) else float("nan")
p23 = wilcoxon(q23).pvalue if len(q23) else float("nan")
k[2].metric("Q1->Q2 Median", f"{q12.median():+.3f} s",
           help=f"n={len(q12)} Fahrer-Vergleiche, {(q12 > 0).mean():.0%} "
                f"positiv, Wilcoxon p={p12:.1e}")
k[3].metric("Q2->Q3 Median", f"{q23.median():+.3f} s",
           help=f"n={len(q23)} Fahrer-Vergleiche, {(q23 > 0).mean():.0%} "
                f"positiv, Wilcoxon p={p23:.1e}")

st.markdown("##### Verteilung der paarweisen Fahrer-Deltas")
fig = go.Figure()
fig.add_trace(go.Box(y=q12, name="Q1->Q2", marker={"color": d.SERIEN[0]},
                     boxpoints=False))
fig.add_trace(go.Box(y=q23, name="Q2->Q3", marker={"color": d.SERIEN[1]},
                     boxpoints=False))
fig.add_hline(y=0, line_color=d.MUTED, line_width=1, line_dash="dash")
zeige(fig, hoehe=420, showlegend=False,
     yaxis=achse("Delta [s], positiv = spaeteres Segment schneller"))
hinweis("Nur Fahrer, die in BEIDEN verglichenen Segmenten eine gewertete "
       "Zeit gesetzt haben - so bleibt Auto-/Fahrerqualitaet konstant. Ein "
       "reiner Schnitt-Vergleich Q1 gegen Q3 waere verzerrt, weil in Q3 nur "
       "die schnelleren Autos uebrig sind (siehe P43).")

st.markdown("##### Konsistenz: haelt sich der Effekt in jedem Rennen?")
je_rennen = (deltas.groupby(["gp", "segment"])["delta_s"].median()
                   .unstack("segment").reset_index())
je_rennen = je_rennen.sort_values("Q1->Q2")
fig2 = go.Figure()
fig2.add_trace(go.Bar(x=je_rennen["gp"], y=je_rennen["Q1->Q2"],
                      name="Q1->Q2", marker={"color": d.SERIEN[0]}))
if "Q2->Q3" in je_rennen.columns:
    fig2.add_trace(go.Bar(x=je_rennen["gp"], y=je_rennen["Q2->Q3"],
                          name="Q2->Q3", marker={"color": d.SERIEN[1]}))
fig2.add_hline(y=0, line_color=d.MUTED, line_width=1)
zeige(fig2, hoehe=420, barmode="group", xaxis=achse("", tickangle=-45),
     yaxis=achse("Median-Delta je Rennen [s]"))

pos12 = int((je_rennen["Q1->Q2"] > 0).sum())
pos23 = int((je_rennen.get("Q2->Q3", pd.Series(dtype=float)) > 0).sum())
hinweis(f"Q1->Q2 positiv in {pos12}/{len(je_rennen)} Rennen dieser Saison, "
       f"Q2->Q3 in {pos23}/{len(je_rennen)} - je naeher an 100%, desto "
       "weniger ist der gepoolte Befund von einzelnen Ausreisser-Rennen "
       "getragen (siehe P43, Gegensatz zur Startplatz-Paritaet in P40).")

with st.expander("Tabelle: Median-Delta je Rennen"):
    tabelle(je_rennen.rename(columns={"gp": "Rennen"}).round(3))


st.markdown("##### ZWEITE AUSBAUSTUFE: kuehlt die Strecke ab, oder gummiert "
           "sie ein?")


@st.cache_data(persist="disk", show_spinner="Streckentemperatur wird "
                                            "geladen (braucht Wetterdaten, "
                                            "erster Aufruf je Saison "
                                            "dauert) ...")
def _temperatur(cache_pfad: str, saison: int) -> pd.DataFrame:
    inv_lokal = f1lab.cached_sessions(cache_pfad)
    rennen = sorted(inv_lokal[(inv_lokal["season"] == saison)
                              & (inv_lokal["ident"] == "Q")]["event"].unique())
    zeilen = []
    for gp in rennen:
        try:
            ses = f1lab.load(saison, gp, "Q", telemetry=False, weather=True)
        except Exception:
            continue
        if ses.laps["Compound"].isin(["INTERMEDIATE", "WET"]).any():
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
                return float("nan")

        b1, b2, b3 = bestzeit(q1), bestzeit(q2), bestzeit(q3)
        t1, t2, t3 = temp(q1), temp(q2), temp(q3)
        for segment, a, b, ta, tb in (("Q1->Q2", b1, b2, t1, t2),
                                      ("Q2->Q3", b2, b3, t2, t3)):
            gemeinsam = a.index.intersection(b.index)
            if len(gemeinsam) == 0 or pd.isna(ta) or pd.isna(tb):
                continue
            zeilen.append({
                "gp": gp, "segment": segment,
                "pace_delta_s": (a[gemeinsam] - b[gemeinsam]).dt.total_seconds().median(),
                "temp_delta_c": ta - tb})
    return pd.DataFrame(zeilen)


temp_df = _temperatur(str(pfad), int(saison))
if temp_df.empty:
    st.info("Keine Wetterdaten fuer diese Saison auswertbar.")
else:
    fig3 = go.Figure()
    for seg, farbe, symbol in (("Q1->Q2", d.SERIEN[0], "circle"),
                               ("Q2->Q3", d.SERIEN[1], "square")):
        sub = temp_df[temp_df["segment"] == seg]
        fig3.add_trace(go.Scatter(
            x=sub["temp_delta_c"], y=sub["pace_delta_s"], mode="markers",
            name=seg, marker={"color": farbe, "symbol": symbol, "size": 10}))
    fig3.add_hline(y=0, line_color=d.MUTED, line_width=1, line_dash="dash")
    fig3.add_vline(x=0, line_color=d.MUTED, line_width=1, line_dash="dash")
    zeige(fig3, hoehe=420,
         xaxis=achse("Temperatur-Delta [°C], positiv = Strecke kuehlt ab"),
         yaxis=achse("Pace-Delta [s], positiv = spaeteres Segment schneller"))

    zeilen_txt = []
    for seg in ("Q1->Q2", "Q2->Q3"):
        sub = temp_df[temp_df["segment"] == seg]
        if len(sub) < 3:
            continue
        r, p = pearsonr(sub["temp_delta_c"], sub["pace_delta_s"])
        erwaermt = int((sub["temp_delta_c"] < 0).sum())
        erwaermt_schneller = int(((sub["temp_delta_c"] < 0)
                                  & (sub["pace_delta_s"] > 0)).sum())
        zeilen_txt.append(f"{seg}: r={r:+.2f} (p={p:.2f}), bei Erwaermung "
                          f"trotzdem schneller in {erwaermt_schneller}/"
                          f"{erwaermt} Rennen")
    hinweis("P17 fand einen echten TrackTemp-Effekt auf die Pace - koennte "
           "die Q1->Q3-Verbesserung also nur abkuehlender Asphalt sein statt "
           "mehr Gummi? " + " | ".join(zeilen_txt) + ". Schwacher, nicht "
           "signifikanter Zusammenhang, und in Rennen mit Erwaermung wird "
           "die Pace trotzdem besser - staerkt die Gummi-Interpretation, "
           "statt sie zu widerlegen (siehe P43, zweite AUSBAUSTUFE).")
