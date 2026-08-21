"""ueberholschwierigkeit je strecke (P38): ueberholungen einer ganzen
saison gegen die streckengeometrie der telemetriefaehigen rennen.

anders als der rest der app an keine in der seitenleiste waehlbare session
gebunden, sondern an eine ganze saison (grund wie in 9_Teamkollegen.py/
11_Boxenstopps.py/14_Historie.py/15_MachineLearning.py). der saison-scan
bleibt deshalb seitenlokal statt in f1lab. gerechnet wird trotzdem nirgends
hier ausser der korrelation selbst: die bausteine kommen aus
f1lab.overtakes_matrix() (P20), f1lab.circuit_dimension() (P02) und (zweite
AUSBAUSTUFE) f1lab.track_status_phases() (P18). dieselbe streckengeometrie
gegen eine zweite rennverlaufs-groesse zeigt, dass nicht jede statistik
automatisch eine streckeneigenschaft ist.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from common import (
    achse,
    hinweis,
    inventar,
    kein_cache_hinweis,
    namensachse,
    setup,
    tabelle,
    zeige,
)
from scipy.stats import pearsonr, spearmanr

import f1lab
from f1lab import design as d

pfad = setup("Ueberholschwierigkeit", "Ueberholungen einer ganzen Saison "
                                      "gegen die Streckengeometrie - ist "
                                      "'hier kann man nicht ueberholen' "
                                      "eine Streckeneigenschaft?")
if kein_cache_hinweis(pfad):
    st.stop()

inv = inventar(str(pfad))
saisons = sorted(inv[inv["ident"] == "R"]["season"].unique(), reverse=True)
with st.sidebar:
    st.header("Saison")
    saison = st.selectbox("Jahr", saisons, key="ueberhol_saison")


@st.cache_data(persist="disk", show_spinner="Ueberholungen werden gezaehlt "
                                            "(erster Aufruf je Saison dauert) ...")
def _ueberholungen(cache_pfad: str, saison: int) -> pd.DataFrame:
    inv_lokal = f1lab.cached_sessions(cache_pfad)
    rennen = sorted(inv_lokal[(inv_lokal["season"] == saison)
                              & (inv_lokal["ident"] == "R")]["event"].unique())
    zeilen = []
    for gp in rennen:
        try:
            ses = f1lab.load(saison, gp, "R", telemetry=False)
        except Exception:
            continue
        zeilen.append({"gp": gp, "overtakes": int(
            f1lab.overtakes_matrix(ses).values.sum())})
    return pd.DataFrame(zeilen)


@st.cache_data(persist="disk", show_spinner="Streckengeometrie wird geladen "
                                            "(erster Aufruf je Saison dauert) ...")
def _geometrie(cache_pfad: str, saison: int) -> pd.DataFrame:
    inv_lokal = f1lab.cached_sessions(cache_pfad)
    tel_rennen = sorted(inv_lokal[(inv_lokal["season"] == saison)
                                  & (inv_lokal["ident"] == "R")
                                  & inv_lokal["telemetry"]]["event"].unique())
    if not tel_rennen:
        return pd.DataFrame()
    return f1lab.circuit_dimension([(saison, gp) for gp in tel_rennen])


ueberholungen = _ueberholungen(str(pfad), int(saison))
if ueberholungen.empty:
    st.info(f"Keine auswertbaren Rennen fuer {saison} im Cache.")
    st.stop()

k = st.columns(3)
k[0].metric("Rennen", len(ueberholungen))
k[1].metric("Ueberholungen gesamt", int(ueberholungen["overtakes"].sum()))
k[2].metric("Spannweite je Rennen",
           f"{ueberholungen['overtakes'].min()}"
           f"\N{EN DASH}{ueberholungen['overtakes'].max()}")

st.markdown("##### Ueberholungen je Rennen")
u = ueberholungen.sort_values("overtakes")
fig = go.Figure(go.Bar(
    x=u["overtakes"], y=u["gp"], orientation="h",
    marker={"color": d.SERIEN[0]}))
zeige(fig, hoehe=max(320, 26 * len(u)), showlegend=False,
     xaxis=achse("Ueberholungen (gruene Flagge, ohne Boxenstopp-Effekt)"),
     yaxis=namensachse())
hinweis("Aus f1lab.overtakes_matrix() (P20): zaehlt nur Positionswechsel auf "
       "gruener Flagge, ohne Boxenstopp- oder Safety-Car-Restart-Effekt. "
       "Braucht keine Telemetrie - deshalb fuer die ganze Saison verfuegbar, "
       "anders als die Streckengeometrie unten.")

st.markdown("##### Gegen Streckengeometrie (nur Rennen mit Telemetrie)")
geo = _geometrie(str(pfad), int(saison))
if geo.empty or len(geo) < 4:
    st.info(f"Fuer {saison} liegt Telemetrie fuer nur {len(geo)} Rennen im "
           "Cache - fuer eine Korrelation werden mindestens 4 gebraucht.")
    st.stop()

geo = geo.merge(ueberholungen, on="gp", how="inner")
geo["kurven_pro_km"] = geo["corners"] / (geo["length_m"] / 1000)

r_kpk, p_kpk = pearsonr(geo["kurven_pro_km"], geo["overtakes"])
r_sp, p_sp = spearmanr(geo["kurven_pro_km"], geo["overtakes"])

fig2 = go.Figure()
fig2.add_trace(go.Scatter(
    x=geo["kurven_pro_km"], y=geo["overtakes"], mode="markers+text",
    text=geo["circuit"], textposition="top center",
    textfont={"color": d.MUTED, "size": 10},
    marker={"size": 11, "color": d.SERIEN[0]}))
steigung, achse0 = np.polyfit(geo["kurven_pro_km"], geo["overtakes"], 1)
x_fit = np.linspace(geo["kurven_pro_km"].min(), geo["kurven_pro_km"].max(), 50)
fig2.add_trace(go.Scatter(x=x_fit, y=steigung * x_fit + achse0, mode="lines",
                          line={"color": d.MUTED, "dash": "dash"}))
zeige(fig2, hoehe=420, showlegend=False, xaxis=achse("Kurven pro Kilometer"),
     yaxis=achse("Ueberholungen"))

hinweis(f"Pearson r={r_kpk:+.2f} (p={p_kpk:.3f}), Spearman r={r_sp:+.2f} "
       f"(p={p_sp:.3f}) ueber {len(geo)} Strecken: mehr Kurven auf derselben "
       "Streckenlaenge haengt mit weniger Ueberholungen zusammen. Streckenlaenge "
       "allein sagt dagegen praktisch nichts voraus (siehe P38). Bei dieser "
       "Stichprobengroesse ist ein einzelner Ausreisser (typisch Monaco) "
       "einflussreich - Spearman als ausreisserrobuster Gegencheck steht "
       "deshalb daneben, nicht nur Pearson.")

with st.expander("Was diese Korrelation nicht zeigt"):
    st.markdown(
        "Ueberholzahlen haengen stark vom Rennverlauf ab, nicht nur von der "
        "Streckenform - ein chaotisches Rennen mit grossen Reifen- oder "
        "Strategieunterschieden erzeugt auf derselben Strecke deutlich mehr "
        "Ueberholungen als ein sauberes. Diese Zahl misst eine Saison, keine "
        "Streckeneigenschaft im Vakuum (siehe P38).")


st.markdown("##### Zweite AUSBAUSTUFE: ist Safety-Car-Haeufigkeit dasselbe?")


@st.cache_data(persist="disk", show_spinner="Safety-Car-/VSC-Phasen werden "
                                            "gezaehlt (erster Aufruf je "
                                            "Saison dauert) ...")
def _sicherheitsauto(cache_pfad: str, saison: int, rennen: tuple[str, ...]
                     ) -> pd.DataFrame:
    zeilen = []
    for gp in rennen:
        s = f1lab.load(saison, gp, "R", telemetry=False)
        ph = f1lab.track_status_phases(s)
        neutral = ph[ph["label"].isin(["safety car", "vsc"])]
        zeilen.append({"gp": gp, "sc_phasen": len(neutral),
                       "sc_dauer_s": round(float(neutral["duration_s"].sum()), 0)})
    return pd.DataFrame(zeilen)


sc = _sicherheitsauto(str(pfad), int(saison), tuple(geo["gp"]))
geo = geo.merge(sc, on="gp", how="inner")
r_sc, p_sc = pearsonr(geo["kurven_pro_km"], geo["sc_dauer_s"])
r_sc_sp, p_sc_sp = spearmanr(geo["kurven_pro_km"], geo["sc_dauer_s"])

fig3 = go.Figure(go.Scatter(
    x=geo["kurven_pro_km"], y=geo["sc_dauer_s"], mode="markers+text",
    text=geo["circuit"], textposition="top center",
    textfont={"color": d.MUTED, "size": 10},
    marker={"size": 11, "color": d.SERIEN[2]}))
zeige(fig3, hoehe=380, showlegend=False, xaxis=achse("Kurven pro Kilometer"),
     yaxis=achse("Safety-Car-/VSC-Dauer [s]"))
hinweis(f"Pearson r={r_sc:+.2f} (p={p_sc:.2f}), Spearman r={r_sc_sp:+.2f} "
       f"(p={p_sc_sp:.2f}) - praktisch kein Zusammenhang, anders als bei "
       "Ueberholungen oben. Eine kurvenreiche Strecke ueberholt schwerer, "
       "ist dadurch aber nicht automatisch unfallanfaelliger - Safety-Car-"
       "Ausloeser (Kollisionen, technische Ausfaelle, Wetter) haengen eher "
       "an Feldgroesse und Zuverlaessigkeit als an der Streckenform "
       "(siehe P38, zweite AUSBAUSTUFE).")

tabelle(geo[["gp", "circuit", "corners", "length_m", "kurven_pro_km",
            "overtakes", "sc_dauer_s"]].round(2).rename(columns={
    "gp": "Rennen", "circuit": "Strecke", "corners": "Kurven",
    "length_m": "Laenge [m]", "kurven_pro_km": "Kurven/km",
    "overtakes": "Ueberholungen", "sc_dauer_s": "SC/VSC [s]"}))
