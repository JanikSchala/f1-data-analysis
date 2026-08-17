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
from scipy.stats import wilcoxon

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
