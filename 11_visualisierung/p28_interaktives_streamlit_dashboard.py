"""
P28 - Interaktives Streamlit-Dashboard
======================================

Klickbares Analyse-Tool: Session waehlen, Fahrer vergleichen, Telemetrie und Strategie in Tabs.

Kategorie:   Visualisierung & Apps
Niveau:      Fortgeschritten
Aufwand:     5-6 h
Schwerpunkt: Datenanalyse, Engineering
Zusaetzliche Pakete: streamlit, plotly

WARUM DAS LOHNT
Ein Dashboard, das man live vorfuehren kann, wirkt im Gespraech staerker als jedes Notebook. Streamlit macht das in wenigen Stunden moeglich.

VORGEHEN
  1. st.cache_data um die Session-Ladefunktion legen
  2. Sidebar: Jahr, Event, Session, Fahrerauswahl
  3. Tab 1 Uebersicht, Tab 2 Telemetrie-Overlay, Tab 3 Strategie
  4. Plotly fuer zoombare Charts
  5. Deploy auf Streamlit Community Cloud

GENUTZTE FASTF1-BAUSTEINE
  - fastf1 gesamt
  - streamlit
  - plotly

AUSBAUSTUFE
Ergaenze einen Vergleichsmodus ueber zwei Sessions hinweg (z. B. Quali 2023 vs. 2024 auf derselben Strecke).
"""

# streamlit run f1_dashboard.py
import fastf1
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

fastf1.Cache.enable_cache("~/f1_cache")
st.set_page_config(page_title="F1 Analyse", layout="wide")


@st.cache_data(show_spinner="Lade Session ...")
def load(year, gp, ident):
    s = fastf1.get_session(year, gp, ident)
    s.load()
    laps = s.laps.copy()
    laps["Sec"] = laps["LapTime"].dt.total_seconds()
    return laps, s.results.copy(), s.event["EventName"]


st.sidebar.title("Auswahl")
year = st.sidebar.selectbox("Saison", list(range(2024, 2017, -1)))
sched = fastf1.get_event_schedule(year, include_testing=False)
gp = st.sidebar.selectbox("Event", sched["EventName"].tolist())
ident = st.sidebar.selectbox("Session", ["R", "Q", "FP3", "FP2", "FP1", "S"])

laps, results, name = load(year, gp, ident)
drivers = sorted(laps["Driver"].dropna().unique())
sel = st.sidebar.multiselect("Fahrer", drivers, default=drivers[:2])

st.title(f"{name} {year} - {ident}")
c1, c2, c3 = st.columns(3)
c1.metric("Runden gesamt", len(laps))
c2.metric("Schnellste Runde", f"{laps['Sec'].min():.3f} s")
c3.metric("Fahrer", laps["Driver"].nunique())

tab1, tab2, tab3 = st.tabs(["Rundenzeiten", "Strategie", "Ergebnis"])

with tab1:
    fig = go.Figure()
    for d in sel:
        g = laps[(laps["Driver"] == d) & laps["IsAccurate"]]
        fig.add_trace(go.Scatter(x=g["LapNumber"], y=g["Sec"],
                                 mode="lines+markers", name=d))
    fig.update_layout(xaxis_title="Runde", yaxis_title="Rundenzeit [s]",
                      height=520)
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    stints = (laps.groupby(["Driver", "Stint", "Compound"])["LapNumber"]
              .agg(["min", "max", "count"]).reset_index())
    st.dataframe(stints, use_container_width=True)

with tab3:
    cols = [c for c in ["Position", "Abbreviation", "TeamName",
                        "GridPosition", "Status", "Points"]
            if c in results.columns]
    st.dataframe(results[cols], use_container_width=True)
