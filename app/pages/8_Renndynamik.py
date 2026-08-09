"""Positionsverlauf, Ueberholungen, Start und Verfolgung ueber ein Rennen.

Drei Reiter, alle auf derselben Session: wer gewinnt/verliert wo Positionen
(P20), wer gewinnt die ersten Meter (P31), und was ein enger Vordermann eine
Rundenzeit kostet (P32). Gerechnet wird nirgends hier - jede Kennzahl kommt
aus f1lab.
"""
from __future__ import annotations

import fastf1.plotting as f1plt
import numpy as np
import plotly.graph_objects as go
import streamlit as st
from common import (
    achse,
    hinweis,
    kein_cache_hinweis,
    kopfzeile,
    lade,
    namensachse,
    nur_rennen,
    setup,
    sidebar_session,
    tabelle,
    zeige,
)

import f1lab
from f1lab import design as d

LINESTYLE = {"-": "solid", "--": "dash", ":": "dot", "-.": "dashdot"}

pfad = setup("Renndynamik", "Positionen, Ueberholungen, Start und "
                           "Verfolgung ueber ein Rennen.")
if kein_cache_hinweis(pfad):
    st.stop()

auswahl = sidebar_session(pfad)
if auswahl is None:
    st.stop()
if nur_rennen(auswahl, "Renndynamik"):
    st.stop()

ses = lade(auswahl, telemetrie=auswahl.telemetrie)
kopfzeile(ses, auswahl)

tab_position, tab_start, tab_verfolgung = st.tabs(
    ["Position & Ueberholungen", "Start", "Verfolgung (Dirty Air)"])


# ================================================== Position & Ueberholungen
with tab_position:
    pos = f1lab.position_progression(ses)
    if pos.empty:
        st.info("Keine Positionsdaten in dieser Session.")
    else:
        st.markdown("##### Positionsverlauf")
        fig = go.Figure()
        for drv in pos.columns:
            style = f1plt.get_driver_style(drv, ["color", "linestyle"],
                                           session=ses)
            fig.add_trace(go.Scatter(
                x=pos.index, y=pos[drv], mode="lines", name=drv,
                line={"color": style["color"], "width": 2,
                     "dash": LINESTYLE.get(style["linestyle"], "solid")},
                hovertemplate=f"<b>{drv}</b><br>Runde %{{x}}<br>"
                              "Position %{y}<extra></extra>"))
        zeige(fig, hoehe=560,
             xaxis=achse("Runde"),
             yaxis=achse("Position", autorange="reversed", dtick=2))

        st.markdown("##### Ueberholungen gesamt")
        with st.spinner("Ueberholmatrix wird berechnet ..."):
            mat = f1lab.overtakes_matrix(ses)
        if mat.empty:
            st.info("Keine auswertbaren Ueberholungen.")
        else:
            gesamt = mat.sum(axis=1).sort_values()
            gesamt = gesamt[gesamt > 0]
            fig = go.Figure(go.Bar(x=gesamt, y=gesamt.index, orientation="h",
                                   marker={"color": d.SERIEN[0]}))
            zeige(fig, hoehe=max(320, 26 * len(gesamt)), showlegend=False,
                 xaxis=achse("Ueberholungen"), yaxis=namensachse())
            hinweis("Ohne Boxenstopp-Effekt und nur auf gruener Flagge - "
                    "ein Safety-Car-Restart oder ein Boxenstopp verschiebt "
                    "Positionen ohne echtes Duell auf der Strecke.")

            stacked = mat.stack()
            stacked = stacked[stacked > 0].sort_values(ascending=False)
            if not stacked.empty:
                top = stacked.head(15).reset_index()
                top.columns = ["Ueberholer", "Ueberholter", "Anzahl"]
                with st.expander("Haeufigste Duelle"):
                    tabelle(top)


# ================================================================== Start
with tab_start:
    if not auswahl.telemetrie:
        st.info("Fuer diese Session liegt keine Telemetrie vor - der Start "
               "braucht sie (Geschwindigkeit ueber die ersten Sekunden).")
    else:
        with st.spinner("Startphase wird fuer alle Fahrer ausgewertet ..."):
            sp = f1lab.start_performance(ses)
        if sp.empty:
            st.info("Keine auswertbaren Starts (nur Boxenstarts, oder keine "
                   "Telemetrie fuer Runde 1).")
        else:
            d_plot = sp.dropna(subset=["m_nach_5s", "gewinn"])
            fig = go.Figure(go.Scatter(
                x=d_plot["m_nach_5s"], y=d_plot["gewinn"], mode="markers+text",
                text=d_plot["driver"], textposition="top center",
                textfont={"color": d.FG, "size": 10},
                marker={"color": d.SERIEN[0], "size": 10}))
            fig.add_hline(y=0, line={"color": d.MUTED, "width": 0.8})
            zeige(fig, hoehe=560, showlegend=False,
                 xaxis=achse("Distanz nach 5 s [m]"),
                 yaxis=achse("Positionsgewinn Runde 1"))
            hinweis("Boxenstarts sind ausgeschlossen - komplett andere "
                    "Ausgangsgeschwindigkeit als ein Ampelstart. Wer weit "
                    "rechts UND deutlich im Minus liegt, hat einen guten "
                    "Start verloren (meist Kurve-1-Chaos, nicht die eigene "
                    "Beschleunigung).")
            with st.expander("Tabelle"):
                tabelle(sp.rename(columns={
                    "driver": "Fahrer", "grid": "Start", "ende_r1": "Ende R1",
                    "gewinn": "Gewinn", "t_100": "bis 100 km/h [s]",
                    "t_200": "bis 200 km/h [s]", "m_nach_5s": "Distanz 5s [m]"}))


# ======================================================= Verfolgung (Dirty Air)
with tab_verfolgung:
    if not auswahl.telemetrie:
        st.info("Fuer diese Session liegt keine Telemetrie vor - die "
               "Verfolgungsanalyse braucht sie (DistanceToDriverAhead).")
    else:
        fahrer = sorted(ses.laps["Driver"].dropna().unique())
        schnellster = str(ses.laps.loc[ses.laps["LapTime"].idxmin(), "Driver"]) \
            if ses.laps["LapTime"].notna().any() else fahrer[0]
        wahl = st.selectbox("Fahrer", fahrer,
                            index=fahrer.index(schnellster) if schnellster in fahrer
                            else 0, key="dirty_air_fahrer")

        with st.spinner(f"Telemetrie von {wahl} wird ausgewertet ..."):
            roh = f1lab.close_following(ses, wahl)
        if roh.empty:
            st.info(f"Keine auswertbaren gruenen Runden fuer {wahl}.")
        else:
            slope, inter, r2, dd = f1lab.dirty_air_effect(roh)
            if dd.empty or len(dd) < 5:
                st.info("Zu wenige Runden fuer eine Regression.")
            else:
                k = st.columns(2)
                k[0].metric("Dirty-Air-Effekt",
                           f"{slope:+.4f} s je 1% Zeit unter 50 m")
                k[1].metric("R²", f"{r2:.3f}",
                           help="Anteil der Streuung, den der Abstand "
                                "erklaert - niedrig heisst: der Effekt ist "
                                "schwach gegenueber der uebrigen Streuung "
                                "einer Rundenzeit.")

                xs = np.linspace(dd["anteil_nah"].min(), dd["anteil_nah"].max(), 50)
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=dd["anteil_nah"], y=dd["sec_corr"], mode="markers",
                    marker={"color": dd["tyre_life"], "colorscale": d.RAMPE_SCALE,
                           "size": 9, "colorbar": {"title": "Reifenalter"}},
                    hovertemplate="Runde %{customdata}<br>%{x:.0f}% unter "
                                 "50m<br>%{y:.2f}s<extra></extra>",
                    customdata=dd["lap"]))
                fig.add_trace(go.Scatter(
                    x=xs, y=slope * xs + inter, mode="lines",
                    line={"color": d.SERIEN[1], "width": 2.5},
                    showlegend=False))
                zeige(fig, hoehe=480, showlegend=False,
                     xaxis=achse("Zeitanteil unter 50 m Abstand [%]"),
                     yaxis=achse("Treibstoff- und degradationsbereinigte "
                                "Rundenzeit [s]"))
                hinweis("Treibstoff- (P04) und degradationsbereinigt (P13) - "
                        "sonst misst die Regression den Sprit- oder "
                        "Reifeneffekt statt Dirty Air, siehe P32.")
