"""streckenkarte, bremszonen, fahrervergleich und DRS aus derselben
telemetrie einer session.

vier reiter statt einer langen seite: streckenkarte (ideallinie oder nach
gang/speed/throttle/DRS eingefaerbt), bremszonen, zwei-fahrer-vergleich
(overlay, delta, mini-sektoren) und DRS-nutzung. gerechnet wird nirgends
hier, jede kennzahl kommt aus f1lab, dieselben funktionen wie in den
p06-p10-skripten.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from common import (
    achse,
    hinweis,
    kein_cache_hinweis,
    lade,
    namensachse,
    setup,
    sidebar_session,
    tabelle,
    zeige,
)
from plotly.subplots import make_subplots

import f1lab
from f1lab import design as d

pfad = setup("Telemetrie", "Wie sieht die Strecke aus, wo wird gebremst, und "
                           "wie unterscheiden sich zwei Fahrer?")
if kein_cache_hinweis(pfad):
    st.stop()
assert pfad is not None  # kein_cache_hinweis() haette sonst schon abgebrochen

auswahl = sidebar_session(pfad, nur_mit_telemetrie=True)
if auswahl is None:
    st.stop()

ses = lade(auswahl, telemetrie=True)
st.subheader(auswahl.titel)

runden = ses.laps.dropna(subset=["LapTime"])
if runden.empty:
    st.info("Keine gewerteten Runden in dieser Session.")
    st.stop()

fahrer = sorted(runden["Driver"].unique())
schnellster = str(runden.loc[runden["LapTime"].idxmin(), "Driver"])

tab_karte, tab_bremsen, tab_vergleich, tab_drs = st.tabs(
    ["Streckenkarte", "Bremszonen", "Zwei Fahrer im Vergleich", "DRS"])


# ============================================================ streckenkarte
with tab_karte:
    geo = f1lab.circuit_geometry(ses)
    g = st.columns(4)

    def _zahl(wert, muster):
        return muster.format(wert) if pd.notna(wert) else "\N{EN DASH}"

    g[0].metric("Kurven", _zahl(geo["corners"], "{:.0f}"))
    g[1].metric("Laenge", _zahl(geo["length_m"], "{:.0f} m"))
    g[2].metric("Anstieg gesamt", _zahl(geo["elev_gain_m"], "{:.0f} m"))
    g[3].metric("Hoehenunterschied", _zahl(geo["elev_span_m"], "{:.0f} m"))
    hinweis("Laenge und Hoehe stammen aus den Positionsdaten der schnellsten "
            "Runde - die gefahrene Ideallinie, nicht die offizielle "
            "Streckenlaenge (schneidet Kurven, faellt rund 1% kuerzer aus).")

    c1, c2 = st.columns([1, 1])
    fahrer_karte = c1.selectbox("Fahrer (schnellste Runde)", fahrer,
                                index=fahrer.index(schnellster), key="tk_fahrer")
    kanal = c2.selectbox("Einfaerben nach",
                         ["Nur Linie", "Gang", "Speed", "Throttle", "DRS"],
                         key="tk_kanal")

    lap = runden[runden["Driver"] == fahrer_karte].pick_fastest()
    try:
        tel = None if pd.isna(lap["LapTime"]) else lap.get_telemetry()
    except Exception:
        tel = None

    if tel is None or tel.empty:
        st.info(f"Keine Telemetrie fuer {fahrer_karte} verfuegbar.")
    else:
        x, y = tel["X"] / 10, tel["Y"] / 10
        achsen_kw = {"xaxis": {"visible": False},
                    "yaxis": {"visible": False, "scaleanchor": "x",
                             "scaleratio": 1}}

        if kanal == "Nur Linie":
            fig = go.Figure(go.Scatter(
                x=x, y=y, mode="lines",
                line={"color": d.SERIEN[0], "width": 3}, hoverinfo="skip"))
            zeige(fig, hoehe=520, showlegend=False, **achsen_kw)
            hinweis(f"Ideallinie der schnellsten Runde ({fahrer_karte}).")
        elif kanal == "Gang":
            fig = go.Figure(go.Scatter(
                x=x, y=y, mode="markers",
                marker={"color": tel["nGear"], "colorscale": "Viridis",
                       "size": 6, "colorbar": {"title": "Gang",
                                               "tickvals": list(range(1, 9))}},
                hoverinfo="skip"))
            zeige(fig, hoehe=520, showlegend=False, **achsen_kw)
            hinweis("Gang bekommt bewusst keine der Hausfarben-Rampen: sieben "
                    "gleich wichtige Zustaende sind mit einer einfarbigen "
                    "Rampe kaum zu unterscheiden (siehe P09).")
        elif kanal in ("Speed", "Throttle"):
            werte = tel["Speed"] if kanal == "Speed" else tel["Throttle"]
            einheit = "km/h" if kanal == "Speed" else "%"
            fig = go.Figure(go.Scatter(
                x=x, y=y, mode="markers",
                marker={"color": werte, "colorscale": d.RAMPE_SCALE,
                       "size": 6, "colorbar": {"title": einheit}},
                hoverinfo="skip"))
            zeige(fig, hoehe=520, showlegend=False, **achsen_kw)
            hinweis(f"{kanal} der schnellsten Runde ({fahrer_karte}).")
        else:  # DRS
            zustand = f1lab.drs_state(tel["DRS"].to_numpy())
            palette = np.array([d.MUTED, d.SERIEN[1], d.POSITIV])
            fig = go.Figure(go.Scatter(
                x=x, y=y, mode="markers",
                marker={"color": palette[zustand], "size": 6},
                hoverinfo="skip"))
            for name, farbe in zip(["zu", "erkannt", "offen"], palette,
                                   strict=True):
                fig.add_trace(go.Scatter(
                    x=[None], y=[None], mode="markers",
                    marker={"color": farbe, "size": 10}, name=name))
            zeige(fig, hoehe=520, **achsen_kw)
            hinweis("Codes unterhalb von 10 sind laut FastF1s eigener "
                    "Dokumentation nicht abschliessend geklaert - 'erkannt' "
                    "ist die in der Community uebliche Lesart von Code 8.")


# ============================================================== bremszonen
with tab_bremsen:
    with st.sidebar:
        st.header("Bremszonen")
        wahl = st.selectbox("Fahrer", fahrer, index=fahrer.index(schnellster),
                            help="Ausgewertet wird die schnellste Runde "
                                 "dieses Fahrers.", key="bz_fahrer")
        vergleich_an = st.checkbox("Mit zweitem Fahrer vergleichen",
                                   key="bz_vergleich")
        wahl2 = None
        if vergleich_an:
            andere = [f for f in fahrer if f != wahl]
            wahl2 = st.selectbox("Zweiter Fahrer", andere, key="bz_fahrer2")
        min_laenge = st.slider(
            "Mindestlaenge einer Zone [m]", 5, 80, 20, 5,
            help="Kuerzere Bremsungen sind meist Messrauschen oder ein "
                 "kurzes Antippen zur Balance, keine echte Bremszone.")

    lap = runden[runden["Driver"] == wahl].pick_fastest()
    zonen = pd.DataFrame() if pd.isna(lap["LapTime"]) else \
        f1lab.driver_braking_zones(ses, wahl, min_length_m=min_laenge)

    if zonen.empty:
        st.info(f"Keine Zone ueberschreitet die Mindestlaenge fuer {wahl}. "
               "Schwelle senken.")
    else:
        car = lap.get_car_data().add_distance()
        k = st.columns(4)
        k[0].metric("Rundenzeit", f"{lap['LapTime'].total_seconds():.3f} s")
        k[1].metric("Bremszonen", len(zonen))
        k[2].metric("Haerteste Bremsung", f"{zonen['decel_g'].max():.2f} g")
        k[3].metric("Gebremste Strecke", f"{zonen['length_m'].sum():.0f} m",
                    f"{100 * zonen['length_m'].sum() / car['Distance'].max():.0f} "
                    f"% der Runde")

        fig = go.Figure()
        for zone in zonen.itertuples():
            fig.add_vrect(x0=zone.start_m, x1=zone.end_m, fillcolor=d.SERIEN[1],
                          opacity=0.2, line_width=0, layer="below")
        fig.add_trace(go.Scatter(
            x=car["Distance"], y=car["Speed"], mode="lines", name=wahl,
            line={"color": d.SERIEN[0], "width": 2},
            hovertemplate="%{x:.0f} m<br>%{y:.0f} km/h<extra></extra>"))

        zeige(fig, hoehe=420, showlegend=False,
             xaxis=achse("Distanz [m]"), yaxis=achse("km/h"))
        hinweis("Die markierten Baender sind die erkannten Bremszonen. Ihre "
                "Grenzen kommen aus den Flanken des binaeren Bremssignals, "
                "nicht aus der Geschwindigkeit.")

        links, rechts = st.columns([3, 2])
        with links:
            fig = go.Figure(go.Bar(
                x=zonen.index + 1, y=zonen["decel_g"],
                marker={"color": d.SERIEN[0], "line": {"color": d.BG, "width": 2}},
                customdata=zonen[["v_entry_kmh", "v_min_kmh", "length_m",
                                  "duration_s", "start_m"]],
                hovertemplate="Zone %{x} bei %{customdata[4]:.0f} m<br>"
                              "%{y:.2f} g mittlere Verzoegerung<br>"
                              "%{customdata[0]:.0f} auf %{customdata[1]:.0f} km/h<br>"
                              "%{customdata[2]:.0f} m in %{customdata[3]:.2f} s"
                              "<extra></extra>"))
            zeige(fig, hoehe=380, showlegend=False,
                 xaxis=achse("Bremszone in Reihenfolge der Runde", dtick=1),
                 yaxis=achse("g"))
        with rechts:
            st.markdown("##### Zonen")
            tabelle(zonen.rename(columns={
                "start_m": "ab [m]", "end_m": "bis [m]", "length_m": "Laenge [m]",
                "v_entry_kmh": "Eintritt [km/h]", "v_min_kmh": "Minimum [km/h]",
                "duration_s": "Dauer [s]", "decel_g": "Verzoegerung [g]"}),
                height=380)

        if vergleich_an and wahl2:
            zonen2 = f1lab.driver_braking_zones(ses, wahl2, min_length_m=min_laenge)
            vgl = f1lab.compare_braking_zones(zonen, zonen2)
            st.markdown(f"##### Bremspunkte {wahl} gegen {wahl2}")
            if vgl.empty:
                st.info("Keine gemeinsam zuordenbaren Bremszonen gefunden.")
            else:
                vgl = vgl.assign(spaeter=np.where(
                    vgl["delta_m"] > 0, wahl2, wahl))
                tabelle(vgl.rename(columns={
                    "start_m_a": f"{wahl} [m]", "start_m_b": f"{wahl2} [m]",
                    "delta_m": "Differenz [m]", "spaeter": "bremst spaeter"}))
                hinweis("Zonen werden ueber die naechstgelegene Distanz "
                        "gepaart (150 m Toleranz) - beide Runden umrunden "
                        "dieselbe Strecke, der Bremspunkt fuer dieselbe "
                        "Kurve liegt also nah beieinander.")


# ==================================================== zwei fahrer im vergleich
with tab_vergleich:
    c1, c2 = st.columns(2)
    v1 = c1.selectbox("Fahrer 1", fahrer, index=fahrer.index(schnellster),
                      key="cmp_d1")
    andere = [f for f in fahrer if f != v1]
    v2 = c2.selectbox("Fahrer 2", andere, index=0, key="cmp_d2")

    lap1 = runden[runden["Driver"] == v1].pick_fastest()
    lap2 = runden[runden["Driver"] == v2].pick_fastest()

    if pd.isna(lap1["LapTime"]) or pd.isna(lap2["LapTime"]):
        st.info("Fuer einen der beiden Fahrer liegt keine schnellste Runde vor.")
    else:
        try:
            t1 = lap1.get_car_data().add_distance()
            t2 = lap2.get_car_data().add_distance()
        except Exception:
            t1 = t2 = None

        if t1 is None or t2 is None or t1.empty or t2.empty:
            st.info("Telemetrie fuer einen der beiden Fahrer fehlt.")
        else:
            import warnings

            from fastf1.utils import delta_time
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", FutureWarning)
                delta, ref, _cmp = delta_time(lap1, lap2)

            farbe1, farbe2 = d.SERIEN[0], d.SERIEN[1]
            fig = make_subplots(rows=5, cols=1, shared_xaxes=True,
                                vertical_spacing=0.03,
                                row_heights=[0.32, 0.17, 0.17, 0.12, 0.12])
            fig.add_trace(go.Scatter(x=t1["Distance"], y=t1["Speed"], name=v1,
                                     line={"color": farbe1}), row=1, col=1)
            fig.add_trace(go.Scatter(x=t2["Distance"], y=t2["Speed"], name=v2,
                                     line={"color": farbe2}), row=1, col=1)
            fig.add_trace(go.Scatter(x=ref["Distance"], y=delta, name="Delta",
                                     line={"color": d.FG}, showlegend=False),
                         row=2, col=1)
            fig.add_trace(go.Scatter(x=t1["Distance"], y=t1["Throttle"],
                                     line={"color": farbe1}, showlegend=False),
                         row=3, col=1)
            fig.add_trace(go.Scatter(x=t2["Distance"], y=t2["Throttle"],
                                     line={"color": farbe2}, showlegend=False),
                         row=3, col=1)
            fig.add_trace(go.Scatter(x=t1["Distance"],
                                     y=t1["Brake"].astype(int),
                                     line={"color": farbe1}, showlegend=False),
                         row=4, col=1)
            fig.add_trace(go.Scatter(x=t2["Distance"],
                                     y=t2["Brake"].astype(int),
                                     line={"color": farbe2}, showlegend=False),
                         row=4, col=1)
            fig.add_trace(go.Scatter(x=t1["Distance"], y=t1["nGear"],
                                     line={"color": farbe1}, showlegend=False),
                         row=5, col=1)
            fig.add_trace(go.Scatter(x=t2["Distance"], y=t2["nGear"],
                                     line={"color": farbe2}, showlegend=False),
                         row=5, col=1)

            fig.update_layout(
                height=760, paper_bgcolor=d.BG, plot_bgcolor=d.BG,
                font={"color": d.FG, "size": 13},
                margin={"l": 60, "r": 24, "t": 24, "b": 50},
                legend={"orientation": "h", "y": 1.04, "x": 0,
                       "bgcolor": "rgba(0,0,0,0)"},
                hoverlabel={"bgcolor": d.BG_HELL, "bordercolor": d.GRID,
                           "font": {"color": d.FG}})
            fig.update_xaxes(gridcolor=d.GRID, zeroline=False,
                             linecolor=d.GRID, tickfont={"color": d.MUTED})
            fig.update_yaxes(gridcolor=d.GRID, zeroline=False,
                             linecolor=d.GRID, tickfont={"color": d.MUTED})
            fig.update_yaxes(title_text="km/h", row=1, col=1)
            fig.update_yaxes(title_text=f"Delta {v2}-{v1} [s]", row=2, col=1)
            fig.update_yaxes(title_text="Throttle %", row=3, col=1)
            fig.update_yaxes(title_text="Brake", row=4, col=1,
                             tickvals=[0, 1])
            fig.update_yaxes(title_text="Gang", row=5, col=1)
            fig.update_xaxes(title_text="Distanz [m]", row=5, col=1)
            st.plotly_chart(fig, width="stretch")
            hinweis("delta_time() gilt laut FastF1 als ungenau - fuer den "
                    "Streckenverlauf trotzdem die richtige Wahl, da drei "
                    "Sektorzeiten keinen Verlauf zeigen koennen (siehe P07).")

            q1 = f1lab.telemetry_source_quality(lap1.get_telemetry()["Source"])
            q2 = f1lab.telemetry_source_quality(lap2.get_telemetry()["Source"])
            with st.expander("Telemetrie-Datenqualitaet (ZWEITE AUSBAUSTUFE)"):
                st.caption("`Telemetry['Source']` (bislang ungenutzt) zeigt, "
                          "welcher Anteil der Punkte echte Messwerte statt "
                          "synthetischer Fuellstellen beim Zusammenfuehren von "
                          "car_data/pos_data ist (siehe P07).")
                kq = st.columns(2)
                kq[0].metric(f"{v1}: interpoliert", f"{q1['interpolation']:.1%}",
                           help=f"n={q1['n']} Punkte")
                kq[1].metric(f"{v2}: interpoliert", f"{q2['interpolation']:.1%}",
                           help=f"n={q2['n']} Punkte")

            zonen1 = f1lab.driver_braking_zones(ses, v1)
            zonen2 = f1lab.driver_braking_zones(ses, v2)
            vgl = f1lab.compare_braking_zones(zonen1, zonen2)
            if not vgl.empty:
                st.markdown("###### Bremspunkte im Vergleich")
                vgl = vgl.assign(spaeter=np.where(vgl["delta_m"] > 0, v2, v1))
                tabelle(vgl.rename(columns={
                    "start_m_a": f"{v1} [m]", "start_m_b": f"{v2} [m]",
                    "delta_m": "Differenz [m]", "spaeter": "bremst spaeter"}))

            st.markdown("###### Mini-Sektoren")
            mini_fahrer = st.multiselect(
                f"Fahrer fuer Mini-Sektoren (hoechstens {d.MAX_SERIEN})",
                fahrer, default=[v1, v2], max_selections=d.MAX_SERIEN,
                key="mini_fahrer")
            if len(mini_fahrer) < 2:
                st.info("Waehle mindestens zwei Fahrer.")
            else:
                r = f1lab.mini_sectors(ses, mini_fahrer, n=25)
                if r["gewinner"] is None:
                    st.info("Nicht genug gueltige Runden fuer einen "
                           "Mini-Sektor-Vergleich.")
                else:
                    farbe_je_fahrer = dict(zip(mini_fahrer, d.SERIEN,
                                               strict=False))
                    ref_drv = mini_fahrer[0]
                    ref_tel = r["telemetrie"][ref_drv]
                    edges, gewinner = r["edges"], r["gewinner"]
                    bin_idx = np.clip(
                        np.digitize(ref_tel["Distance"], edges) - 1,
                        0, len(gewinner) - 1)
                    farben = [farbe_je_fahrer[gewinner[i]] for i in bin_idx]

                    fig = go.Figure(go.Scatter(
                        x=ref_tel["X"], y=ref_tel["Y"], mode="markers",
                        marker={"color": farben, "size": 6}, hoverinfo="skip"))
                    for f in mini_fahrer:
                        n_gewonnen = int((gewinner == f).sum())
                        fig.add_trace(go.Scatter(
                            x=[None], y=[None], mode="markers",
                            marker={"color": farbe_je_fahrer[f], "size": 10},
                            name=f"{f} ({n_gewonnen})"))
                    zeige(fig, hoehe=520,
                         xaxis={"visible": False},
                         yaxis={"visible": False, "scaleanchor": "x",
                               "scaleratio": 1})
                    hinweis(f"{len(gewinner)} gleich lange Distanzabschnitte, "
                            f"eingefaerbt nach dem schnellsten der gewaehlten "
                            f"Fahrer je Abschnitt (Referenzlinie {ref_drv}).")


# ===================================================================== DRS
with tab_drs:
    with st.spinner("DRS-Kanal wird fuer alle Fahrer ausgewertet ..."):
        nutzung = f1lab.drs_usage(ses)

    if nutzung.empty:
        st.info("Keine DRS-Daten in dieser Session.")
    else:
        st.markdown("##### DRS-Zeitanteil")
        p = nutzung.sort_values("drs_pct")
        fig = go.Figure(go.Bar(x=p["drs_pct"], y=p["driver"], orientation="h",
                               marker={"color": d.SERIEN[0]}))
        zeige(fig, hoehe=max(320, 26 * len(p)), showlegend=False,
             xaxis=achse("DRS offen [% der Runde]"), yaxis=namensachse())
        hinweis("Zeitanteil der schnellsten Runde mit offenem DRS (Code "
                "10/12/14 im Feed).")

        st.markdown("##### Topspeed-Wirkung")
        wirkung = nutzung.dropna(subset=["gewinn_kmh"]).sort_values("gewinn_kmh")
        fig = go.Figure(go.Bar(x=wirkung["gewinn_kmh"], y=wirkung["driver"],
                               orientation="h", marker={"color": d.SERIEN[0]}))
        zeige(fig, hoehe=max(320, 26 * len(wirkung)), showlegend=False,
             xaxis=achse("Topspeed-Gewinn durch DRS [km/h]"),
             yaxis=namensachse())

        st.markdown("##### DRS-Zonen")
        wahl_drs = st.selectbox("Fahrer", fahrer,
                                index=fahrer.index(schnellster), key="drs_fahrer")
        zonen = f1lab.drs_zones(ses, wahl_drs)
        if zonen.empty:
            st.info(f"Keine DRS-Zone fuer {wahl_drs} erkannt (evtl. nasse "
                   "Session oder DRS deaktiviert).")
        else:
            tabelle(zonen.rename(columns={
                "start_m": "Start [m]", "end_m": "Ende [m]",
                "length_m": "Laenge [m]"}))
