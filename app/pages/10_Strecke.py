"""Streckenkarte mit Kurven und Marshal-Sektoren, Kurvengeschwindigkeit je
Fahrer, und derselbe Rundkurs ueber Jahre verglichen.

Drei Reiter aus P11/P12/P33 - gerechnet wird nirgends hier, jede Kennzahl
kommt aus f1lab.corner_labels()/corner_speeds()/marshal_sector_labels().
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
    namensachse,
    setup,
    sidebar_session,
    tabelle,
    zeige,
)

import f1lab
from f1lab import design as d

pfad = setup("Strecke", "Kurven, Marshal-Sektoren, Kurvengeschwindigkeit je "
                        "Fahrer - und derselbe Rundkurs ueber Jahre.")
if kein_cache_hinweis(pfad):
    st.stop()

auswahl = sidebar_session(pfad, nur_mit_telemetrie=True)
if auswahl is None:
    st.stop()

ses = f1lab.load(auswahl.season, auswahl.event, auswahl.ident, telemetry=True)
st.subheader(auswahl.titel)

tab_karte, tab_speed, tab_jahre = st.tabs(
    ["Streckenkarte", "Kurvengeschwindigkeit", "Ueber Jahre verglichen"])


# ==================================================================== Karte
with tab_karte:
    geo = f1lab.circuit_geometry(ses)

    def _zahl(schluessel: str, muster: str) -> str:
        wert = geo.get(schluessel)
        return muster.format(wert) if pd.notna(wert) else "\N{EN DASH}"

    g = st.columns(4)
    g[0].metric("Kurven", _zahl("corners", "{:.0f}"))
    g[1].metric("Laenge", _zahl("length_m", "{:.0f} m"))
    g[2].metric("Anstieg gesamt", _zahl("elev_gain_m", "{:.0f} m"))
    g[3].metric("Hoehenunterschied", _zahl("elev_span_m", "{:.0f} m"))

    ref_lap = ses.laps.pick_fastest()
    try:
        ref = ref_lap.get_telemetry().add_distance()
    except Exception:
        ref = None

    if ref is None or ref.empty:
        st.info("Keine Telemetrie fuer die schnellste Runde verfuegbar.")
    else:
        corners = f1lab.corner_labels(ses)
        sektoren = f1lab.marshal_sector_labels(ses)
        lichter = f1lab.marshal_light_labels(ses)

        x, y = ref["X"] / 10, ref["Y"] / 10
        dist = ref["Distance"].to_numpy()
        grenzen = sektoren["distance"].to_numpy()
        sektor_idx = np.searchsorted(grenzen, dist, side="right")
        farben = [d.BG_HELL if i % 2 == 0 else d.MUTED for i in sektor_idx]

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=x, y=y, mode="markers",
            marker={"color": farben, "size": 7}, hoverinfo="skip",
            showlegend=False))

        ref_xy = ref[["X", "Y"]].to_numpy(dtype=float) / 10
        for c in corners.itertuples():
            i = int(np.argmin(np.abs(dist - c.Distance)))
            fig.add_trace(go.Scatter(
                x=[ref_xy[i, 0]], y=[ref_xy[i, 1]], mode="markers+text",
                marker={"color": d.SERIEN[1], "size": 16},
                text=[c.label], textposition="top center",
                textfont={"color": d.FG, "size": 10}, hoverinfo="skip",
                showlegend=False))

        # ZWEITE AUSBAUSTUFE: Gelbflaggen-Lichttafeln (CircuitInfo.
        # marshal_lights, bislang im ganzen Repo ungenutzt), siehe P11.
        licht_idx = [int(np.argmin(np.abs(dist - lp.distance)))
                    for lp in lichter.itertuples()]
        fig.add_trace(go.Scatter(
            x=ref_xy[licht_idx, 0], y=ref_xy[licht_idx, 1], mode="markers",
            marker={"color": d.PHASE["gelb"], "size": 10, "symbol": "triangle-up",
                   "line": {"color": d.FG, "width": 1}},
            name="Lichttafel", hoverinfo="skip"))

        zeige(fig, hoehe=560, showlegend=False, xaxis={"visible": False},
             yaxis={"visible": False, "scaleanchor": "x", "scaleratio": 1})
        hinweis(f"Referenzrunde: {ref_lap['Driver']}. Abwechselnd eingefaerbte "
               f"Abschnitte sind die {len(sektoren)} Marshal-Sektoren, nicht "
               "die Kurven - beides kommt aus get_circuit_info(), aber es "
               f"sind zwei verschiedene Einteilungen. Die {len(lichter)} "
               "gelben Dreiecke sind die Gelbflaggen-Lichttafeln (siehe "
               "P11).")

        st.markdown("##### Minimalgeschwindigkeit je Kurve (Referenzfahrer)")
        speeds = f1lab.corner_speeds(ses)
        if str(ref_lap["Driver"]) in speeds.index:
            tab = corners[["label", "Distance"]].copy()
            tab["v_min_kmh"] = (speeds.loc[str(ref_lap["Driver"])]
                                .reindex(corners["label"]).to_numpy())
            tabelle(tab.round(1).rename(columns={
                "label": "Kurve", "Distance": "Distanz [m]",
                "v_min_kmh": "v min [km/h]"}), height=300)


# ============================================================== Kurvenspeed
with tab_speed:
    st.markdown("##### Kurvengeschwindigkeit im Feld, relativ zum Schnellsten")
    mat = f1lab.corner_speeds(ses)
    if mat.empty:
        st.info("Keine Kurvengeschwindigkeiten fuer diese Session berechenbar.")
    else:
        rel = mat.sub(mat.max(axis=0), axis=1)
        order = rel.mean(axis=1).sort_values(ascending=False).index
        rel = rel.loc[order]
        vmin = float(np.percentile(rel.to_numpy(), 10))

        fig = go.Figure(go.Heatmap(
            z=rel.to_numpy(), x=list(rel.columns), y=list(rel.index),
            colorscale=d.RAMPE_SCALE, zmin=vmin, zmax=0,
            colorbar={"title": "Delta [km/h]"},
            hovertemplate="%{y} %{x}<br>%{z:.1f} km/h<extra></extra>"))
        zeige(fig, hoehe=max(420, 24 * len(rel)), xaxis=namensachse(),
             yaxis=namensachse())
        hinweis("Auf dem 10. Perzentil gekappt statt auf das absolute Minimum "
               "(siehe P12) - eine einzelne unruhige Kurve wuerde sonst allein "
               "die Farbskala strecken. Die Reihenfolge der Fahrer ist nach "
               "mittlerem Rueckstand ueber alle Kurven, nicht nach Position.")

        with st.expander("Warum das keine Setup-Typen trennt"):
            st.markdown(
                "Die naheliegende Erwartung waere, dass Fahrer in zwei Lager "
                "zerfallen - staerker in langsamen oder staerker in schnellen "
                "Kurven, je nach Abtriebsphilosophie des Autos. Die "
                "Nachpruefung in P12 widerlegt das: die Korrelation zwischen "
                "dem eigenen Kurvenprofil und der Kurvengeschwindigkeit ist "
                "fuer fast das ganze Feld negativ und kontinuierlich verteilt, "
                "kein Bruch, an dem sich zwei Gruppen trennen liessen. Ein "
                "k-Means-Clustering trennt stattdessen nach Streuung "
                "(gleichmaessiges gegen unruhiges Profil), nicht nach "
                "Kurventyp.")


# ============================================================= Jahresvergleich
with tab_jahre:
    st.markdown("##### Pole-Runde derselben Strecke, verschiedene Jahre")
    inv = f1lab.cached_sessions(str(pfad))
    passend = inv[(inv["event"] == auswahl.event) & (inv["ident"] == "Q")
                 & inv["telemetry"]]
    jahre = sorted(passend["season"].unique())

    if len(jahre) < 2:
        st.info(f"Fuer {auswahl.event} liegt Qualifying-Telemetrie nur fuer "
               f"{len(jahre)} Saison(en) im Cache - fuer einen Vergleich "
               "werden mindestens zwei gebraucht.")
    else:
        c1, c2 = st.columns(2)
        jahr_a = c1.selectbox("Jahr A", jahre, index=0, key="sv_a")
        jahr_b = c2.selectbox("Jahr B", jahre, index=len(jahre) - 1, key="sv_b")

        if jahr_a == jahr_b:
            st.info("Zwei unterschiedliche Jahre waehlen.")
        else:
            @st.cache_data(show_spinner="Pole-Runden laden ...")
            def _profile(cache_pfad: str, event: str, jahr: int):
                s = f1lab.load(jahr, event, "Q", telemetry=True)
                lap = s.laps.pick_fastest()
                tel = lap.get_telemetry().add_distance().add_relative_distance()
                r = tel["RelativeDistance"].to_numpy()
                v = tel["Speed"].to_numpy()
                ok = ~np.isnan(r) & ~np.isnan(v)
                grid = np.linspace(0, 1, 500)
                profil = np.interp(grid, r[ok], v[ok])
                meta = {
                    "jahr": jahr, "pole": str(lap["Driver"]),
                    "zeit_s": round(lap["LapTime"].total_seconds(), 3),
                    "laenge_m": round(float(tel["Distance"].max()), 0),
                    "kurven": len(s.get_circuit_info().corners),
                    "vmax": round(float(v.max()), 1),
                    "vmean": round(float(np.nanmean(v)), 1),
                }
                return grid, profil, meta

            grid, profil_a, meta_a = _profile(str(pfad), auswahl.event, int(jahr_a))
            _, profil_b, meta_b = _profile(str(pfad), auswahl.event, int(jahr_b))

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=grid * 100, y=profil_a, name=str(jahr_a),
                                     line={"color": d.SERIEN[0]}))
            fig.add_trace(go.Scatter(x=grid * 100, y=profil_b, name=str(jahr_b),
                                     line={"color": d.SERIEN[1]}))
            zeige(fig, hoehe=380, xaxis=achse("Rundenanteil [%]"),
                 yaxis=achse("Speed [km/h]"))

            delta = profil_b - profil_a
            peak = int(np.argmax(np.abs(delta)))
            fig2 = go.Figure(go.Scatter(x=grid * 100, y=delta,
                                        line={"color": d.SERIEN[2]}))
            fig2.add_hline(y=0, line_color=d.MUTED, line_width=1)
            fig2.add_annotation(x=grid[peak] * 100, y=delta[peak],
                               text=f"{delta[peak]:+.0f} km/h bei "
                                    f"{grid[peak] * 100:.0f}%",
                               showarrow=True, arrowcolor=d.FG,
                               font={"color": d.FG}, ay=-30)
            zeige(fig2, hoehe=280, showlegend=False,
                 xaxis=achse("Rundenanteil [%]"),
                 yaxis=achse(f"Delta {jahr_b}-{jahr_a} [km/h]"))
            hinweis("Ein grosser, punktueller Ausschlag ist meistens keine "
                   "Geschwindigkeitsaenderung, sondern eine Streckenaenderung "
                   "(entfernte oder neue Schikane) - siehe P33 fuer den "
                   "Fall Spanien 2018/2024, wo genau das passiert.")

            tabelle(pd.DataFrame([meta_a, meta_b]).rename(
                columns={"jahr": "Jahr", "pole": "Pole", "zeit_s": "Zeit [s]",
                        "laenge_m": "Laenge [m]", "kurven": "Kurven",
                        "vmax": "vmax [km/h]", "vmean": "vmean [km/h]"}))
