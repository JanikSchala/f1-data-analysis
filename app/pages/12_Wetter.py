"""wetter-impact: temperaturverlauf, TrackTemp-effekt auf die pace,
nass-/trocken-phasen und ein klassifikator der allein aus feld-aggregaten
erkennt ob das feld auf regenreifen faehrt.

vier reiter aus P17. gerechnet wird nirgends hier. jede kennzahl kommt aus
f1lab.weather_join()/temperature_effect()/weather_phases()/
wet_dry_classifier(). anders als im skript (dort Japan fuer den effekt,
Kanada fuer regen/klassifikator, weil eine einzelne session selten beides
zeigt) laeuft hier alles auf der in der seitenleiste gewaehlten session.
manche reiter zeigen dann eben einen hinweis statt eines ergebnisses.
"""
from __future__ import annotations

import fastf1
import numpy as np
import plotly.graph_objects as go
import streamlit as st
from common import (
    achse,
    hinweis,
    kein_cache_hinweis,
    setup,
    sidebar_session,
    tabelle,
    zeige,
)

import f1lab
from f1lab import design as d

pfad = setup("Wetter", "Temperaturverlauf, TrackTemp-Effekt auf die Pace, "
                       "Nass/Trocken-Phasen und ein Klassifikator dafuer.")
if kein_cache_hinweis(pfad):
    st.stop()
assert pfad is not None  # kein_cache_hinweis() haette sonst schon abgebrochen

auswahl = sidebar_session(pfad)
assert auswahl is not None
ses = f1lab.load(auswahl.season, auswahl.event, auswahl.ident, weather=True)
st.subheader(auswahl.titel)

try:
    w = ses.weather_data
except fastf1.exceptions.DataNotLoadedError:
    # FastF1 scheitert im Offline-Modus nicht beim Laden, sondern erst beim
    # Zugriff, wenn die angeforderte Kategorie fuer diese Session gar nicht
    # im Cache liegt (siehe CLAUDE.md) - w=None faengt nur den Fall ab, dass
    # sie geladen wurde und leer ist, nicht diesen.
    w = None
if w is None or w.empty:
    st.info("Fuer diese Session liegen keine Wetterdaten vor.")
    st.stop()

tab_profil, tab_temp, tab_phasen, tab_klass = st.tabs(
    ["Temperaturverlauf", "TrackTemp-Effekt", "Nass/Trocken-Phasen",
     "Klassifikator"])

# ================================================================== profil
with tab_profil:
    st.markdown("##### Luft- und Streckentemperatur ueber die Session")
    t = w["Time"].dt.total_seconds() / 60
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=w["TrackTemp"], mode="lines",
                             name="Strecke", line={"color": d.SERIEN[1]}))
    fig.add_trace(go.Scatter(x=t, y=w["AirTemp"], mode="lines",
                             name="Luft", line={"color": d.SERIEN[0]}))
    zeige(fig, hoehe=380, xaxis=achse("Sessionzeit [min]"),
         yaxis=achse("Temperatur [\N{DEGREE SIGN}C]"))

    k = st.columns(4)
    k[0].metric("TrackTemp min/max",
               f"{w['TrackTemp'].min():.1f}/{w['TrackTemp'].max():.1f} \N{DEGREE SIGN}C")
    k[1].metric("AirTemp min/max",
               f"{w['AirTemp'].min():.1f}/{w['AirTemp'].max():.1f} \N{DEGREE SIGN}C")
    k[2].metric("Luftfeuchte", f"{w['Humidity'].mean():.0f} %")
    k[3].metric("Regen gemeldet",
               f"{w['Rainfall'].mean() * 100:.0f} % der Session")

# =============================================================== TrackTemp
with tab_temp:
    st.markdown("##### Streckentemperatur gegen Rundenzeit, um Fahrer und "
               "Reifenalter bereinigt")
    merged = f1lab.weather_join(ses)
    erg = f1lab.temperature_effect(merged)

    if erg["n"] == 0:
        st.info("Zu wenige trockene, gewertete Runden mit vollstaendigen "
               "Werten fuer eine belastbare Regression.")
    else:
        d_ = erg["dry"]
        fig = go.Figure(go.Scatter(
            x=d_["TrackTemp"], y=d_["partial"], mode="markers",
            marker={"size": 6, "color": d.MUTED, "opacity": 0.5}))
        xs = np.linspace(d_["TrackTemp"].min(), d_["TrackTemp"].max(), 50)
        fig.add_trace(go.Scatter(
            x=xs, y=erg["coef_temp"] * xs + erg["intercept"], mode="lines",
            line={"color": d.SERIEN[1], "width": 2.5}))
        zeige(fig, hoehe=400, showlegend=False,
             xaxis=achse("Streckentemperatur [\N{DEGREE SIGN}C]"),
             yaxis=achse("Rundenzeit ggue. Fahrer-Median, um TyreLife "
                        "bereinigt [s]"))

        k = st.columns(3)
        k[0].metric("Naive gepoolte Regression",
                   f"{erg['naiv_slope']:+.3f} s/\N{DEGREE SIGN}C",
                   f"R\N{SUPERSCRIPT TWO} {erg['naiv_r2']:.3f}")
        k[1].metric("Kontrolliert: nur TyreLife",
                   f"R\N{SUPERSCRIPT TWO} {erg['r2_tyre_only']:.3f}")
        k[2].metric("Kontrolliert: + TrackTemp",
                   f"{erg['coef_temp']:+.3f} s/\N{DEGREE SIGN}C",
                   f"R\N{SUPERSCRIPT TWO} {erg['r2_voll']:.3f}")
        t_wert = erg["coef_temp"] / erg["se_temp"]
        hinweis(f"n={erg['n']} Runden, t={t_wert:.1f}. Die naive, ueber alle "
                "Fahrer gepoolte Regression findet auf den meisten trockenen "
                "Sessions praktisch kein Signal (R\N{SUPERSCRIPT TWO}<0.06) - "
                "Reifenalter und Fahrer-Tempounterschiede dominieren die "
                "Streuung. Erst mit Fahrer-Median-Bereinigung und TyreLife "
                "als zweiter Variable wird der TrackTemp-Effekt sichtbar "
                "(siehe P17).")

# ================================================================= phasen
with tab_phasen:
    st.markdown("##### Nass/Trocken-Phasen ueber das Rainfall-Flag")
    phasen = f1lab.weather_phases(ses)

    if phasen["nass"].nunique() < 2:
        st.info("Diese Session war durchgehend "
               f"{'nass' if phasen['nass'].iloc[0] else 'trocken'} - keine "
               "Phasenwechsel.")
    else:
        fig = go.Figure()
        for p in phasen.itertuples():
            farbe = d.SERIEN[1] if p.nass else d.SERIEN[0]
            fig.add_vrect(x0=p.start.total_seconds() / 60,
                         x1=p.end.total_seconds() / 60,
                         fillcolor=farbe, opacity=0.6, line_width=0)
        for lbl, farbe in (("trocken", d.SERIEN[0]),
                          ("Regen gemeldet", d.SERIEN[1])):
            fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers",
                                     marker={"color": farbe, "size": 12},
                                     name=lbl))
        zeige(fig, hoehe=200, xaxis=achse("Sessionzeit [min]"),
             yaxis={"visible": False})
        hinweis(f"{len(phasen)} Wetter-Phasen. Das Rainfall-Flag ist nicht "
                "dasselbe wie \"Strecke slick-tauglich\" - viele Runden ohne "
                "Regenmeldung laufen trotzdem noch auf Intermediates, weil "
                "die Strecke noch zu nass zum Wechseln ist (siehe P17).")

        tabelle(phasen.assign(
            start_min=lambda x: (x["start"].dt.total_seconds() / 60).round(1),
            end_min=lambda x: (x["end"].dt.total_seconds() / 60).round(1),
        )[["nass", "start_min", "end_min"]].rename(columns={
            "nass": "Regen gemeldet", "start_min": "Start [min]",
            "end_min": "Ende [min]"}))

# ============================================================= klassifikator
with tab_klass:
    st.markdown("##### Nass/Trocken allein aus Rundenzeit-Streuung und "
               "Speed-Trap erkennen")
    laps_ok = ses.laps.pick_accurate()
    mehrheit = laps_ok.groupby("LapNumber")["Compound"].agg(
        lambda s: s.value_counts().idxmax())
    hat_beide = mehrheit.isin(["INTERMEDIATE", "WET"]).any() and \
        (~mehrheit.isin(["INTERMEDIATE", "WET"])).any()

    if not hat_beide:
        st.info("Diese Session hat keinen Mischungswechsel zwischen Slicks "
               "und Regenreifen - der Klassifikator braucht beide Klassen, "
               "um kreuzvalidiert zu werden.")
    else:
        je_runde, y, pred = f1lab.wet_dry_classifier(ses)
        richtig = pred == y
        acc = richtig.mean()
        basislinie = max(y.mean(), 1 - y.mean())

        k = st.columns(3)
        k[0].metric("Leave-one-out-Trefferquote", f"{acc:.1%}",
                   f"Basislinie {basislinie:.1%}")
        k[1].metric("Runden", len(y))
        k[2].metric("Speed-Trap trocken vs. nass",
                   f"{je_runde.loc[y == 0, 'mean_speedFL'].mean():.0f} / "
                   f"{je_runde.loc[y == 1, 'mean_speedFL'].mean():.0f} km/h")

        fig = go.Figure()
        for label, istnass, symbol in ((0, False, "circle"), (1, True, "triangle-up")):
            m = y == label
            fig.add_trace(go.Scatter(
                x=je_runde.loc[m, "mean_speedFL"], y=je_runde.loc[m, "std_sec"],
                mode="markers", name="Regenreifen" if istnass else "Slicks",
                marker={"color": d.SERIEN[1] if istnass else d.SERIEN[0],
                       "symbol": symbol,
                       "size": [8 if r else 14 for r in richtig[m]],
                       "line": {"width": 1.5,
                               "color": ["rgba(0,0,0,0)" if r else d.FG
                                        for r in richtig[m]]}}))
        zeige(fig, hoehe=420, xaxis=achse("Mittlere Speed-Trap-Geschwindigkeit "
                                          "[km/h]"),
             yaxis=achse("Rundenzeit-Streuung im Feld [s]"))
        hinweis("Umrandete Punkte sind Fehlklassifikationen der Leave-one-"
                "out-Kreuzvalidierung. Keine Compound-Spalte als Feature - "
                "nur was auch ohne Boxenfunk beobachtbar waere. Die "
                "Rundenzeit-Streuung traegt in P17 praktisch nichts bei, die "
                "mittlere Speed-Trap-Geschwindigkeit allein erklaert fast "
                "die ganze Trefferquote.")
