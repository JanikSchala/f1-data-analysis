"""Streckenverlauf und Bremszonen der schnellsten Runde.

Beides kommt aus derselben Quelle: den Positions- und Fahrzeugkanaelen einer
einzelnen Runde. Der Bremskanal ist binaer - ueber seine Flanken lassen sich
zusammenhaengende Bremsphasen finden, und je Zone Eintrittsgeschwindigkeit,
Laenge und mittlere Verzoegerung berechnen.
"""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from common import (
    achse,
    hinweis,
    kein_cache_hinweis,
    lade,
    setup,
    sidebar_session,
    tabelle,
    zeige,
)

import f1lab
from f1lab import design as d

pfad = setup("Telemetrie", "Wie sieht die Strecke aus, und wo wird gebremst?")
if kein_cache_hinweis(pfad):
    st.stop()

auswahl = sidebar_session(pfad, nur_mit_telemetrie=True)
if auswahl is None:
    st.stop()

ses = lade(auswahl, telemetrie=True)
st.subheader(auswahl.titel)

runden = ses.laps.dropna(subset=["LapTime"])
if runden.empty:
    st.info("Keine gewerteten Runden in dieser Session.")
    st.stop()

# --- Streckengeometrie ----------------------------------------------------
st.markdown("##### Die Strecke")

geo = f1lab.circuit_geometry(ses)
g = st.columns(4)


def _zahl(wert, muster):
    """NA-sichere Anzeige: pd.NA und NaN werden zum Gedankenstrich."""
    return muster.format(wert) if pd.notna(wert) else "\N{EN DASH}"


g[0].metric("Kurven", _zahl(geo["corners"], "{:.0f}"))
g[1].metric("Laenge", _zahl(geo["length_m"], "{:.0f} m"))
g[2].metric("Anstieg gesamt", _zahl(geo["elev_gain_m"], "{:.0f} m"))
g[3].metric("Hoehenunterschied", _zahl(geo["elev_span_m"], "{:.0f} m"))

hinweis("Laenge und Hoehe stammen aus den Positionsdaten der schnellsten "
        "Runde. Gemessen wird die gefahrene Ideallinie - die schneidet Kurven "
        "und faellt dadurch rund ein Prozent kuerzer aus als die offizielle "
        "Streckenlaenge. Eine Abweichung in dieser Groessenordnung ist also "
        "kein Fehler, sondern die Bestaetigung, dass richtig gemessen wird.")

if pd.isna(geo["corners"]):
    st.caption(
        "Die Kurvenzahl fehlt: sie ist der einzige Wert hier, der nicht aus "
        "der Telemetrie kommt, sondern aus einer eigenen Abfrage - und die "
        "geht im Offline-Betrieb nicht. Laenge und Hoehenmeter sind davon "
        "nicht betroffen.")

schnellste = runden.pick_fastest()
try:
    pos = schnellste.get_pos_data()
except Exception:
    pos = None

if pos is not None and not pos.empty:
    fig = go.Figure(go.Scatter(
        x=pos["X"] / 10, y=pos["Y"] / 10, mode="lines",
        line={"color": d.SERIEN[0], "width": 3}, hoverinfo="skip"))
    zeige(fig, hoehe=520, showlegend=False,
          margin={"l": 10, "r": 10, "t": 10, "b": 10},
          xaxis={"visible": False},
          yaxis={"visible": False, "scaleanchor": "x", "scaleratio": 1})
    hinweis(f"Ideallinie der schnellsten Runde ({schnellste['Driver']}).")

# --- Bremszonen -----------------------------------------------------------
st.markdown("##### Bremszonen")

fahrer = sorted(runden["Driver"].unique())
schnellster = str(runden.loc[runden["LapTime"].idxmin(), "Driver"])

with st.sidebar:
    st.header("Bremszonen")
    wahl = st.selectbox("Fahrer", fahrer, index=fahrer.index(schnellster),
                        help="Ausgewertet wird die schnellste Runde dieses "
                             "Fahrers.")
    min_laenge = st.slider(
        "Mindestlaenge einer Zone [m]", 5, 80, 20, 5,
        help="Kuerzere Bremsungen sind meist Messrauschen oder ein kurzes "
             "Antippen zur Balance, keine echte Bremszone.")

lap = runden[runden["Driver"] == wahl].pick_fastest()
if lap is None or pd.isna(lap["LapTime"]):
    st.info(f"Keine schnellste Runde fuer {wahl} bestimmbar.")
    st.stop()

try:
    car = lap.get_car_data().add_distance()
except Exception as exc:
    st.info("Die Telemetrie dieser Runde laesst sich nicht aufbereiten. "
            f"({type(exc).__name__})")
    st.stop()

zonen = pd.DataFrame(f1lab.braking_zones(
    car["Brake"], car["Distance"], car["Speed"],
    car["Time"].dt.total_seconds(), min_length_m=min_laenge))

if zonen.empty:
    st.info("Keine Zone ueberschreitet die Mindestlaenge. Schwelle senken.")
    st.stop()

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
    x=car["Distance"], y=car["Speed"], mode="lines", name="Geschwindigkeit",
    line={"color": d.SERIEN[0], "width": 2},
    hovertemplate="%{x:.0f} m<br>%{y:.0f} km/h<extra></extra>"))

zeige(fig, hoehe=420, showlegend=False,
      xaxis=achse("Distanz [m]"), yaxis=achse("km/h"))
hinweis("Die markierten Baender sind die erkannten Bremszonen. Ihre Grenzen "
        "kommen aus den Flanken des binaeren Bremssignals, nicht aus der "
        "Geschwindigkeit - deshalb endet eine Zone dort, wo der Fahrer vom "
        "Bremspedal geht, und nicht am Scheitelpunkt.")

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
    hinweis("Mittelwerte ueber die ganze Zone, nicht Spitzenwerte - die liegen "
            "deutlich hoeher.")

with rechts:
    st.markdown("##### Zonen")
    tabelle(zonen.rename(columns={
        "start_m": "ab [m]", "end_m": "bis [m]", "length_m": "Laenge [m]",
        "v_entry_kmh": "Eintritt [km/h]", "v_min_kmh": "Minimum [km/h]",
        "duration_s": "Dauer [s]", "decel_g": "Verzoegerung [g]"}), height=380)

with st.expander("Wie die Zonen gefunden werden"):
    st.markdown(
        "Der Kanal `Brake` ist ein Schalter, kein Druck: er sagt nur, ob "
        "gebremst wird. Ueber die Differenz aufeinanderfolgender Werte lassen "
        "sich die Flanken finden - eine steigende Flanke oeffnet eine Zone, "
        "eine fallende schliesst sie.\n\n"
        "Zwei Randfaelle muss die Funktion abfangen: eine Runde, die bereits "
        "bremsend beginnt, und eine, die bremsend endet. In beiden Faellen "
        "fehlt die zugehoerige Flanke, und die Zone ginge sonst verloren.\n\n"
        "Die mittlere Verzoegerung ist der Geschwindigkeitsabbau geteilt durch "
        "die Dauer der Zone, umgerechnet in Vielfache von g.")
