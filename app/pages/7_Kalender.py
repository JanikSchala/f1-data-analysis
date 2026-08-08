"""Kalender und Streckengeometrie - die Dimensionstabellen aus p02.

Die Event-Dimension kommt ohne jeden Session-Download aus. Die Circuit-
Dimension braucht dagegen Telemetrie, weil `get_circuit_info()` zwar die
Kurven liefert, aber kein Z - Laenge und Hoehe stammen aus dem Positionskanal
der schnellsten Runde.
"""
from __future__ import annotations

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

import f1lab
from f1lab import design as d

pfad = setup("Kalender", "Rennwochenenden und vermessene Strecken.")
if kein_cache_hinweis(pfad):
    st.stop()

inv = inventar(str(pfad))
saisons = sorted(inv["season"].unique())


@st.cache_data(show_spinner="Kalender wird aufgebaut ...")
def events(jahre: tuple[int, ...]) -> pd.DataFrame:
    """Kalender aus dem gespeicherten Bestand. Kein Session-Download noetig."""
    return f1lab.event_dimension(jahre)


@st.cache_data(show_spinner="Strecken werden vermessen ...")
def geometrie(paare: tuple[tuple[int, str], ...]) -> pd.DataFrame:
    """Streckengeometrie. Braucht Telemetrie und laeuft daher nur auf
    Anforderung - eine Session je Strecke genuegt, Geometrie ist pro Layout
    konstant."""
    return f1lab.circuit_dimension(list(paare))


# --- Kalender -------------------------------------------------------------
st.markdown("##### Rennkalender")

try:
    dim = events(tuple(saisons))
except Exception as exc:
    st.info(f"Der Kalender laesst sich nicht aufbauen. ({type(exc).__name__})")
    st.stop()

if dim.empty:
    st.info("Fuer keine gespeicherte Saison liegt ein Kalender vor.")
    st.stop()

k = st.columns(4)
k[0].metric("Wochenenden", len(dim))
k[1].metric("Sprint-Wochenenden", int(dim["is_sprint"].sum()))
k[2].metric("Saisons", dim["season"].nunique())
k[3].metric("Strecken", dim["location"].nunique())

fig = go.Figure()
for ist_sprint, teil in dim.groupby("is_sprint"):
    fig.add_trace(go.Scatter(
        x=teil["round"], y=teil["season"].astype(str), mode="markers",
        name="Sprint" if ist_sprint else "konventionell",
        marker={"size": 15, "symbol": "square",
                "color": d.SPRINT if ist_sprint else d.KONVENTIONELL,
                "line": {"color": d.BG, "width": 2}},
        customdata=teil[["event_name", "country", "event_format"]],
        hovertemplate="<b>%{customdata[0]}</b><br>%{customdata[1]}<br>"
                      "Runde %{x}, %{customdata[2]}<extra></extra>"))

zeige(fig, hoehe=110 + 34 * dim["season"].nunique(),
      xaxis=achse("Runde", dtick=2),
      yaxis=namensachse(autorange="reversed"))
hinweis("Das Sprint-Format heisst je nach Jahr sprint, sprint_shootout oder "
        "sprint_qualifying - erkannt wird es deshalb ueber das Teilwort und "
        "nicht ueber Gleichheit.")

with st.expander("Event-Dimension als Tabelle"):
    tabelle(dim)
    st.download_button("Als CSV speichern", dim.to_csv(index=False),
                       file_name="dim_events.csv", mime="text/csv")

# --- Strecken -------------------------------------------------------------
st.markdown("##### Streckengeometrie")

mit_telemetrie = inv[inv["telemetry"]]
if mit_telemetrie.empty:
    st.info("**Streckenvermessung braucht Telemetrie.**\n\n"
            "Fuer keine gespeicherte Session liegt sie vor. Im Warmup aus p01 "
            "laesst sie sich mit `telemetry=True` anfordern.")
    st.stop()

saison = st.selectbox("Saison", sorted(mit_telemetrie["season"].unique(),
                                       reverse=True))
verfuegbar = (mit_telemetrie[mit_telemetrie["season"] == saison]
              .sort_values("event_date")["event"].unique().tolist())

wahl = st.multiselect(
    "Strecken", verfuegbar, default=verfuegbar[:6],
    help="Je Strecke wird eine Session mit Telemetrie geoeffnet. Das dauert "
         "beim ersten Mal einige Sekunden pro Strecke.")

if not wahl:
    st.caption("Mindestens eine Strecke waehlen.")
    st.stop()

if not st.button(f"{len(wahl)} Strecken vermessen", type="primary"):
    st.stop()

geo = geometrie(tuple((int(saison), str(e)) for e in wahl))
fertig = (geo.dropna(subset=["length_m"]).copy()
          if "length_m" in geo.columns else pd.DataFrame())

if fertig.empty:
    st.info("Keine der gewaehlten Strecken liess sich vermessen - vermutlich "
            "fehlt die Telemetrie der Qualifying-Session.")
    st.stop()

fertig["km"] = fertig["length_m"] / 1000

# Die Kurvenzahl ist der einzige Wert, der nicht aus der Telemetrie stammt,
# sondern aus einer eigenen Abfrage - im Offline-Betrieb fehlt sie deshalb.
# Dann traegt die Hoehenspanne die zweite Achse, statt ein leeres Bild zu
# zeigen.
hat_kurven = fertig["corners"].notna().any()
y_spalte = "corners" if hat_kurven else "elev_span_m"
y_titel = "Kurven" if hat_kurven else "Hoehenspanne [m]"

fig = go.Figure(go.Scatter(
    x=fertig["km"], y=fertig[y_spalte], mode="markers+text",
    text=fertig["circuit"], textposition="top center",
    textfont={"color": d.MUTED, "size": 11},
    marker={"size": 18, "color": fertig["elev_span_m"],
            "colorscale": d.RAMPE_SCALE, "line": {"width": 2, "color": d.BG},
            "colorbar": {"title": {"text": "Hoehenspanne [m]", "side": "right"},
                         "tickfont": {"color": d.MUTED}, "outlinewidth": 0}},
    customdata=fertig[["length_m", "elev_gain_m", "elev_span_m"]],
    hovertemplate="<b>%{text}</b><br>%{customdata[0]:.0f} m<br>"
                  "Anstieg %{customdata[1]:.0f} m, Spanne "
                  "%{customdata[2]:.0f} m<extra></extra>"))

zeige(fig, hoehe=500, showlegend=False,
      xaxis=achse("Streckenlaenge [km], gefahrene Linie"),
      yaxis=achse(y_titel))
hinweis("Gemessen wird die gefahrene Linie, nicht die offizielle "
        "Streckenlaenge. Die Ideallinie schneidet Kurven und faellt dadurch "
        "typisch ein bis zwei Prozent kuerzer aus.")

if not hat_kurven:
    st.caption(
        "Die Kurvenzahl fehlt fuer alle gewaehlten Strecken - sie kommt aus "
        "einer eigenen Abfrage, die im Offline-Betrieb nicht geht. Deshalb "
        "steht hier die Hoehenspanne auf der zweiten Achse.")

with st.expander("Circuit-Dimension als Tabelle"):
    tabelle(geo)
    if "error" in geo.columns and geo["error"].notna().any():
        st.caption("Zeilen mit Eintrag in `error` haben keine Telemetrie im "
                   "Bestand - sie bleiben absichtlich stehen, damit sichtbar "
                   "ist, was fehlt.")
