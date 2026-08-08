"""Race Pace mit Konfidenzintervallen.

Die Aussage steht nicht im Ranking, sondern in den Intervallen: ueberlappen
sich zwei, ist der Unterschied mit diesen Daten nicht belegbar. Genau deshalb
traegt hier jeder Balken seine Fehlergrenzen mit.
"""
from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st
from common import (
    achse,
    hinweis,
    kein_cache_hinweis,
    kopfzeile,
    lade,
    namensachse,
    setup,
    sidebar_session,
    tabelle,
    zeige,
)

import f1lab
from f1lab import design as d

pfad = setup("Pace", "Bereinigte Rundenzeiten je Fahrer, mit Bootstrap-"
                     "Intervall.")
if kein_cache_hinweis(pfad):
    st.stop()

auswahl = sidebar_session(pfad)
ses = lade(auswahl)
kopfzeile(ses, auswahl)

with st.sidebar:
    st.header("Filter")
    schwelle = st.slider(
        "Ausreisser-Schwelle", 1.02, 1.20, 1.07, 0.01,
        help="Runden langsamer als dieser Faktor mal Bestzeit fliegen raus. "
             "Eng gefasst bleibt nur die Bestform uebrig, weit gefasst kommen "
             "Verkehr und Reifenabbau wieder mit hinein.")
    min_runden = st.slider(
        "Mindestrunden je Fahrer", 3, 25, 8, 1,
        help="Unter dieser Rundenzahl wird kein Median gebildet. Wenige "
             "Runden ergeben ein Intervall, das jede Aussage verschluckt.")

tab = f1lab.pace_table(ses, threshold=schwelle, min_laps=min_runden)
if tab.empty:
    st.warning("Keine Runden ueberstehen diese Filter. Schwelle anheben oder "
               "Mindestrunden senken.")
    st.stop()

# --- Wo ist der Unterschied belegbar? -------------------------------------
# Das Intervall des Schnellsten ist der Massstab: wessen Intervall sich damit
# ueberschneidet, ist von ihm nicht unterscheidbar.
bester = tab.iloc[0]
abgesetzt = tab[tab["ci_lo"] > bester["ci_hi"]]

k = st.columns(3)
k[0].metric("Schnellster", bester["driver"], f"{bester['median_s']:.3f} s Median")
k[1].metric("Nicht unterscheidbar", f"{len(tab) - len(abgesetzt)} Fahrer",
            help="Deren Intervall ueberschneidet sich mit dem des "
                 "Schnellsten - der Rueckstand ist mit diesen Runden nicht "
                 "belegbar.")
k[2].metric("Ausgewertete Runden", int(tab["laps"].sum()))

# --- Diagramm -------------------------------------------------------------
# Sortiertes Balkendiagramm mit einer Farbe: die Identitaet haengt an der
# Achse, nicht an der Farbe - damit traegt es beliebig viele Fahrer, ohne die
# Drei-Serien-Grenze der Kategoriepalette zu verletzen. Die zweite Farbe ist
# keine Kategorie, sondern eine Wertung: abgesetzt oder nicht.
p = tab.iloc[::-1]
farben = [d.POSITIV if lo > bester["ci_hi"] else d.SERIEN[0]
          for lo in p["ci_lo"]]
farben[-1] = d.FG                       # der Schnellste selbst

fig = go.Figure(go.Bar(
    x=p["delta_s"], y=p["driver"], orientation="h",
    marker={"color": farben, "line": {"color": d.BG, "width": 2}},
    error_x={"type": "data", "symmetric": False,
             "array": p["ci_hi"] - p["delta_s"],
             "arrayminus": p["delta_s"] - p["ci_lo"],
             "color": d.MUTED, "thickness": 1.4, "width": 5},
    customdata=p[["team", "laps", "ci_width"]],
    hovertemplate="<b>%{y}</b><br>%{customdata[0]}<br>"
                  "Rueckstand %{x:.3f} s<br>%{customdata[1]} Runden, "
                  "Intervallbreite %{customdata[2]:.3f} s<extra></extra>"))

zeige(fig, hoehe=max(320, 26 * len(p)), showlegend=False,
      xaxis=achse("Rueckstand auf den Schnellsten [s]"), yaxis=namensachse())

hinweis("Die Balkenenden sind Punktschaetzer, die Querstriche das "
        "95-Prozent-Intervall des Medians. Gruen ist, wer sich vom Schnellsten "
        "nachweisbar abgesetzt hat - bei den blauen ueberlappen die "
        "Intervalle, dort gibt der Datensatz keine Rangfolge her.")

with st.expander("Tabelle"):
    tabelle(tab.rename(columns={
        "driver": "Fahrer", "team": "Team", "laps": "Runden",
        "median_s": "Median [s]", "delta_s": "Rueckstand [s]",
        "ci_lo": "Intervall von", "ci_hi": "Intervall bis",
        "ci_width": "Breite [s]"}))

with st.expander("Was hier herausgefiltert wird"):
    roh = len(ses.laps)
    sauber = len(f1lab.clean_laps(ses, threshold=schwelle))
    st.markdown(
        f"Von **{roh}** Runden im Feed bleiben **{sauber}** uebrig "
        f"({100 * sauber / roh:.0f} Prozent).\n\n"
        "`clean_laps()` entfernt der Reihe nach Boxenrunden, von FastF1 als "
        "unplausibel markierte Runden, alles ausser gruener Flagge, von der "
        "Rennleitung gestrichene Runden und zuletzt die Ausreisser oberhalb "
        "der Schwelle.\n\n"
        "Die Safety-Car-Runden sind dabei der grosse Posten: sie verschieben "
        "einen ungefilterten Median um mehrere Zehntel und treffen die Fahrer "
        "ungleich, je nachdem wo sie zum Zeitpunkt der Neutralisation im Feld "
        "standen.")

if auswahl.ident not in ("R", "S"):
    st.info("Ausserhalb von Rennen und Sprint ist Race Pace nur bedingt "
            "aussagekraeftig: im Training bestimmen Spritmenge und Programm "
            "die Rundenzeit staerker als das Auto.")
