"""uebersicht einer session: ergebnis und rundenzeiten im verlauf."""
from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st
from common import (
    achse,
    hinweis,
    kein_cache_hinweis,
    kopfzeile,
    lade,
    setup,
    sidebar_session,
    tabelle,
    zeige,
)

import f1lab
from f1lab import design as d

pfad = setup("Uebersicht", "Wie lief die Session, und wie entwickelten sich "
                           "die Rundenzeiten?")
if kein_cache_hinweis(pfad):
    st.stop()
assert pfad is not None  # kein_cache_hinweis() haette sonst schon abgebrochen

auswahl = sidebar_session(pfad)
assert auswahl is not None
ses = lade(auswahl)
kopfzeile(ses, auswahl)

laps = ses.laps

# --- ergebnis -------------------------------------------------------------
st.markdown("##### Ergebnis")
try:
    r = ses.results
    spalten = [c for c in ("Position", "Abbreviation", "TeamName",
                           "GridPosition", "Points", "Status")
               if c in r.columns]
    tabelle(r[spalten].rename(columns={
        "Position": "Platz", "Abbreviation": "Fahrer", "TeamName": "Team",
        "GridPosition": "Start", "Points": "Punkte"}), height=430)
except Exception:
    st.info("Fuer diese Session liegt keine Ergebnistabelle vor.")

# --- rundenzeiten ---------------------------------------------------------
st.markdown("##### Rundenzeiten im Verlauf")

fahrer = sorted(laps["Driver"].dropna().unique())
wahl = st.multiselect(
    f"Fahrer vergleichen (hoechstens {d.MAX_SERIEN})", fahrer,
    default=fahrer[:2], max_selections=d.MAX_SERIEN)
hinweis("Auf drei Fahrer begrenzt: mehr Linien lassen sich farblich nicht "
        "mehr zuverlaessig unterscheiden - auch nicht bei Farbsehschwaeche. "
        "Das Tempo aller Fahrer steht auf der Seite <b>Pace</b>.")

if not wahl:
    st.info("Waehle oben mindestens einen Fahrer.")
    st.stop()

nur_saubere = st.checkbox(
    "Nur bereinigte Runden", value=False,
    help="Blendet Boxenrunden, Safety-Car-Phasen und Ausreisser aus - "
         "dieselbe Filterung, auf der die Pace-Auswertung aufbaut.")

grundlage = f1lab.clean_laps(ses) if nur_saubere else laps[laps["IsAccurate"]]

fig = go.Figure()
for i, drv in enumerate(wahl):
    g = grundlage[grundlage["Driver"] == drv].sort_values("LapNumber")
    fig.add_trace(go.Scatter(
        x=g["LapNumber"], y=g["LapTime"].dt.total_seconds(),
        mode="lines+markers", name=str(drv),
        line={"color": d.SERIEN[i], "width": 2}, marker={"size": 5},
        hovertemplate=f"<b>{drv}</b><br>Runde %{{x}}<br>"
                      "%{y:.3f} s<extra></extra>"))

zeige(fig, hoehe=520, xaxis=achse("Runde"),
      yaxis=achse("Rundenzeit [s]"))

if nur_saubere:
    hinweis(f"Von {len(laps)} Runden im Feed bleiben {len(grundlage)} uebrig. "
            "Entfernt sind Boxenrunden, unplausible Runden, alles ausser "
            "gruener Flagge, gestrichene Runden und Ausreisser.")
else:
    hinweis("Nur als plausibel markierte Runden. Ausreisser nach oben sind "
            "meist Boxenstopps oder Gelbphasen - die Schaltflaeche darueber "
            "blendet sie aus.")
