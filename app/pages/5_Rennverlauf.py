"""positionsverlauf vor dem hintergrund der flaggenphasen.

safety car und VSC sind der groesste einzelne stoerfaktor jeder pace-analyse,
deshalb stehen sie hier nicht als fussnote, sondern als hintergrund hinter
dem rennverlauf: so ist zu sehen, welche positionswechsel dem feld und
welche der rennleitung zuzuschreiben sind.
"""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from common import (
    achse,
    hinweis,
    kein_cache_hinweis,
    kopfzeile,
    lade,
    nur_rennen,
    setup,
    sidebar_session,
    tabelle,
    zeige,
)

import f1lab
from f1lab import design as d

pfad = setup("Rennverlauf", "Positionen und Flaggenphasen ueber die Distanz.")
if kein_cache_hinweis(pfad):
    st.stop()
assert pfad is not None  # kein_cache_hinweis() haette sonst schon abgebrochen

auswahl = sidebar_session(pfad)
assert auswahl is not None
ses = lade(auswahl)
kopfzeile(ses, auswahl)

if nur_rennen(auswahl, "Der Positionsverlauf"):
    st.stop()

# --- flaggenphasen --------------------------------------------------------
try:
    phasen = f1lab.track_status_phases(ses)
except Exception:
    phasen = pd.DataFrame()
    st.info("Fuer diese Session liegen keine Flaggendaten vor.")

# gruene phasen sind der normalfall und wuerden das bild nur zustellen.
stoerungen = (phasen[phasen["label"] != "gruen"] if not phasen.empty
              else pd.DataFrame())

if not stoerungen.empty:
    neutral = stoerungen[stoerungen["label"].isin(["safety car", "vsc"])]
    k = st.columns(3)
    k[0].metric("Phasen ausser Gruen", len(stoerungen))
    k[1].metric("Neutralisationen", len(neutral))
    k[2].metric("Zeit unter Neutralisation",
                f"{neutral['duration_s'].sum() / 60:.1f} min")

# --- positionsverlauf -----------------------------------------------------
st.markdown("##### Positionen ueber die Renndistanz")

laps = ses.laps[["Driver", "LapNumber", "Position"]].dropna(subset=["Position"])
if laps.empty:
    st.info("Keine Positionsdaten in dieser Session.")
    st.stop()

alle = sorted(laps["Driver"].unique())
zeigen = st.multiselect(
    f"Fahrer hervorheben (hoechstens {d.MAX_SERIEN})", alle,
    default=alle[:d.MAX_SERIEN], max_selections=d.MAX_SERIEN)

fig = go.Figure()

# neutralisationen zuerst, damit sie hinter den linien liegen.
for zeile in stoerungen.itertuples():
    if zeile.lap_end < zeile.lap_start:
        continue
    fig.add_vrect(x0=zeile.lap_start,
                  x1=max(zeile.lap_end, zeile.lap_start + 0.4),
                  fillcolor=d.PHASE.get(zeile.label, d.MUTED), opacity=0.18,
                  line_width=0, layer="below")

# alle anderen fahrer bleiben als graue linien im hintergrund, ohne eine
# eigene farbe zu beanspruchen.
for fahrer, teil in laps.groupby("Driver"):
    if fahrer in zeigen:
        continue
    teil = teil.sort_values("LapNumber")
    fig.add_trace(go.Scatter(
        x=teil["LapNumber"], y=teil["Position"], mode="lines", name=str(fahrer),
        showlegend=False, line={"color": d.GRID, "width": 1},
        hovertemplate=f"<b>{fahrer}</b><br>Runde %{{x}}<br>"
                      f"Position %{{y}}<extra></extra>"))

for i, fahrer in enumerate(zeigen):
    teil = laps[laps["Driver"] == fahrer].sort_values("LapNumber")
    fig.add_trace(go.Scatter(
        x=teil["LapNumber"], y=teil["Position"], mode="lines", name=str(fahrer),
        line={"color": d.SERIEN[i], "width": 2.6},
        hovertemplate=f"<b>{fahrer}</b><br>Runde %{{x}}<br>"
                      f"Position %{{y}}<extra></extra>"))

zeige(fig, hoehe=560, xaxis=achse("Runde"),
      yaxis=achse("Position", autorange="reversed", dtick=2))

if not stoerungen.empty:
    namen = ", ".join(sorted(stoerungen["label"].unique()))
    hinweis(f"Die eingefaerbten Baender sind Phasen ausser Gruen ({namen}). "
            "Positionswechsel, die genau dort passieren, sind meist "
            "Boxenstopps unter Neutralisation - die kosten nur einen Bruchteil "
            "und sind keine Leistung auf der Strecke.")
else:
    hinweis("Durchgehend gruen: alle Positionswechsel sind auf der Strecke "
            "oder in der Box entstanden.")

# --- phasentabelle --------------------------------------------------------
if not stoerungen.empty:
    with st.expander("Flaggenphasen im Detail"):
        t = stoerungen[["label", "lap_start", "lap_end", "duration_s"]].copy()
        t["duration_s"] = t["duration_s"].round(1)
        tabelle(t.rename(columns={
            "label": "Phase", "lap_start": "ab Runde", "lap_end": "bis Runde",
            "duration_s": "Dauer [s]"}))
        st.caption(
            "Die Rundennummern stammen vom Fuehrenden - wer eine Runde "
            "zurueckliegt, erlebt dieselbe Phase eine Runde spaeter. Kurze "
            "Gelbphasen von wenigen Sekunden sind lokale Warnungen an einer "
            "einzelnen Streckenposition, keine Neutralisation.")
