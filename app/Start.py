"""Startseite: was liegt auf dem Rechner, und was macht die App damit.

Start mit::

    streamlit run app/Start.py

Oder per Doppelklick auf "2_Dashboard_starten.command".
"""
from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st
from common import (
    IDENT_ORDER,
    achse,
    hinweis,
    inventar,
    kein_cache_hinweis,
    namensachse,
    setup,
    tabelle,
    zeige,
)

from f1lab import design as d

pfad = setup(
    "\N{CHEQUERED FLAG} Formel-1-Analyse",
    "Renndaten auswerten statt Code lesen. Gerechnet wird ausschliesslich in "
    "f1lab - derselbe Code, den auch die Skripte und die Tests verwenden.")

if kein_cache_hinweis(pfad):
    st.stop()

inv = inventar(str(pfad))
geladen = inv[inv["timing"]]

# --- Eckdaten -------------------------------------------------------------
k = st.columns(4)
k[0].metric("Sessions gespeichert", f"{len(geladen):,}".replace(",", "."))
k[1].metric("davon mit Telemetrie", int(geladen["telemetry"].sum()))
k[2].metric("Saisons", geladen["season"].nunique())
k[3].metric("Rennwochenenden", geladen.groupby(["season", "event"]).ngroups)

# --- Abdeckung ------------------------------------------------------------
st.markdown("##### Was gespeichert ist")

# Die Bestandsaufnahme kennt keine Rundennummern - die Reihenfolge im Kalender
# ergibt sie aber eindeutig, weil Testfahrten nicht im Cache landen.
wochenenden = (geladen.groupby(["season", "event", "event_date"])
               .agg(sessions=("ident", "nunique"))
               .reset_index())
wochenenden["runde"] = (wochenenden.sort_values("event_date")
                        .groupby("season").cumcount() + 1)
raster = wochenenden.pivot(index="season", columns="runde", values="sessions")
raster = raster.reindex(sorted(raster.index))

fig = go.Figure(go.Heatmap(
    z=raster.to_numpy(), x=raster.columns.tolist(),
    y=[str(s) for s in raster.index],
    colorscale=[[0, d.BG_HELL]] + [[(i + 1) / len(d.RAMPE), f]
                                   for i, f in enumerate(d.RAMPE)],
    hovertemplate="%{y} Runde %{x}<br>%{z} Sessions<extra></extra>",
    colorbar={"title": {"text": "Sessions", "side": "right"},
              "tickfont": {"color": d.MUTED}, "outlinewidth": 0},
    xgap=2, ygap=2))
zeige(fig, hoehe=110 + 34 * len(raster),
      xaxis=achse("Runde", dtick=2),
      yaxis=namensachse(autorange="reversed"))
hinweis("Leere Felder sind Wochenenden, die noch nie geladen wurden - keine "
        "Ausfaelle, sondern Luecken im Warmup. Sieben Sessions sind das "
        "Maximum, konventionelle Wochenenden haben fuenf.")

# --- Verteilung -----------------------------------------------------------
links, rechts = st.columns(2)

with links:
    je_typ = (geladen.groupby("ident")
              .agg(sessions=("ident", "size"), telemetrie=("telemetry", "sum"))
              .reindex([i for i in IDENT_ORDER if i in set(geladen["ident"])]))
    fig = go.Figure([
        go.Bar(name="nur Rundendaten",
               y=[d.IDENT_NAME.get(i, i) for i in je_typ.index],
               x=je_typ["sessions"] - je_typ["telemetrie"],
               orientation="h", marker={"color": d.SERIEN[0]}),
        go.Bar(name="mit Telemetrie",
               y=[d.IDENT_NAME.get(i, i) for i in je_typ.index],
               x=je_typ["telemetrie"], orientation="h",
               marker={"color": d.SERIEN[1]}),
    ])
    st.markdown("##### Nach Session-Typ")
    zeige(fig, hoehe=340, barmode="stack",
          xaxis=achse("Sessions"), yaxis=namensachse())

with rechts:
    je_saison = (geladen.groupby("season")
                 .agg(sessions=("ident", "size"),
                      telemetrie=("telemetry", "sum")).reset_index())
    fig = go.Figure([
        go.Bar(name="nur Rundendaten", x=je_saison["season"],
               y=je_saison["sessions"] - je_saison["telemetrie"],
               marker={"color": d.SERIEN[0]}),
        go.Bar(name="mit Telemetrie", x=je_saison["season"],
               y=je_saison["telemetrie"], marker={"color": d.SERIEN[1]}),
    ])
    st.markdown("##### Nach Saison")
    zeige(fig, hoehe=340, barmode="stack",
          xaxis=achse(type="category"), yaxis=achse("Sessions"))

hinweis("Telemetrie ist ein eigener, um ein Vielfaches groesserer Download. "
        "Ohne sie laufen Pace, Reifen und Strategie trotzdem - nur "
        "Streckenverlauf und Bremszonen brauchen sie zwingend.")

# --- Wegweiser ------------------------------------------------------------
st.markdown("##### Die Seiten")
st.markdown("""
| Seite | Was sie beantwortet | Quelle in f1lab |
| --- | --- | --- |
| **Uebersicht** | Wie lief die Session, und wie entwickelten sich die Rundenzeiten? | `clean_laps` |
| **Pace** | Wer war wirklich schnell, und ab wann ist der Unterschied belegbar? | `pace_table`, `bootstrap_median` |
| **Reifen** | Wie stark baut welche Mischung ab, und welchem Fit ist zu trauen? | `stints`, `degradation` |
| **Strategie** | Was kostet ein Stopp hier, und wann lohnt der Undercut? | `pit_loss`, `undercut_gain` |
| **Rennverlauf** | Wo lagen Safety Car und VSC, und was hat das verschoben? | `track_status_phases` |
| **Telemetrie** | Wo wird gebremst, wie hart, und wie sieht die Strecke aus? | `braking_zones`, `circuit_geometry` |
| **Kalender** | Wie sieht der Kalender aus, und wie vermisst man eine Strecke? | `event_dimension`, `circuit_dimension` |
""")

with st.expander("Warum nur gespeicherte Daten?"):
    st.markdown(
        "FastF1 begrenzt sich auf 500 Abfragen pro Stunde. Wuerde die App "
        "fehlende Sessions selbst nachladen, waere die Bedienung von "
        "minutenlangen Wartezeiten und Zwangspausen durchsetzt - mitten im "
        "Klicken.\n\n"
        "Das Laden ist deshalb ein eigener Vorgang: der Warmup aus **p01** "
        "zieht ganze Saisons, faengt das Limit ab und laesst sich jederzeit "
        "abbrechen und fortsetzen. Die App liest nur, was dort angekommen "
        "ist. Sie laeuft damit auch ohne Netz und bietet nur Sessions an, die "
        "tatsaechlich auswertbar sind.")

with st.expander("Bestand im Detail"):
    t = geladen.copy()
    t["Session"] = [d.IDENT_NAME.get(i, i) for i in t["ident"]]
    t["Telemetrie"] = t["telemetry"].map({True: "ja", False: "\N{EN DASH}"})
    tabelle(
        t[["season", "event_date", "event", "Session", "Telemetrie"]]
        .rename(columns={"season": "Saison", "event_date": "Datum",
                         "event": "Rennen"}),
        column_config={"Datum": st.column_config.DateColumn(
            format="DD.MM.YYYY")})
