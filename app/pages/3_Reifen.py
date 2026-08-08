"""Stints und Reifendegradation.

Die Steigung je Stint ist mit herausgerechnetem Treibstoffeffekt geschaetzt.
Ohne diese Korrektur wird die Degradation systematisch unterschaetzt: das Auto
wird leichter, waehrend der Reifen abbaut, und beide Effekte heben sich
teilweise auf.
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

pfad = setup("Reifen", "Wer faehrt welche Mischung wie lange - und was kostet "
                       "sie pro Runde?")
if kein_cache_hinweis(pfad):
    st.stop()

auswahl = sidebar_session(pfad)
ses = lade(auswahl)
kopfzeile(ses, auswahl)

with st.sidebar:
    st.header("Filter")
    schwelle = st.slider(
        "Ausreisser-Schwelle", 1.02, 1.25, 1.10, 0.01,
        help="Fuer Degradation bewusst weiter gefasst als bei der Pace: der "
             "Abbau am Stint-Ende soll drinbleiben, sonst schaetzt der Fit "
             "ihn weg.")
    min_runden = st.slider("Mindestrunden je Stint", 4, 20, 6, 1)

# --- Stints ---------------------------------------------------------------
st.markdown("##### Reifenwechsel je Fahrer")

stints = f1lab.stints(ses)
if stints.empty:
    st.info("Fuer diese Session liegen keine Stint-Daten vor.")
    st.stop()

fig = go.Figure()
gezeigt = set()
for row in stints.itertuples():
    mischung = str(row.Compound).upper()
    fig.add_trace(go.Bar(
        y=[row.Driver], x=[row.laps], base=[row.start - 1], orientation="h",
        name=mischung, legendgroup=mischung,
        showlegend=mischung not in gezeigt,
        marker={"color": d.COMPOUND.get(mischung, d.MUTED),
                "line": {"color": d.BG, "width": 2}},
        hovertemplate=f"<b>{row.Driver}</b><br>{mischung.title()}<br>"
                      f"Runde {row.start}\N{EN DASH}{row.end} "
                      f"({row.laps} Runden)<extra></extra>"))
    gezeigt.add(mischung)

zeige(fig, hoehe=max(360, 24 * stints["Driver"].nunique()), barmode="stack",
      xaxis=achse("Runde"), yaxis=namensachse())
hinweis("Ein Balken je Stint, eingefaerbt nach Mischung. Jeder Wechsel ist "
        "ein Boxenstopp. Auffaellig kurze Stints am Anfang sind meist "
        "Schadensbegrenzung nach einem Zwischenfall, nicht Strategie.")

# --- Degradation je Stint -------------------------------------------------
st.markdown("##### Reifenabbau je Stint")

deg = f1lab.degradation(ses, threshold=schwelle, min_laps=min_runden)
if deg.empty:
    st.info("Kein Stint hat genug saubere Runden fuer eine Schaetzung. "
            "Mindestrunden senken oder Schwelle anheben.")
    st.stop()

belastbar = deg[deg["reliable"]]
k = st.columns(3)
k[0].metric("Geschaetzte Stints", len(deg))
k[1].metric("Davon belastbar", len(belastbar),
            help="Plausibilitaetspruefung aus f1lab: unplausible Steigungen "
                 "und zu schwache Fits fallen raus.")
if not belastbar.empty:
    schlimmster = belastbar.iloc[-1]
    k[2].metric("Staerkster Abbau", f"{schlimmster['deg_s_per_lap']:.3f} s/Runde",
                f"{schlimmster['driver']} auf "
                f"{str(schlimmster['compound']).title()}")

fig = go.Figure()
for mischung, teil in deg.groupby("compound"):
    name = str(mischung).upper()
    fig.add_trace(go.Scatter(
        x=teil["laps"], y=teil["deg_s_per_lap"], mode="markers", name=name,
        customdata=teil[["driver", "stint", "r2"]],
        marker={"size": 13, "color": d.COMPOUND.get(name, d.MUTED),
                "line": {"width": 1.5, "color": d.BG},
                "symbol": ["circle" if r else "x" for r in teil["reliable"]]},
        hovertemplate="<b>%{customdata[0]}</b> Stint %{customdata[1]}<br>"
                      "%{y:.3f} s pro Runde<br>%{x} Runden, "
                      "R2 %{customdata[2]:.2f}<extra></extra>"))

fig.add_hline(y=0, line_color=d.MUTED, line_width=1, line_dash="dot")
zeige(fig, xaxis=achse("Ausgewertete Runden im Stint"),
      yaxis=achse("Sekunden pro Runde Reifenalter"))
hinweis("Kreuze sind Fits, die die Plausibilitaetspruefung nicht bestanden "
        "haben - meist zu wenige Runden oder ein Stint, in dem Verkehr die "
        "Rundenzeit staerker bestimmt hat als der Reifen. Werte unter null "
        "heissen nicht, dass der Reifen besser wird: dort ist der Fit schlicht "
        "nicht identifiziert.")

# --- Je Mischung ----------------------------------------------------------
je_mischung = f1lab.degradation_by_compound(ses, threshold=schwelle,
                                            min_laps=min_runden)
links, rechts = st.columns([1, 1])

with links:
    st.markdown("##### Mittel je Mischung")
    if je_mischung.empty:
        st.info("Kein belastbarer Fit uebrig.")
    else:
        fig = go.Figure(go.Bar(
            x=[str(c).title() for c in je_mischung.index],
            y=je_mischung["mean"],
            marker={"color": [d.COMPOUND.get(str(c).upper(), d.MUTED)
                              for c in je_mischung.index],
                    "line": {"color": d.BG, "width": 2}},
            error_y={"type": "data", "array": je_mischung["std"].fillna(0),
                     "color": d.MUTED, "thickness": 1.4},
            customdata=je_mischung[["stints"]],
            hovertemplate="%{x}<br>%{y:.3f} s pro Runde<br>"
                          "%{customdata[0]} Stints<extra></extra>"))
        zeige(fig, hoehe=380, showlegend=False,
              xaxis=namensachse(), yaxis=achse("s pro Runde"))
        hinweis("Nur belastbare Fits. Die Fehlerbalken sind "
                "Standardabweichungen zwischen den Stints, keine "
                "Konfidenzintervalle - bei zwei oder drei Stints je Mischung "
                "sagen sie wenig.")

with rechts:
    st.markdown("##### Alle Fits")
    tabelle(deg.rename(columns={
        "driver": "Fahrer", "team": "Team", "stint": "Stint",
        "compound": "Mischung", "laps": "Runden",
        "deg_s_per_lap": "Abbau [s/Runde]", "base_s": "Basiszeit [s]",
        "r2": "R2", "reliable": "belastbar"}), height=380)

with st.expander("Warum die Treibstoffkorrektur noetig ist"):
    st.markdown(
        "Ein Auto verliert ueber die Renndistanz rund 100 kg Sprit und wird "
        "dadurch kontinuierlich schneller - Groessenordnung 0,03 Sekunden je "
        "Kilogramm. Ueber einen Stint von 20 Runden ist das gut eine Sekunde, "
        "die der Reifen scheinbar gutmacht.\n\n"
        "`fuel_correct()` normiert alle Rundenzeiten auf leeren Tank, bevor "
        "die Steigung geschaetzt wird. Die verwendeten Faustwerte "
        "(1,8 kg pro Runde, 0,03 s pro kg) sind Literaturwerte - die "
        "Groessenordnung stimmt, exakte Messwerte sind es nicht.")
