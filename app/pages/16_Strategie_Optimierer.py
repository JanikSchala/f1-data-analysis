"""exakter boxenstopp-plan (P35): kuerzester pfad ueber alle legalen
strategien statt einer handvoll kandidaten, plus eine safety-car-politik per
rueckwaertsinduktion.

die mischungsparameter kommen aus f1lab.race_config_from_session(): echte
degradation (P13) und echter pitloss dieser session, keine von hand
gesetzten zahlen. gerechnet wird nirgends hier. jede zahl kommt aus
f1lab.optimal_strategy()/frontier_by_stops()/solve_policy()/hindsight_value().
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
    namensachse,
    nur_rennen,
    setup,
    sidebar_session,
    tabelle,
    zeige,
)

import f1lab
from f1lab import design as d

pfad = setup("Strategie-Optimierer", "Der exakt beste Boxenstopp-Plan dieses "
                                     "Rennens - und was Unsicherheit ueber "
                                     "das Safety Car kostet.")
if kein_cache_hinweis(pfad):
    st.stop()

auswahl = sidebar_session(pfad)
ses = lade(auswahl)
kopfzeile(ses, auswahl)

if nur_rennen(auswahl, "Der Strategie-Optimierer"):
    st.stop()

with st.sidebar:
    st.header("Modell")
    zwei_mischungen = st.checkbox(
        "Zweimischungs-Regel erzwingen", value=True,
        help="Vorschrift der FIA: mindestens zwei verschiedene "
             "Trockenmischungen im Rennen. Ausschalten, wenn fuer diese "
             "Session nur eine Mischung belastbar geschaetzt werden kann.")

try:
    cfg = f1lab.race_config_from_session(ses, require_two_compounds=zwei_mischungen)
except ValueError as exc:
    st.info(f"Kein Modell fuer diese Session bildbar: {exc}")
    st.stop()

st.markdown("##### Rundenzeitmodell je Mischung (aus dieser Session geschaetzt)")
k = st.columns(4)
k[0].metric("Runden", cfg.n_laps)
k[1].metric("Pitloss", f"{cfg.pit_loss:.1f} s")
k[2].metric("Mischungen (belastbar)", len(cfg.tyres))
k[3].metric("Kandidaten-Stints", len(f1lab.stint_arcs(cfg)))

tabelle(
    pd.DataFrame([
        {"Mischung": t.compound.title(), "Basiszeit [s]": round(t.base_time, 3),
        "Degradation [s/Runde]": round(t.deg_linear, 4),
        "max. Stintlaenge (beobachtet)": t.max_age}
        for t in cfg.tyres]))
hinweis("Basiszeit und Degradation sind der Median aller belastbaren "
        "Stint-Fits (`f1lab.degradation()`, siehe P13) je Mischung - "
        "dieselben Zahlen wie auf der Reifen-Seite. `max. Stintlaenge` ist "
        "die laengste tatsaechlich gefahrene Stint-Laenge, keine gemessene "
        "Haltbarkeitsgrenze.")

# --- optimum ----------------------------------------------------------------
st.markdown("##### Exaktes Optimum")
try:
    beste = f1lab.optimal_strategy(cfg)
except f1lab.InfeasibleRace as exc:
    st.info(f"Keine Strategie erfuellt die Bedingungen: {exc}")
    st.stop()

fig = go.Figure()
gezeigt = set()
for s in beste.stints:
    mischung = s.compound.upper()
    fig.add_trace(go.Bar(
        y=["Optimum"], x=[s.length], base=[s.start_lap - 1], orientation="h",
        name=mischung, legendgroup=mischung, showlegend=mischung not in gezeigt,
        marker={"color": d.COMPOUND.get(mischung, d.MUTED),
               "line": {"color": d.BG, "width": 2}},
        hovertemplate=f"<b>{mischung.title()}</b><br>Runde {s.start_lap}"
                      f"\N{EN DASH}{s.end_lap} ({s.length} Runden)<extra></extra>"))
    gezeigt.add(mischung)
zeige(fig, hoehe=180, barmode="stack", xaxis=achse("Runde"),
     yaxis={"visible": False})

k = st.columns(3)
k[0].metric("Stopps", beste.n_stops)
k[1].metric("Fahrzeit", f"{beste.green_time:.1f} s")
k[2].metric("Gesamtzeit (mit Treibstoff)", f"{beste.total_time:.1f} s",
           help="Fahrzeit plus der Treibstoffterm, der fuer jede Strategie "
                "gleich ist und nur zur Einordnung der Groessenordnung "
                "wieder addiert wird.")
hinweis(f"Boxenstopps am Ende von Runde {', '.join(str(p) for p in beste.pit_laps) or 'keine'}. "
        "Kein Kandidat unter den Kandidaten - der exakte kuerzeste Pfad "
        "ueber alle legalen Stint-Kombinationen (siehe P35).")

# --- bester plan je stoppzahl ------------------------------------------------
st.markdown("##### Bester Plan je Stoppzahl")
frontier = f1lab.frontier_by_stops(cfg, up_to=4)
zeilen = []
for n, s in frontier.items():
    if s is None:
        zeilen.append({"Stopps": n, "Mischungen": "nicht moeglich",
                       "Abstand zum Optimum [s]": None})
    else:
        zeilen.append({"Stopps": n, "Mischungen": " / ".join(s.compounds),
                       "Abstand zum Optimum [s]": round(
                           s.green_time - beste.green_time, 2)})

machbar = [z for z in zeilen if z["Abstand zum Optimum [s]"] is not None]
if machbar:
    fig2 = go.Figure(go.Bar(
        x=[f"{z['Stopps']}-Stopp" for z in machbar],
        y=[z["Abstand zum Optimum [s]"] for z in machbar],
        marker={"color": [d.SERIEN[0] if z["Abstand zum Optimum [s]"] == 0
                          else d.MUTED for z in machbar]},
        text=[f"+{z['Abstand zum Optimum [s]']:.1f}s" if z["Abstand zum Optimum [s]"] > 0
             else "Optimum" for z in machbar],
        textposition="outside"))
    zeige(fig2, hoehe=340, showlegend=False, xaxis=namensachse(),
         yaxis=achse("Abstand zum Optimum [s]"))
    hinweis("Der Abstand misst auch, wie viel Risiko man kauft: drei "
            "Sekunden sind ein Muenzwurf, fuenfundzwanzig nicht (siehe "
            "P35).")
tabelle(pd.DataFrame(zeilen))

lo, hi = max(5.0, cfg.pit_loss - 15), cfg.pit_loss + 15
grenzen = f1lab.pit_loss_crossovers(cfg, lo, hi)
if grenzen:
    hinweis(f"Die Stoppzahl kippt bei Pitloss {', '.join(f'{g:.1f}s' for g in grenzen)} "
            f"(aktuell {cfg.pit_loss:.1f}s) - solange der echte Pitloss "
            "dazwischen liegt, aendert sich am Plan nichts.")

# --- safety car ---------------------------------------------------------------
st.markdown("##### Was kostet die Unsicherheit ueber das Safety Car?")
hinweis("Ein Safety Car macht Boxenstopps billig, aber niemand weiss vorher, "
        "wann es kommt. Die optimale Politik ist deshalb kein fester Plan, "
        "sondern eine Regel: was pro Runde zu tun ist, abhaengig von "
        "Reifenalter und aktueller Flagge (siehe P35).")

s1, s2 = st.columns(2)
p_deploy = s1.slider("P(Safety Car kommt) je Runde", 0.0, 0.10, 0.025, 0.005)
p_end = s2.slider("P(Safety Car endet) je Runde", 0.05, 0.60, 0.25, 0.05)
prozess = f1lab.SafetyCarProcess(p_deploy=p_deploy, p_end=p_end)


@st.cache_data(show_spinner="Politik und Erwartungswerte werden berechnet ...")
def _sc_auswertung(cfg, prozess, plan):
    wert, _politik = f1lab.solve_policy(cfg, prozess)
    naiv = f1lab.expected_cost_of_plan(cfg, plan, prozess)
    blind, se = f1lab.hindsight_value(cfg, prozess, n=150, seed=0)
    return wert, naiv, blind, se


try:
    wert, naiv, blind, se = _sc_auswertung(cfg, prozess, beste)
except f1lab.InfeasibleRace as exc:
    st.info(f"Keine Politik erfuellt die Bedingungen: {exc}")
else:
    fig3 = go.Figure(go.Bar(
        x=["Optimum bei bekanntem\nVerlauf", "Optimale Politik",
          "Fester Plan,\nstur durchgezogen"],
        y=[blind, wert, naiv],
        marker={"color": [d.MUTED, d.SERIEN[0], d.SERIEN[1]]},
        text=[f"{v:.1f}s" for v in (blind, wert, naiv)], textposition="outside"))
    zeige(fig3, hoehe=360, showlegend=False, xaxis=namensachse(),
         yaxis=achse("Erwartete Fahrzeit [s]"))

    k = st.columns(2)
    k[0].metric("Preis des Nichtwissens", f"{wert - blind:.2f} s",
               help="optimale Politik minus Optimum bei bekanntem Verlauf - "
                    "was es kostet, den Rennverlauf nicht im Voraus zu "
                    "kennen.")
    k[1].metric("Wert des Reagierens", f"{naiv - wert:.2f} s",
               help="fester Plan minus optimale Politik - was eine Politik, "
                    "die auf das Safety Car reagiert, gegenueber einem "
                    "stur durchgezogenen Plan bringt.")
    hinweis(f"'Optimum bei bekanntem Verlauf' ist ueber {150} simulierte "
            "Rennverlaeufe gemittelt (Standardfehler "
            f"\N{PLUS-MINUS SIGN}{se:.2f}s) - eine untere Schranke, die kein "
            "reales Team erreichen kann, weil niemand den Rennverlauf im "
            "Voraus kennt.")

with st.expander("Was das Modell nicht kann"):
    st.markdown(
        "Es faehrt genau ein Auto gegen die Uhr - keine Position, kein "
        "Ueberholen, kein Herauskommen hinter einem Langsameren. Verkehr "
        "wuerde die Trennbarkeit der Stintkosten brechen, auf der der "
        "kuerzeste Pfad beruht, und braeuchte ein anderes Verfahren "
        "(Simulation oder ein ganzzahliges Programm) - siehe P35.")
