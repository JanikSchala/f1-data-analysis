"""Verkehr (P41): was P35s Rundenzeitmodell nicht sieht - ein Auto gegen die
Uhr kennt kein Herauskommen hinter einem Langsameren.

Nutzt dieselbe f1lab.race_config_from_session() wie die Strategie-
Optimierer-Seite, simuliert daneben aber ein Duell gegen einen Rivalen mit
f1lab.gap_evolution()/traffic_cost() - eine Ueberholwahrscheinlichkeit statt
der Annahme, dass ein Tempovorteil sich sofort in Position uebersetzt.
Rivalen-Abstand, Tempo-Delta und Ueberholwahrscheinlichkeit sind hier
bewusst einstellbare Szenario-Parameter, keine aus der Session gemessenen
Werte (siehe P41).
"""
from __future__ import annotations

import random

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
    zeige,
)

import f1lab
from f1lab import design as d

pfad = setup("Verkehr-Simulation", "Was P35s Rundenzeitmodell nicht sieht: "
                                   "Ueberholwahrscheinlichkeit statt "
                                   "sofortiger Positionsuebernahme.")
if kein_cache_hinweis(pfad):
    st.stop()

auswahl = sidebar_session(pfad)
ses = lade(auswahl)
kopfzeile(ses, auswahl)

if nur_rennen(auswahl, "Die Verkehr-Simulation"):
    st.stop()

try:
    cfg = f1lab.race_config_from_session(ses)
except ValueError as exc:
    st.info(f"Kein Modell fuer diese Session bildbar: {exc}")
    st.stop()

zweistopp = f1lab.optimal_strategy(cfg)
frontier = f1lab.frontier_by_stops(cfg, up_to=4)
kandidaten = {n: s for n, s in frontier.items() if s is not None and n != zweistopp.n_stops}
if not kandidaten:
    st.info("Keine zweite Stoppzahl fuer diese Session moeglich - kein "
           "Vergleich moeglich.")
    st.stop()

with st.sidebar:
    st.header("Szenario")
    alt_n = st.selectbox("Vergleichsplan (Stoppzahl)", sorted(kandidaten),
                         index=0)
    delta = st.slider("Tempo-Rivale [s/Runde langsamer]", 0.05, 0.5, 0.15,
                      step=0.05,
                      help="Rivale faehrt dieselben Boxenrunden wie der "
                           "Zweistopp-Plan, aber konstant so viel langsamer "
                           "je Runde - eine Annahme, kein Messwert.")
    start_gap = st.slider("Startabstand zum Rivalen [s]", 0.5, 8.0, 3.0,
                          step=0.5)
    p_overtake = st.slider("Ueberholwahrscheinlichkeit je blockierter Runde",
                           0.02, 0.6, 0.15, step=0.01,
                           help="Streckeneigenschaft (siehe P38: Kurven/km "
                                "korreliert mit Ueberholzahlen) - hier "
                                "bewusst frei einstellbar statt aus P38/P39 "
                                "berechnet, siehe P41-Docstring.")

alt = kandidaten[alt_n]
st.markdown(f"##### Zweistopp (Optimum) gegen {alt_n}-Stopp-Alternative")
k = st.columns(2)
k[0].metric("Zweistopp, frei", f"{zweistopp.green_time:.2f} s")
k[1].metric(f"{alt_n}-Stopp, frei", f"{alt.green_time:.2f} s",
           delta=f"{alt.green_time - zweistopp.green_time:+.2f} s",
           delta_color="inverse")


@st.cache_data(show_spinner="Verkehr wird simuliert ...")
def _traffic(cache_pfad: str, season: int, event: str, ident: str,
            alt_n: int, delta: float, start_gap: float, p_overtake: float):
    s = f1lab.load(season, event, ident, telemetry=False)
    c = f1lab.race_config_from_session(s)
    hero = f1lab.optimal_strategy(c)
    kand = {n: st_ for n, st_ in f1lab.frontier_by_stops(c, up_to=4).items()
           if st_ is not None}
    alt_ = kand[alt_n]
    hero_t = f1lab.lap_times_for_strategy(c, hero)
    alt_t = f1lab.lap_times_for_strategy(c, alt_)
    rivale_t = hero_t + delta
    c_hero, se_hero = f1lab.traffic_cost(hero_t, rivale_t, start_gap,
                                         p_overtake, n_sim=3000, seed=1)
    c_alt, se_alt = f1lab.traffic_cost(alt_t, rivale_t, start_gap,
                                       p_overtake, n_sim=3000, seed=1)
    verlauf, _ = f1lab.gap_evolution(hero_t, rivale_t, start_gap, p_overtake,
                                     rng=random.Random(1))
    return c_hero, se_hero, c_alt, se_alt, verlauf


c_hero, se_hero, c_alt, se_alt, verlauf = _traffic(
    str(pfad), auswahl.season, auswahl.event, auswahl.ident, alt_n, delta,
    start_gap, p_overtake)

t_hero = zweistopp.green_time + c_hero
t_alt = alt.green_time + c_alt

k2 = st.columns(2)
k2[0].metric("Zweistopp, mit Verkehr", f"{t_hero:.2f} s",
            help=f"+{c_hero:.2f}±{se_hero:.2f}s durch Verkehr")
k2[1].metric(f"{alt_n}-Stopp, mit Verkehr", f"{t_alt:.2f} s",
            delta=f"{t_alt - t_hero:+.2f} s", delta_color="inverse",
            help=f"+{c_alt:.2f}±{se_alt:.2f}s durch Verkehr")

if t_alt < t_hero:
    st.warning(f"Mit diesem Verkehrs-Szenario waere der {alt_n}-Stopp-Plan "
              "besser als das DAG-Optimum aus P35 - die Empfehlung kippt.")
else:
    naiv = alt.green_time - zweistopp.green_time
    mit_verkehr = t_alt - t_hero
    hinweis(f"Der Zweistopp bleibt besser, aber die Marge waechst von "
           f"{naiv:+.2f}s (ohne Verkehr) auf {mit_verkehr:+.2f}s (mit "
           "Verkehr) - der Vergleichsplan verbringt typischerweise mehr "
           "Rennstrecke in Reichweite des Rivalen, bevor die Flagge faellt "
           "(siehe P41).")

st.markdown("##### Ein Beispiel-Rennverlauf (Zweistopp gegen Rivale)")
fig = go.Figure()
fig.add_trace(go.Scatter(y=verlauf, mode="lines", line={"color": d.SERIEN[0]}))
fig.add_hline(y=0, line_color=d.MUTED, line_width=1)
zeige(fig, hoehe=360, showlegend=False, xaxis=achse("Runde"),
     yaxis=achse("Abstand zum Rivalen [s] (positiv = dahinter)"))
hinweis("Eine einzelne Zufallsrealisierung, nicht der Mittelwert - je nach "
       "Zufall sieht ein Nachladen dieser Seite mit anderem Seed anders "
       "aus. Die Kennzahlen oben sind der Mittelwert ueber 3000 Laeufe.")

with st.expander("Was diese Simulation nicht ist"):
    st.markdown(
        "Kein Ersatz fuer P35s exakten DAG-Optimierer - der bleibt "
        "unangetastet, diese Seite rechnet daneben. Rivalen-Tempo, "
        "Startabstand und Ueberholwahrscheinlichkeit sind Szenario-Annahmen, "
        "keine aus dieser Session gemessenen Werte (P38/P39 liefern die "
        "Groessenordnung fuer p_overtake, keine exakte Formel - siehe "
        "P41). Der Rivale faehrt dieselben Boxenrunden wie der "
        "Zweistopp-Plan, damit nur der Tempounterschied die Dynamik "
        "treibt, nicht zufaellige Stopp-Zeitpunkte.")
