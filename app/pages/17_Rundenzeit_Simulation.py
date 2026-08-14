"""Rundenzeit-Simulation (P37): ein Punktmassenmodell aus Streckenkruemmung
und vier Fahrzeugparametern, kalibriert gegen die echte Geschwindigkeitsspur
einer Referenzrunde - und dieselben Parameter unveraendert auf andere
Strecken uebertragen.

Gerechnet wird nirgends hier: jede Zahl kommt aus
f1lab.lap_speed_profile()/calibrate_lap_model()/simulate_lap().
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
    setup,
    sidebar_session,
    tabelle,
    zeige,
)

import f1lab
from f1lab import design as d

pfad = setup("Rundenzeit-Simulation", "Kruemmung plus vier "
                                      "Fahrzeugparameter statt Nachschlagen "
                                      "- kalibriert gegen echte Telemetrie, "
                                      "getestet auf anderen Strecken.")
if kein_cache_hinweis(pfad):
    st.stop()

auswahl = sidebar_session(pfad, nur_mit_telemetrie=True)
if auswahl is None:
    st.stop()
ses = lade(auswahl, telemetrie=True)
kopfzeile(ses, auswahl)


@st.cache_data(show_spinner="Kalibrierung laeuft ...")
def _kalibrierung(cache_pfad: str, season: int, event: str, ident: str):
    s = f1lab.load(season, event, ident, telemetry=True)
    dist, kappa, speed_real = f1lab.lap_speed_profile(s)
    t_real = float(s.laps.pick_fastest()["LapTime"].total_seconds())
    params = f1lab.calibrate_lap_model(dist, kappa, speed_real)
    v_sim, t_sim = f1lab.simulate_lap(dist, kappa, params["mu_g"],
                                      params["a_accel"], params["a_brake"],
                                      params["v_top"])
    return dist, speed_real, v_sim, t_real, t_sim, params


dist, speed_real, v_sim, t_real, t_sim, params = _kalibrierung(
    str(pfad), auswahl.season, auswahl.event, auswahl.ident)

st.markdown("##### Kalibrierte Fahrzeugparameter (kleinste Quadrate gegen "
           "die echte Geschwindigkeitsspur)")
k = st.columns(5)
k[0].metric("Kurvengrenze", f"{params['mu_g'] / 9.81:.2f} g",
           help=f"{params['mu_g']:.1f} m/s\N{SUPERSCRIPT TWO}")
k[1].metric("Beschleunigung", f"{params['a_accel']:.1f} m/s\N{SUPERSCRIPT TWO}")
k[2].metric("Bremsverzoegerung", f"{params['a_brake']:.1f} m/s\N{SUPERSCRIPT TWO}")
k[3].metric("Hoechstgeschwindigkeit", f"{params['v_top'] * 3.6:.0f} km/h")
k[4].metric("Rundenzeit-Fehler",
           f"{100 * (t_sim - t_real) / t_real:+.1f}%",
           help=f"simuliert {t_sim:.2f}s gegen real {t_real:.2f}s "
                f"(RMSE Geschwindigkeit {params['rmse_ms']:.1f} m/s)")

fig = go.Figure()
fig.add_trace(go.Scatter(x=dist, y=speed_real * 3.6, name="Echte Telemetrie",
                         line={"color": d.MUTED}))
fig.add_trace(go.Scatter(x=dist, y=v_sim * 3.6, name="Simulation (kalibriert)",
                         line={"color": d.SERIEN[0], "dash": "dash"}))
zeige(fig, hoehe=380, xaxis=achse("Distanz [m]"), yaxis=achse("Speed [km/h]"))
hinweis("Referenzrunde: die schnellste dieser Session. Die Simulation kennt "
       "nur die Streckenkruemmung und die vier oben kalibrierten "
       "Fahrzeugparameter - keine echte Geschwindigkeit fliesst direkt "
       "ein (siehe P37).")

st.markdown("##### AUSBAUSTUFE: dieselben Parameter auf anderen Strecken")
hinweis("Nicht neu kalibriert - dieselben vier oben ermittelten "
       "Fahrzeugparameter werden unveraendert auf die Streckenkruemmung "
       "anderer Sessions angewendet. Traegt das an dieser Strecke "
       "identifizierte 'Auto' auch anderswo?")

inv = f1lab.cached_sessions(str(pfad))
kandidaten = inv[(inv["ident"] == auswahl.ident) & inv["telemetry"]
                 & (inv["event"] != auswahl.event)]
if auswahl.season in set(kandidaten["season"]):
    kandidaten = kandidaten[kandidaten["season"] == auswahl.season]

optionen = sorted(kandidaten["event"].unique())
if not optionen:
    st.info("Keine weitere Session mit Telemetrie und derselben Art im "
           "Cache, um die Uebertragung zu pruefen.")
    st.stop()

gewaehlt = st.multiselect(
    "Andere Strecken (dieselbe Session-Art, aus dem Cache)", optionen,
    default=optionen[: min(6, len(optionen))])
if not gewaehlt:
    st.stop()


@st.cache_data(show_spinner="Uebertragung wird gerechnet ...")
def _uebertragung(cache_pfad: str, season: int, ident: str,
                  events: tuple[str, ...], params: dict):
    zeilen = []
    for ev in events:
        jahr = int(kandidaten[kandidaten["event"] == ev]["season"].iloc[0])
        s = f1lab.load(jahr, ev, ident, telemetry=True)
        d2, k2, _ = f1lab.lap_speed_profile(s)
        t2_real = float(s.laps.pick_fastest()["LapTime"].total_seconds())
        _, t2_sim = f1lab.simulate_lap(d2, k2, params["mu_g"], params["a_accel"],
                                       params["a_brake"], params["v_top"])
        diff_pct = 100 * (t2_sim - t2_real) / t2_real
        zeilen.append({"strecke": ev, "real_s": round(t2_real, 2),
                       "sim_s": round(t2_sim, 2), "diff_pct": round(diff_pct, 1)})
    return pd.DataFrame(zeilen)


ergebnisse = _uebertragung(str(pfad), auswahl.season, auswahl.ident,
                           tuple(gewaehlt), params)
ergebnisse = ergebnisse.sort_values("diff_pct")

fig2 = go.Figure(go.Bar(
    x=ergebnisse["diff_pct"], y=ergebnisse["strecke"], orientation="h",
    marker={"color": [d.SERIEN[0] if abs(v) < 5 else d.SERIEN[1]
                      for v in ergebnisse["diff_pct"]]},
    text=[f"{v:+.1f}%" for v in ergebnisse["diff_pct"]], textposition="outside"))
fig2.add_vline(x=0, line_color=d.MUTED, line_width=1)
zeige(fig2, hoehe=max(280, 40 * len(ergebnisse)), showlegend=False,
     xaxis=achse("Abweichung Simulation von echter Rundenzeit [%]"),
     yaxis=namensachse())

fehler_abs = ergebnisse["diff_pct"].abs().mean()
fehler_signiert = ergebnisse["diff_pct"].mean()
hinweis(f"Mittlerer absoluter Fehler {fehler_abs:.1f}%, mittlerer Fehler mit "
       f"Vorzeichen {fehler_signiert:+.1f}% ueber {len(ergebnisse)} "
       "Strecken. Grosse Ausreisser sind meist keine Modellschwaeche im "
       "Kern, sondern eine zu grobe Vereinfachung: 2D-Kruemmung ohne "
       "Hoehenprofil, ein Fahrzeugparametersatz fuer alle Fluegel-/"
       "Abtriebseinstellungen (siehe P37).")

tabelle(ergebnisse.rename(columns={"strecke": "Strecke", "real_s": "Real [s]",
                                   "sim_s": "Simuliert [s]",
                                   "diff_pct": "Abweichung [%]"}))

with st.expander("Was das Modell nicht kann"):
    st.markdown(
        "Das Modell ist ein reines Punktmassenmodell auf der 2D-Ideallinie: "
        "Kruemmung begrenzt die Kurvengeschwindigkeit, zwei konstante "
        "Beschleunigungsgrenzen begrenzen Beschleunigen und Bremsen "
        "dazwischen. Nicht abgebildet sind Hoehenprofil (Steigungen und "
        "Gefaelle veraendern effektiven Grip und Bremsweg), "
        "streckenspezifische Fluegel-/Abtriebseinstellungen (ein Auto faehrt "
        "in Monza mit weniger Abtrieb als in Monaco - hier wird ein "
        "einziger v_top fuer alle Strecken verwendet), Reifentemperatur "
        "und -abbau innerhalb der Runde, sowie Fahrfehler oder Verkehr auf "
        "der Referenzrunde selbst. Die Uebertragung auf andere Strecken "
        "zeigt genau deshalb, wo die Grenze dieser Vereinfachung liegt "
        "(siehe P37).")
