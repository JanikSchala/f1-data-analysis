"""boxenstopp-performance-ranking einer ganzen saison, aus Ergast/jolpica.

anders als jede andere seite hier braucht das eine echte
netzwerkverbindung. Ergast/jolpica ist eine eigene API, kein teil des
lokalen FastF1-caches, den der rest der app ausschliesslich offline liest.
das ergebnis wird deshalb mit persist="disk" gecacht: nur der erste aufruf
je saison braucht netz, danach kommt es aus streamlits eigenem diskcache.
der saison-scan (paging, retry) bleibt seitenlokal, aus demselben grund wie
in 9_Teamkollegen.py: er wird nur hier gebraucht (siehe P05-begruendung
dort).

kennzahlen wie in P16: Ergasts `duration`-feld ist die volle
boxengassen-zeit (einfahrt- bis ausfahrt-zeitschranke), nicht die reine
standzeit. DAUER_MIN/DAUER_MAX unten filtern danach.
"""
from __future__ import annotations

import time

import numpy as np
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
from fastf1.ergast import Ergast

from f1lab import design as d

DAUER_MIN, DAUER_MAX = 15.0, 35.0

pfad = setup("Boxenstopps", "Boxengassen-Zeiten einer ganzen Saison, aus "
                            "Ergast/jolpica - braucht Netzwerk.")
if kein_cache_hinweis(pfad):
    st.stop()

inv = inventar(str(pfad))
saisons = sorted(inv["season"].unique(), reverse=True)
with st.sidebar:
    st.header("Saison")
    saison = st.selectbox("Jahr", saisons, key="pit_saison")


def _mit_wiederholung(fn, *args, versuche: int = 5, **kwargs):
    for i in range(versuche):
        try:
            return fn(*args, **kwargs)
        except Exception:
            if i == versuche - 1:
                raise
            time.sleep(3 * (i + 1))


@st.cache_data(persist="disk", show_spinner=False)
def _saison_pitstops(saison: int) -> pd.DataFrame:
    erg = Ergast(result_type="pandas", auto_cast=True)
    sched = _mit_wiederholung(erg.get_race_schedule, season=saison)

    pit_frames, res_frames = [], []
    for _, race in sched.iterrows():
        rnd = int(race["round"])
        stops = _mit_wiederholung(erg.get_pit_stops, season=saison,
                                  round=rnd, limit=200).content
        res = _mit_wiederholung(erg.get_race_results, season=saison,
                                round=rnd).content
        if stops and not stops[0].empty:
            s = stops[0].copy()
            s["round"] = rnd
            pit_frames.append(s)
        if res and not res[0].empty:
            r = res[0].copy()
            r["round"] = rnd
            res_frames.append(r)
        time.sleep(0.4)

    if not pit_frames:
        return pd.DataFrame()

    pit_raw = pd.concat(pit_frames, ignore_index=True)
    res = (pd.concat(res_frames, ignore_index=True) if res_frames
          else pd.DataFrame(columns=["round", "driverId", "constructorName",
                                     "position"]))
    pit_raw["dur"] = pd.to_timedelta(
        pit_raw["duration"], errors="coerce").dt.total_seconds()
    pit = pit_raw[pit_raw["dur"].between(DAUER_MIN, DAUER_MAX)].copy()
    return pit.merge(res[["round", "driverId", "constructorName", "position"]],
                     on=["round", "driverId"], how="left")


try:
    with st.spinner(f"Saison {saison} wird von Ergast/jolpica geladen "
                   "(gebremst wegen Rate-Limit, dauert beim ersten Mal "
                   "eine Weile) ..."):
        pit = _saison_pitstops(int(saison))
except Exception as exc:
    st.error(f"Ergast/jolpica war nicht erreichbar: {exc}")
    st.stop()

if pit.empty:
    st.info(f"Keine verwertbaren Boxenstopp-Daten fuer {saison} bei "
           "Ergast/jolpica - bei sehr alten Saisons fehlt das "
           "`duration`-Feld meist komplett.")
    st.stop()

rank = (pit.groupby("constructorName")["dur"]
        .agg(Median="median", Bester="min",
             IQR=lambda s: s.quantile(0.75) - s.quantile(0.25),
             Stopps="count").sort_values("Median"))

k = st.columns(3)
k[0].metric("Teams", len(rank))
k[1].metric("Boxenstopps", len(pit))
k[2].metric("Median-Spannweite",
           f"{rank['Median'].max() - rank['Median'].min():.1f} s",
           help="Abstand zwischen dem schnellsten und langsamsten "
                "Team-Median - siehe P16, meist unter 5% der Stoppzeit "
                "selbst.")

# --- ranking ----------------------------------------------------------------
st.markdown("##### Ranking nach Median")
order = rank.sort_values("Median", ascending=False)
teams = list(order.index)
schwelle_top3 = rank["Median"].nsmallest(3).max()
farben = [d.SERIEN[0] if v <= schwelle_top3 else d.MUTED for v in order["Median"]]

fig = go.Figure()
for t in teams:
    fig.add_trace(go.Scatter(
        x=[order.loc[t, "Bester"], order.loc[t, "Median"]], y=[t, t],
        mode="lines", line={"color": d.GRID, "width": 2},
        showlegend=False, hoverinfo="skip"))
fig.add_trace(go.Scatter(
    x=order["Bester"], y=teams, mode="markers", name="Bester Stopp",
    marker={"symbol": "line-ns-open", "size": 12, "color": d.MUTED,
           "line": {"width": 2}},
    hovertemplate="%{y}<br>Bester: %{x:.2f}s<extra></extra>"))
fig.add_trace(go.Scatter(
    x=order["Median"], y=teams, mode="markers", name="Median",
    marker={"size": 12, "color": farben},
    hovertemplate="%{y}<br>Median: %{x:.2f}s<extra></extra>"))
zeige(fig, hoehe=max(360, 30 * len(teams)), showlegend=False,
     xaxis=achse("Boxengassen-Zeit [s]"), yaxis=namensachse())
hinweis("Punkt = Median, Strich = bester Einzelstopp des Teams. Kein "
        "Fehlerbalken mit dem vollen Interquartilsabstand - der reicht "
        "teils ueber 5s und wuerde die eigentliche Aussage (die Mediane "
        "liegen extrem dicht beieinander) auf dieser Achse unlesbar "
        "machen. Der IQR steht in der Tabelle.")

tabelle(rank.round(2).reset_index().rename(columns={
    "constructorName": "Team", "Median": "Median [s]", "Bester": "Bester [s]",
    "IQR": "IQR [s]", "Stopps": "Stopps"}))

# --- AUSBAUSTUFE: druck ------------------------------------------------------
links, rechts = st.columns(2)

with links:
    st.markdown("##### Volles Boxenfenster kostet Zeit")
    stapel = pit.groupby(["round", "lap"]).size().rename("n_stops")
    p = pit.merge(stapel, on=["round", "lap"])
    iso = p.loc[p["n_stops"] == 1, "dur"]
    voll = p.loc[p["n_stops"] >= 4, "dur"]
    if iso.empty or voll.empty:
        st.info("Zu wenige isolierte bzw. vollbesetzte Boxenfenster fuer "
               "diese Saison.")
    else:
        fig2 = go.Figure()
        fig2.add_trace(go.Box(y=iso, name="isoliert\n(1 Stopp)",
                              marker={"color": d.SERIEN[0]}))
        fig2.add_trace(go.Box(y=voll, name="Boxenfenster\n(>=4 gleichzeitig)",
                              marker={"color": d.SERIEN[1]}))
        delta = voll.median() - iso.median()
        zeige(fig2, hoehe=380, showlegend=False,
             yaxis=achse("Boxengassen-Zeit [s]"))
        hinweis(f"Volles Boxenfenster (4+ Autos zeitgleich, meist SC/VSC) "
               f"kostet {delta:+.1f}s Median gegen einen isolierten "
               "Einzelstopp - der Boxengassen-Stau, nicht die Crew, "
               "erklaert den Unterschied.")

with rechts:
    st.markdown("##### Position vs. Stoppzeit")
    ok = pit.dropna(subset=["position"])
    if len(ok) < 10:
        st.info("Zu wenige Stopps mit bekannter Zielposition.")
    else:
        slope, inter = np.polyfit(ok["position"], ok["dur"], 1)
        korr = np.corrcoef(ok["position"], ok["dur"])[0, 1]
        xs = np.linspace(ok["position"].min(), ok["position"].max(), 50)
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(
            x=ok["position"], y=ok["dur"], mode="markers", name="Stopps",
            marker={"size": 5, "color": d.MUTED, "opacity": 0.5}))
        fig3.add_trace(go.Scatter(
            x=xs, y=slope * xs + inter, mode="lines", name="Trend",
            line={"color": d.SERIEN[1], "width": 2.5}))
        zeige(fig3, hoehe=380, showlegend=False,
             xaxis=achse("Zielposition"), yaxis=achse("Boxengassen-Zeit [s]"))
        hinweis(f"r={korr:+.2f} - vorne klassierte Fahrer stoppen im Median "
                "minimal schneller, aber der Effekt ist schwach (siehe "
                "P16). Der Boxenfenster-Effekt links zeigt den Drucksignal "
                "deutlicher als die Tabellenposition.")

with st.expander("Warum das Ranking fast bedeutungslos fuer Crew-Leistung ist"):
    st.markdown(
        "Ergasts `duration`-Feld ist nicht die reine Standzeit (die "
        "Groessenordnung, die als \"2-Sekunden-Stopp\" durch die "
        "Uebertragung geht), sondern die volle Boxengassen-Zeit von der "
        "Einfahrt- bis zur Ausfahrt-Zeitschranke - dominiert von der festen "
        "Tempolimit-Strecke, nicht vom eigentlichen Reifenwechsel (der "
        "macht nur rund 2s davon aus). Der urspruenglich angenommene "
        "1.5-6.0s-Filter verwarf deshalb 2024 ausnahmslos jeden Stopp; "
        "korrigiert auf 15-35s (Werte darueber sind Reparaturen oder "
        "Strafen).")
