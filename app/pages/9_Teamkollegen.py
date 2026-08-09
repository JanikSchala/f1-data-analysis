"""Teamkollegen-Duell mit Elo-Rating ueber eine ganze Saison.

Anders als jede andere Seite hier: keine einzelne Session, sondern ein Scan
ueber den ganzen Kalender (Quali + Rennen je Wochenende). Das dauert beim
ersten Aufruf eine Weile - Streamlit cacht das Ergebnis danach fuer den Rest
der Sitzung. Die Duell-Extraktion selbst kommt aus f1lab.teammate_duels()
(siehe P05), die Elo-Mechanik aus f1lab.elo_update() - der Saison-Scan bleibt
hier, aus demselben Grund wie im Skript: er wird nur hier gebraucht.
"""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from common import achse, hinweis, inventar, kein_cache_hinweis, setup, tabelle, zeige

import f1lab
from f1lab import design as d

START_ELO = 1500.0
K = 24.0

pfad = setup("Teamkollegen-Elo", "Wer gewinnt die internen Duelle - eine "
                                 "ganze Saison lang.")
if kein_cache_hinweis(pfad):
    st.stop()

inv = inventar(str(pfad))
saisons = sorted(inv[inv["timing"]]["season"].unique(), reverse=True)
with st.sidebar:
    st.header("Saison")
    saison = st.selectbox("Jahr", saisons, key="elo_saison")


@st.cache_data(show_spinner=False)
def _saison_events(cache_pfad: str, saison: int) -> list[str]:
    """Wochenenden mit Quali UND Rennen im Cache, chronologisch."""
    i = f1lab.cached_sessions(cache_pfad)
    info = (i[(i["season"] == saison) & (i["timing"])]
           .groupby("event").agg(datum=("event_date", "min"),
                                 idents=("ident", lambda s: set(s)))
           .reset_index())
    info = info[info["idents"].apply(lambda s: {"Q", "R"} <= s)]
    return info.sort_values("datum")["event"].tolist()


@st.cache_data(show_spinner=False)
def _saison_duelle(cache_pfad: str, saison: int,
                   events: tuple[str, ...]) -> pd.DataFrame:
    f1lab.enable_cache(cache_pfad, offline=True)
    rows = []
    for rnd, gp in enumerate(events, start=1):
        for ident in ("Q", "R"):
            try:
                ses = f1lab.load(saison, gp, ident)
                duelle = f1lab.teammate_duels(ses)
            except Exception:
                continue
            for dd in duelle:
                rows.append({"round": rnd, "event": gp, "session": ident, **dd})
    return pd.DataFrame(rows)


def _elo_verlauf(duelle: pd.DataFrame, k: float = K,
                 start: float = START_ELO) -> pd.DataFrame:
    rating: dict[str, float] = {}
    verlauf = []
    for rnd, gruppe in duelle.sort_values(["round", "session"]).groupby("round"):
        for row in gruppe.itertuples():
            ra, rb = rating.get(row.a, start), rating.get(row.b, start)
            rating[row.a], rating[row.b] = f1lab.elo_update(
                ra, rb, row.score_a, k=k)
        for drv, elo in rating.items():
            verlauf.append({"round": rnd, "driver": drv, "elo": elo})
    return pd.DataFrame(verlauf)


events = _saison_events(str(pfad), saison)
if len(events) < 2:
    st.info(f"Zu wenige Wochenenden mit Quali UND Rennen im Cache fuer "
           f"{saison} - warme zuerst mehr Sessions vor (P01).")
    st.stop()

with st.spinner(f"Saison {saison} wird gescannt ({len(events)} Wochenenden, "
               f"Quali + Rennen) - beim ersten Mal dauert das eine Weile ..."):
    duelle = _saison_duelle(str(pfad), saison, tuple(events))

if duelle.empty:
    st.info("Keine Teamkollegen-Duelle gefunden.")
    st.stop()

verlauf = _elo_verlauf(duelle)
pivot = verlauf.pivot(index="round", columns="driver", values="elo")
endstand = pivot.iloc[-1].dropna().sort_values(ascending=False)
top = endstand.index[: d.MAX_SERIEN]

k = st.columns(3)
k[0].metric("Fuehrend", endstand.index[0], f"{endstand.iloc[0]:.0f} Elo")
k[1].metric("Duelle gesamt", len(duelle))
k[2].metric("Wochenenden", len(events))

st.markdown("##### Elo-Verlauf")
fig = go.Figure()
for drv in pivot.columns:
    if drv in top:
        continue
    fig.add_trace(go.Scatter(x=pivot.index, y=pivot[drv], mode="lines",
                             line={"color": d.GRID, "width": 1},
                             showlegend=False, hoverinfo="skip"))
for i, drv in enumerate(top):
    serie = pivot[drv].dropna()
    fig.add_trace(go.Scatter(
        x=serie.index, y=serie, mode="lines+markers", name=drv,
        line={"color": d.SERIEN[i], "width": 2.5}, marker={"size": 5},
        hovertemplate=f"<b>{drv}</b><br>Wochenende %{{x}}<br>"
                      "%{y:.0f} Elo<extra></extra>"))
fig.add_hline(y=START_ELO, line={"color": d.MUTED, "width": 0.8, "dash": "dot"})
zeige(fig, hoehe=520, xaxis=achse("Rennwoche"), yaxis=achse("Elo-Rating"))
hinweis(f"Ein Duell je Wochenende und Session-Typ (Quali, Rennen), nicht je "
        f"Runde - sonst wuerde eine Zehntelsekunde ueber 60 Runden als 60 "
        f"Siege zaehlen. Top {d.MAX_SERIEN} am Saisonende hervorgehoben, "
        f"Rest im Hintergrund.")

links, rechts = st.columns(2)
with links:
    st.markdown("##### Endstand")
    tabelle(endstand.rename("Elo").round(0).astype(int).reset_index()
           .rename(columns={"index": "Fahrer"}))
with rechts:
    st.markdown("##### Meiste Quali-Siege gegen den Teamkollegen")
    quali_siege = (duelle[duelle["session"] == "Q"].groupby("a").size()
                  .rename("Siege").sort_values(ascending=False).reset_index()
                  .rename(columns={"a": "Fahrer"}))
    tabelle(quali_siege.head(10))

with st.expander("Wie das Rating funktioniert"):
    st.markdown(
        "Jedes Wochenende liefert bis zu zwei Duelle je Team: die schnellste "
        "gueltige Quali-Runde und die bereinigte, treibstoffkorrigierte Race "
        "Pace (`f1lab.pace_table`, siehe P04). Der schnellere Teamkollege "
        "gilt als Sieger. Elo bewegt das Rating staerker, wenn das Ergebnis "
        "ueberrascht (ein niedrig bewerteter Fahrer schlaegt einen hoch "
        "bewerteten) und kaum, wenn es erwartet war - dieselbe Mechanik wie "
        "im Schach.")
