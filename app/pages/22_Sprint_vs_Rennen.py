"""sprint gegen rennen (P44): wird im sprint wirklich weniger ueberholt?

feste zeitspanne (2023+2024, die beiden saisons mit dem aktuellen
sprint-format) statt waehlbarer saison, wie der 75-jahre-trend-teil von
14_Historie.py. sprint-wochenenden sind selten, eine einzelne saison waere
kaum aussagekraeftig. `persist="disk"`-gecacht. gerechnet wird nirgends
hier: f1lab.overtakes_matrix()/overtake_events() (P20/P39) zaehlen
identisch in sprint und rennen, nur die gegenueberstellung ist neu.
"""
from __future__ import annotations

import fastf1
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from common import achse, hinweis, kein_cache_hinweis, setup, tabelle, zeige
from scipy.stats import wilcoxon

import f1lab
from f1lab import design as d

SAISONS = (2023, 2024)

pfad = setup("Sprint vs. Rennen", "Ueberholt es sich im kurzen Sprint "
                                  "wirklich anders als im Hauptrennen "
                                  "desselben Wochenendes?")
if kein_cache_hinweis(pfad):
    st.stop()


@st.cache_data(show_spinner=False)
def _sprint_namen(saison: int) -> list[str]:
    sched = fastf1.get_event_schedule(saison, include_testing=False)
    return sched[sched["EventFormat"].str.contains(
        "sprint", case=False, na=False)]["EventName"].tolist()


@st.cache_data(persist="disk", show_spinner="Sprint- und Rennwochenenden "
                                            "werden verglichen (erster "
                                            "Aufruf dauert) ...")
def _vergleich(cache_pfad: str, saisons: tuple[int, ...]):
    zeilen, nass, anteile_s, anteile_r = [], [], [], []
    for saison in saisons:
        for gp in _sprint_namen(saison):
            try:
                ses_s = f1lab.load(saison, gp, "S", telemetry=False)
                ses_r = f1lab.load(saison, gp, "R", telemetry=False)
            except Exception:
                continue
            nasse_session = (ses_s.laps["Compound"].isin(
                ["INTERMEDIATE", "WET"]).any()
                or ses_r.laps["Compound"].isin(["INTERMEDIATE", "WET"]).any())
            if nasse_session:
                nass.append(f"{saison} {gp}")
                continue
            n_s = int(f1lab.overtakes_matrix(ses_s).values.sum())
            n_r = int(f1lab.overtakes_matrix(ses_r).values.sum())
            zeilen.append({
                "saison": saison, "gp": gp,
                "rate_sprint": n_s / ses_s.total_laps,
                "rate_rennen": n_r / ses_r.total_laps})
            ev_s = f1lab.overtake_events(ses_s)
            ev_r = f1lab.overtake_events(ses_r)
            if not ev_s.empty:
                anteile_s.append(ev_s["lap"] / ses_s.total_laps)
            if not ev_r.empty:
                anteile_r.append(ev_r["lap"] / ses_r.total_laps)
    df = pd.DataFrame(zeilen)
    as_ = pd.concat(anteile_s, ignore_index=True) if anteile_s else pd.Series(dtype=float)
    ar_ = pd.concat(anteile_r, ignore_index=True) if anteile_r else pd.Series(dtype=float)
    return df, nass, as_, ar_


df, nasse_wochenenden, anteile_sprint, anteile_rennen = _vergleich(
    str(pfad), SAISONS)

if df.empty:
    st.info("Keine auswertbaren trockenen Sprint-Wochenenden im Cache.")
    st.stop()

df["faktor"] = df["rate_rennen"] / df["rate_sprint"]

k = st.columns(4)
k[0].metric("Trockene Sprint-Wochenenden", len(df))
k[1].metric("Wegen Regen ausgeschlossen", len(nasse_wochenenden),
           help=", ".join(nasse_wochenenden) if nasse_wochenenden else None)
k[2].metric("Median Sprint", f"{df['rate_sprint'].median():.2f}/Runde")
k[3].metric("Median Rennen", f"{df['rate_rennen'].median():.2f}/Runde")

st.markdown("##### Ueberholungen pro Runde, je Wochenende")
e = df.sort_values("faktor")
labels = [f"{row.gp.replace(' Grand Prix', '')} {row.saison}"
         for row in e.itertuples()]
fig = go.Figure()
fig.add_trace(go.Bar(y=labels, x=e["rate_sprint"], orientation="h",
                     name="Sprint", marker={"color": d.SERIEN[0]}))
fig.add_trace(go.Bar(y=labels, x=e["rate_rennen"], orientation="h",
                     name="Rennen", marker={"color": d.SERIEN[1]},
                     text=[f"{f:.2f}x" for f in e["faktor"]],
                     textposition="outside"))
zeige(fig, hoehe=max(340, 40 * len(e)), barmode="group",
     xaxis=achse("Ueberholungen pro Runde (nur gruene Flagge)"))

diff = df["rate_rennen"] - df["rate_sprint"]
test = wilcoxon(diff)
hoeher = int((df["rate_rennen"] > df["rate_sprint"]).sum())
hinweis(f"Rennen ueberholt haeufiger als der Sprint desselben Wochenendes "
       f"in {hoeher}/{len(df)} Faellen (Wilcoxon p={test.pvalue:.4f}) - "
       "kein einziges Gegenbeispiel unter den trockenen Sprint-Wochenenden "
       "2023+2024 (siehe P44). Nasse Sessions ausgeschlossen: dieselbe "
       "Regen-Falle wie in P43, hier zwischen Sprint und Rennen desselben "
       "Wochenendes statt zwischen Rennen einer Saison.")

if not anteile_sprint.empty and not anteile_rennen.empty:
    st.markdown("##### Wann im Rennverlauf wird ueberholt?")
    fig2 = go.Figure()
    fig2.add_trace(go.Histogram(x=anteile_sprint, histnorm="probability density",
                                name=f"Sprint (n={len(anteile_sprint)})",
                                marker={"color": d.SERIEN[0]}, opacity=0.65,
                                xbins={"start": 0, "end": 1, "size": 0.1}))
    fig2.add_trace(go.Histogram(x=anteile_rennen, histnorm="probability density",
                                name=f"Rennen (n={len(anteile_rennen)})",
                                marker={"color": d.SERIEN[1]}, opacity=0.65,
                                xbins={"start": 0, "end": 1, "size": 0.1}))
    zeige(fig2, hoehe=380, barmode="overlay",
         xaxis=achse("Renndistanz, zu der die Ueberholung passiert"),
         yaxis=achse("Dichte"))
    hinweis("Das Rennen konzentriert Ueberholungen staerker im ersten "
           "Renndrittel (Startrunden-Chaos mit vollem Feld), der Sprint "
           "verschiebt sie relativ ins letzte Drittel - eine plausible "
           "Teilerklaerung, aber nicht die ganze: der Faktor bleibt auch "
           "innerhalb der Drittel bestehen (siehe P44, AUSBAUSTUFE).")

with st.expander("Tabelle: alle trockenen Sprint-Wochenenden"):
    tabelle(df.rename(columns={
        "saison": "Saison", "gp": "Rennen", "rate_sprint": "Sprint/Runde",
        "rate_rennen": "Rennen/Runde", "faktor": "Faktor"}).round(2))
