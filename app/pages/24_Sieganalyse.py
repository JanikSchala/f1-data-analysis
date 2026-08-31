"""Sieg-Attribution (P51): warum hat der Sieger dieses Rennens gewonnen?

wie 18_Ueberholschwierigkeit.py an keine in der Seitenleiste waehlbare
Session gebunden, sondern an eine ganze Saison - "wer hat wie gewonnen"
ist eine Frage je Rennen, aber die Uebersicht braucht die ganze Saison auf
einmal. Beide Reiter nutzen dieselbe, einmal je Saison berechnete
f1lab.sieg_attribution()-Tabelle (keine Telemetrie noetig, deshalb
schnell). Der Renn-Detail-Reiter laedt zusaetzlich die eine gewaehlte
Session neu, um den Positionsverlauf zu zeichnen.
"""
from __future__ import annotations

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

import f1lab
from f1lab import design as d

GRUENDE_REIHENFOLGE = ["Start-Vorteil", "Erkaempft auf der Strecke",
                       "Strategie/Boxenstopp", "Safety-Car-Wende",
                       "Ausfall des Rivalen", "Einbruch des Rivalen",
                       "Ungeklaert"]

pfad = setup("Sieganalyse", "Warum hat der Sieger gewonnen - Start, "
                            "Strategie, Ausfall des Rivalen, Safety Car "
                            "oder eine echte Ueberholung? Je Rennen einer "
                            "Saison.")
if kein_cache_hinweis(pfad):
    st.stop()
assert pfad is not None

inv = inventar(str(pfad))
saisons = sorted(inv[inv["ident"] == "R"]["season"].unique(), reverse=True)
with st.sidebar:
    st.header("Saison")
    saison = st.selectbox("Jahr", saisons, key="sieg_saison")


@st.cache_data(persist="disk", show_spinner="Sieg-Attribution wird "
                                            "berechnet (erster Aufruf je "
                                            "Saison dauert) ...")
def _saison_attribution(cache_pfad: str, saison: int) -> pd.DataFrame:
    inv_lokal = f1lab.cached_sessions(cache_pfad)
    rennen = sorted(inv_lokal[(inv_lokal["season"] == saison)
                              & (inv_lokal["ident"] == "R")]["event"].unique())
    zeilen = []
    for gp in rennen:
        try:
            ses = f1lab.load(saison, gp, "R", telemetry=False)
            r = f1lab.sieg_attribution(ses)
        except Exception:
            continue
        r["event"] = gp
        zeilen.append(r)
    return pd.DataFrame(zeilen)


daten = _saison_attribution(str(pfad), int(saison))
if daten.empty:
    st.info(f"Keine auswertbaren Rennen fuer {saison} im Cache.")
    st.stop()

tab_uebersicht, tab_detail = st.tabs(["Saisonuebersicht", "Renn-Detail"])

with tab_uebersicht:
    k = st.columns(3)
    k[0].metric("Rennen", len(daten))
    k[1].metric("Haeufigster Grund", str(daten["grund"].mode().iloc[0]))
    k[2].metric("Anteil Start-Vorteil",
               f"{(daten['grund'] == 'Start-Vorteil').mean():.0%}")

    st.markdown("##### Warum hat der Sieger gewonnen?")
    counts = (daten["grund"].value_counts()
             .reindex(GRUENDE_REIHENFOLGE).dropna().astype(int))
    farben = [d.SERIEN[0] if g == counts.idxmax() else d.MUTED
             for g in counts.index]
    fig = go.Figure(go.Bar(x=counts.to_numpy(), y=counts.index,
                           orientation="h", marker={"color": farben}))
    zeige(fig, hoehe=340, showlegend=False, xaxis=achse("Anzahl Rennen"),
         yaxis=namensachse())
    hinweis("f1lab.sieg_attribution() (P51) ermittelt je Rennen die letzte "
           "Fuehrungsuebernahme des Siegers, die bis zum Ziel haelt, und "
           "klassifiziert sie ueber vier Signale: Ausfall des vorherigen "
           "Fuehrenden, Safety-Car-/VSC-Fenster, Boxenstopp-Zeitpunkt des "
           "Rivalen, echter gruener Ueberholvorgang. 'Start-Vorteil' heisst: "
           "durchgehend gefuehrt, kein Rivale zu bewerten - auch bei einer "
           "Startplatzstrafe, sofern Runde 1 trotzdem in Fuehrung liegt.")

    st.markdown("##### Alle Rennen der Saison")
    tabelle(daten[["event", "sieger", "startplatz", "entscheidende_runde",
                  "alter_fuehrender", "grund", "pace_rang", "abstand_p2_s"]]
           .rename(columns={"event": "Rennen", "sieger": "Sieger",
                            "startplatz": "Startplatz",
                            "entscheidende_runde": "Entscheidende Runde",
                            "alter_fuehrender": "Vorher fuehrend",
                            "grund": "Grund", "pace_rang": "Pace-Rang",
                            "abstand_p2_s": "Abstand P2 [s]"}))

with tab_detail:
    auswahl_rennen = st.selectbox("Rennen", daten["event"].tolist(),
                                  key="sieg_rennen")
    zeile = daten[daten["event"] == auswahl_rennen].iloc[0]

    ses = f1lab.load(int(saison), auswahl_rennen, "R", telemetry=False)
    pos = f1lab.position_progression(ses)
    phasen = f1lab.track_status_phases(ses)
    neutral = phasen[phasen["label"].isin(["safety car", "vsc"])]
    sieger = zeile["sieger"]

    k2 = st.columns(4)
    k2[0].metric("Startplatz", int(zeile["startplatz"]))
    k2[1].metric("Fuehrungsanteil", f"{zeile['fuehrungsanteil']:.0%}")
    k2[2].metric("Pace-Rang", zeile["pace_rang"] if pd.notna(zeile["pace_rang"]) else "-")
    k2[3].metric("Abstand zu P2",
                f"{zeile['abstand_p2_s']:.1f}s" if pd.notna(zeile["abstand_p2_s"])
                else "-")

    st.markdown(f"##### {sieger} gewinnt: {zeile['grund']}")
    if pd.notna(zeile["entscheidende_runde"]):
        st.write(f"Uebernahm die Fuehrung endgueltig in Runde "
                f"{int(zeile['entscheidende_runde'])} von "
                f"{zeile['alter_fuehrender']}.")

    fig2 = go.Figure()
    for p in neutral.itertuples():
        fig2.add_vrect(x0=p.lap_start, x1=p.lap_end,
                       fillcolor=d.SERIEN[1], opacity=0.15, line_width=0)
    for drv in pos.columns:
        if drv == sieger:
            continue
        fig2.add_trace(go.Scatter(x=pos.index, y=pos[drv], mode="lines",
                                  line={"color": d.MUTED, "width": 0.9},
                                  opacity=0.4, showlegend=False,
                                  hoverinfo="skip"))
    fig2.add_trace(go.Scatter(x=pos.index, y=pos[sieger], mode="lines",
                              name=sieger,
                              line={"color": d.SERIEN[0], "width": 2.4}))
    if pd.notna(zeile["entscheidende_runde"]):
        fig2.add_vline(x=int(zeile["entscheidende_runde"]), line_color=d.FG,
                       line_dash="dash")
    zeige(fig2, hoehe=440, xaxis=achse("Runde"),
         yaxis=achse("Position", autorange="reversed", dtick=2))
    hinweis("Rot/violett schattiert: Safety-Car-/VSC-Phasen "
           "(f1lab.track_status_phases(), P18). Gestrichelte Linie: die "
           "Runde, ab der der Sieger die Fuehrung bis zum Ziel haelt.")

    if zeile["undercut_bilanz"] is not None:
        bilanz = zeile["undercut_bilanz"]
        st.caption(f"Eigene Undercut-Versuche in diesem Rennen: "
                  f"{bilanz['erfolge']}/{bilanz['versuche']} erfolgreich "
                  "(f1lab.undercut_duels(), P42).")
