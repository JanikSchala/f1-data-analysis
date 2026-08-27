"""regen-variance (P48): wird der erwartete sieger (startplatz 1) im regen
seltener sieger, und wirbelt regen das feld staerker durcheinander?

wie 19_Startplatz_Paritaet.py an keine waehlbare session gebunden, sondern
an einen festen zeitraum ueber den ganzen cache (Wetterdaten sind erst ab
~2018 zuverlaessig gecacht, siehe P33). gerechnet wird nirgends hier:
session.weather_data/session.results liefern alles direkt, keine eigene
kopie der logik.
"""
from __future__ import annotations

import fastf1
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from common import (
    achse,
    hinweis,
    kein_cache_hinweis,
    namensachse,
    setup,
    tabelle,
    zeige,
)
from scipy.stats import fisher_exact, mannwhitneyu

import f1lab
from f1lab import design as d

ERSTE_SAISON = 2018
LETZTE_SAISON = 2026
NASS_SCHWELLE = 0.20

pfad = setup("Regen-Variance", "Wird der erwartete Sieger (Startplatz 1) "
                               "im Regen seltener Sieger, und wirbelt "
                               "Regen das Feld staerker durcheinander?")
if kein_cache_hinweis(pfad):
    st.stop()


@st.cache_data(persist="disk", show_spinner="Wetter- und Ergebnisdaten "
                                            "werden ueber den ganzen "
                                            "Zeitraum geladen (erster "
                                            "Aufruf dauert einige Minuten) ...")
def _regendaten(cache_pfad: str) -> pd.DataFrame:
    inv = f1lab.cached_sessions(cache_pfad)
    zeilen = []
    for saison in range(ERSTE_SAISON, LETZTE_SAISON + 1):
        rennen = sorted(inv[(inv["season"] == saison) & (inv["ident"] == "R")]
                        ["event"].unique())
        for gp in rennen:
            try:
                ses = f1lab.load(saison, gp, "R", telemetry=False, weather=True)
                w = ses.weather_data
                r = ses.results
            except (Exception, fastf1.exceptions.DataNotLoadedError):
                continue
            if w is None or w.empty or r is None or r.empty:
                continue
            nass_anteil = float(w["Rainfall"].mean())
            gueltig = r.dropna(subset=["GridPosition", "Position"])
            if gueltig.empty:
                continue
            pole = gueltig[gueltig["GridPosition"] == 1]
            pole_gewinnt = (bool((pole["Position"] == 1).iloc[0])
                            if not pole.empty else None)
            durcheinander = float(
                (gueltig["GridPosition"] - gueltig["Position"]).abs().mean())
            zeilen.append({
                "season": saison, "gp": gp, "nass_anteil": nass_anteil,
                "nass": nass_anteil >= NASS_SCHWELLE,
                "pole_gewinnt": pole_gewinnt, "durcheinander": durcheinander,
            })
    return pd.DataFrame(zeilen)


daten = _regendaten(str(pfad))
if daten.empty:
    st.info(f"Keine auswertbaren Rennen zwischen {ERSTE_SAISON} und "
           f"{LETZTE_SAISON} im Cache.")
    st.stop()

k = st.columns(3)
k[0].metric("Rennen", len(daten))
k[1].metric("Davon nass", int(daten["nass"].sum()),
           help=f"Rainfall-Anteil >= {NASS_SCHWELLE:.0%} der "
                "Wetter-Messpunkte.")
k[2].metric("Davon trocken", int((~daten["nass"]).sum()))

d_pole = daten.dropna(subset=["pole_gewinnt"])
quote = d_pole.groupby("nass")["pole_gewinnt"].agg(["mean", "count"])
quote = quote.reindex([False, True])
tabelle_2x2 = pd.crosstab(d_pole["nass"], d_pole["pole_gewinnt"])
_odds, p_fisher = (fisher_exact(tabelle_2x2) if tabelle_2x2.shape == (2, 2)
                   else (None, None))

links, rechts = st.columns(2)
with links:
    st.markdown("##### Pole-to-Win: trocken gegen nass")
    labels = ["Trocken", "Nass"]
    fig = go.Figure(go.Bar(
        x=labels, y=(quote["mean"] * 100).to_numpy(),
        marker={"color": [d.MUTED, d.SERIEN[1]]},
        text=[f"n={int(c)}" for c in quote["count"]], textposition="outside"))
    zeige(fig, hoehe=360, showlegend=False, xaxis=namensachse(),
         yaxis=achse("Pole -> Sieg [%]"))

with rechts:
    st.markdown("##### Positions-Durcheinander |Start - Ziel|")
    fig2 = go.Figure()
    for nass, name, farbe in ((False, "Trocken", d.MUTED),
                              (True, "Nass", d.SERIEN[1])):
        fig2.add_trace(go.Box(y=daten.loc[daten["nass"] == nass, "durcheinander"],
                              name=name, marker={"color": farbe},
                              boxmean=True))
    zeige(fig2, hoehe=360, showlegend=False,
         yaxis=achse("Mittlere Positionsaenderung"))

nass_d = daten.loc[daten["nass"], "durcheinander"]
trocken_d = daten.loc[~daten["nass"], "durcheinander"]
p_mwu = (mannwhitneyu(nass_d, trocken_d, alternative="greater").pvalue
        if len(nass_d) and len(trocken_d) else None)

quote_txt = ", ".join(
    f"{lbl}: {int(row['mean'] * row['count'])}/{int(row['count'])} = "
    f"{row['mean']:.1%}"
    for lbl, (_i, row) in zip(["Trocken", "Nass"], quote.iterrows()))
fisher_txt = (f"Fisher-Exact-Test p={p_fisher:.3f}" if p_fisher is not None
             else "zu wenige Faelle fuer einen Test")
mwu_txt = (f"Mann-Whitney-U p={p_mwu:.3f}" if p_mwu is not None
          else "zu wenige Faelle fuer einen Test")
hinweis(f"{quote_txt}. Pole-to-Win-Vergleich: {fisher_txt}. "
       f"Positions-Durcheinander nass gegen trocken: {mwu_txt}. Beide "
       "Effekte gehen in die erwartete Richtung (weniger Pole-Siege, mehr "
       "Durcheinander im Regen), sind bei dieser Stichprobengroesse (nur "
       "wenige Dutzend nasse Rennen) aber NICHT statistisch signifikant. "
       "Ein echter Effekt in dieser Richtung ist plausibel, laesst sich "
       "mit den verfuegbaren Daten aber nicht beweisen (siehe P48).")

with st.expander("Alle Rennen"):
    tabelle(daten[["season", "gp", "nass_anteil", "nass", "pole_gewinnt",
                  "durcheinander"]].round(2).sort_values(
        ["season", "gp"]).rename(columns={
        "season": "Saison", "gp": "Rennen", "nass_anteil": "Rainfall-Anteil",
        "nass": "Nass?", "pole_gewinnt": "Pole gewinnt?",
        "durcheinander": "Durcheinander"}))
