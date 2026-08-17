"""Startplatz-Paritaet (P40): hat die Seite der Startaufstellung (gerade/
ungerade Startplatz) einen echten, ueber Jahre konsistenten Effekt?

Wie 15_MachineLearning.py an keinen waehlbaren Zeitraum gebunden, sondern
an den ganzen Cache (alle Saisons, alle Strecken) - die Frage braucht
moeglichst viele Jahre je Strecke, nicht eine einzelne Session. Gerechnet
wird nirgends hier: f1lab.grid_lap1_positions() liefert Startplatz und
Runde-1-Position je Fahrer, keine Telemetrie noetig.
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
    kein_cache_hinweis,
    namensachse,
    setup,
    tabelle,
    zeige,
)
from scipy.stats import ttest_ind

import f1lab
from f1lab import design as d

MIN_SAISONS = 4
KONSISTENZ_STRECKEN = ["Saudi Arabian Grand Prix", "Australian Grand Prix",
                       "Belgian Grand Prix", "Dutch Grand Prix",
                       "Mexico City Grand Prix"]

pfad = setup("Startplatz-Paritaet", "Gerade oder ungerade Startplaetze - "
                                    "hat die 'schmutzige Seite' einen "
                                    "echten, ueber Jahre konsistenten "
                                    "Effekt?")
if kein_cache_hinweis(pfad):
    st.stop()


@st.cache_data(persist="disk", show_spinner="Startdaten werden ueber den "
                                            "ganzen Cache geladen (erster "
                                            "Aufruf dauert einige Minuten) ...")
def _startdaten(cache_pfad: str):
    inv = f1lab.cached_sessions(cache_pfad)
    rennen = inv[inv["ident"] == "R"][["season", "event"]].drop_duplicates()
    alle, saison_diffs = [], []
    for _, r in rennen.iterrows():
        saison, gp = int(r["season"]), r["event"]
        try:
            ses = f1lab.load(saison, gp, "R", telemetry=False)
            m = f1lab.grid_lap1_positions(ses)
        except Exception:
            continue
        time.sleep(0.2)      # ergast/jolpica etwas schonen (siehe P40)
        if m.empty:
            continue
        m["parity"] = np.where(m["grid"].astype(int) % 2 == 0,
                               "gerade", "ungerade")
        m["gp"], m["season"] = gp, saison
        alle.append(m)
        if gp in KONSISTENZ_STRECKEN:
            ung = m.loc[m["parity"] == "ungerade", "gewinn"]
            ger = m.loc[m["parity"] == "gerade", "gewinn"]
            saison_diffs.append({"gp": gp, "season": saison,
                                 "diff": ung.mean() - ger.mean()})
    return pd.concat(alle, ignore_index=True), pd.DataFrame(saison_diffs)


df, saison_diffs = _startdaten(str(pfad))

zeilen = []
for gp, g in df.groupby("gp"):
    if g["season"].nunique() < MIN_SAISONS:
        continue
    ung = g.loc[g["parity"] == "ungerade", "gewinn"]
    ger = g.loc[g["parity"] == "gerade", "gewinn"]
    t, p = ttest_ind(ung, ger)
    zeilen.append({"gp": gp, "saisons": g["season"].nunique(), "n": len(g),
                   "diff": round(ung.mean() - ger.mean(), 2), "p": round(p, 4)})
ergebnisse = pd.DataFrame(zeilen).sort_values("p")

k = st.columns(4)
k[0].metric("Starts (ohne Boxenstarts)", len(df))
k[1].metric("Rennen", df[["season", "gp"]].drop_duplicates().shape[0])
k[2].metric("Strecken getestet", len(ergebnisse),
           help=f"Nur Strecken mit mindestens {MIN_SAISONS} Saisons im "
                "Cache - sonst ist der t-Test nicht belastbar.")
signifikant = ergebnisse[ergebnisse["p"] < 0.05]
k[3].metric("Davon p<0.05", f"{len(signifikant)}/{len(ergebnisse)}",
           help=f"Zufallserwartung bei {len(ergebnisse)} Tests: "
                f"{0.05 * len(ergebnisse):.1f}. Bonferroni-Schwelle: "
                f"{0.05 / max(len(ergebnisse), 1):.4f}.")

st.markdown("##### Paritaets-Effekt je Strecke")
e = ergebnisse.sort_values("diff")
farben = [d.SERIEN[1] if p < 0.05 else d.MUTED for p in e["p"]]
fig = go.Figure(go.Bar(
    x=e["diff"], y=e["gp"], orientation="h", marker={"color": farben},
    text=[f"p={p:.3f}" for p in e["p"]], textposition="outside"))
fig.add_vline(x=0, line_color=d.MUTED, line_width=1)
zeige(fig, hoehe=max(320, 26 * len(e)), showlegend=False,
     xaxis=achse("Ungerade minus gerade Startplaetze [Positionen nach Runde 1]"),
     yaxis=namensachse())
hinweis("Rot markiert p<0.05. Bei so vielen getesteten Strecken sind ein "
       "paar zufaellige Treffer erwartbar (siehe Kennzahl oben) - der "
       "p-Wert allein beweist noch keinen echten Effekt, siehe Konsistenz-"
       "Check unten (P40).")

st.markdown("##### AUSBAUSTUFE: haelt die Richtung in jeder einzelnen Saison?")
fig2 = go.Figure()
for s in sorted(saison_diffs["season"].unique()):
    werte = saison_diffs[saison_diffs["season"] == s].set_index("gp")
    werte = werte.reindex(KONSISTENZ_STRECKEN)
    fig2.add_trace(go.Bar(
        x=[gp.replace(" Grand Prix", "") for gp in KONSISTENZ_STRECKEN],
        y=werte["diff"], name=str(s)))
fig2.add_hline(y=0, line_color=d.MUTED, line_width=1)
zeige(fig2, hoehe=380, barmode="group",
     xaxis=achse(""), yaxis=achse("Ungerade minus gerade [Positionen]"))
hinweis("Saudi-Arabien und die Niederlande zeigen in JEDER verfuegbaren "
       "Saison dieselbe Richtung - deutlich staerkere Evidenz als der "
       "gepoolte p-Wert allein. Australiens und Belgiens gepoolte Befunde "
       "haengen dagegen an einzelnen abweichenden Saisons (Australien 2023, "
       "Belgien gleich zweimal) - die unruhigsten der fuenf trotz p<0.05 "
       "(siehe P40).")

with st.expander("Tabelle: alle getesteten Strecken"):
    tabelle(ergebnisse.rename(columns={
        "gp": "Strecke", "saisons": "Saisons", "n": "Starts",
        "diff": "Diff. [Positionen]", "p": "p-Wert"}))
