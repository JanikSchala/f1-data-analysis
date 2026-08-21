"""boxenstopp-kosten und undercut-rechnung.

der pitloss kommt aus den daten der gewaehlten session, das undercut-modell
rechnet daraufhin durch, was ein frueherer stopp einbraechte. die annahmen
sind stellschrauben, ihre wirkung sieht man erst, wenn man daran dreht.
"""
from __future__ import annotations

import numpy as np
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
    tabelle,
    zeige,
)

import f1lab
from f1lab import design as d

pfad = setup("Strategie", "Was kostet ein Stopp auf dieser Strecke, und wann "
                          "lohnt sich der Undercut?")
if kein_cache_hinweis(pfad):
    st.stop()

auswahl = sidebar_session(pfad)
ses = lade(auswahl)
kopfzeile(ses, auswahl)

# --- pitloss --------------------------------------------------------------
st.markdown("##### Zeitverlust eines Boxenstopps")

verlust = None
if not nur_rennen(auswahl, "Die Pitloss-Schaetzung"):
    try:
        verlust = f1lab.pit_loss(ses)
    except Exception as exc:
        st.info(f"Pitloss nicht schaetzbar - es fehlen In- oder Out-Laps. "
                f"({type(exc).__name__})")

if verlust is not None:
    k = st.columns(3)
    k[0].metric("Pitloss", f"{verlust:.2f} s")
    k[1].metric("Beobachtete Stopps", int(ses.laps["PitInTime"].notna().sum()))
    k[2].metric("Runden im Rennen", int(ses.total_laps))
    hinweis("Enthaelt Anfahrt, Standzeit und Ausfahrt - gemessen als Aufschlag "
            "von In- und Out-Lap gegenueber der normalen Rundenzeit desselben "
            "Fahrers. Der Median macht die Schaetzung robust gegen die Stopps, "
            "bei denen etwas schiefging.")

# --- undercut -------------------------------------------------------------
st.markdown("##### Lohnt der Undercut?")
hinweis("Der Verfolger stoppt frueher und faehrt auf frischen Reifen, waehrend "
        "der Vordermann weiter altert. Der Pitloss faellt fuer beide an und "
        "kuerzt sich heraus - entscheidend ist allein die Pace-Differenz im "
        "Fenster.")

# ses.total_laps ist nur fuer Rennen/Sprint gesetzt, fuer Qualifying/Practice
# faellt f1lab.fuel_correct() sonst mit TypeError auf None minus int.
je_mischung = (f1lab.degradation_by_compound(ses)
               if ses.total_laps is not None else None)
if je_mischung is not None and not je_mischung.empty:
    vor_alt = float(je_mischung["mean"].max())
    vor_neu = float(je_mischung["mean"].min())
    st.caption(
        f"Startwerte aus dieser Session: staerkster Abbau {vor_alt:.3f} s/Runde "
        f"({str(je_mischung['mean'].idxmax()).title()}), schwaechster "
        f"{vor_neu:.3f} s/Runde ({str(je_mischung['mean'].idxmin()).title()}).")
else:
    vor_alt, vor_neu = 0.08, 0.03
    st.caption("Kein belastbarer Fit in dieser Session - die Startwerte sind "
               "Faustwerte, bitte selbst setzen.")

s1, s2, s3 = st.columns(3)
deg_alt = s1.slider("Abbau alter Reifen [s/Runde]", 0.0, 0.30,
                    round(min(max(vor_alt, 0.0), 0.30), 3), 0.005,
                    help="Was der Vordermann pro Runde verliert, weil er "
                         "draussen bleibt.")
deg_neu = s2.slider("Abbau frischer Reifen [s/Runde]", 0.0, 0.30,
                    round(min(max(vor_neu, 0.0), 0.30), 3), 0.005,
                    help="Was der Verfolger auf dem neuen Satz verliert.")
strafe = s3.slider("Aufschlag Out-Lap [s]", 0.0, 2.0, 0.6, 0.1,
                   help="Der frische Reifen ist in der ersten Runde noch kalt. "
                        "Groessenordnung eine halbe bis eine Sekunde.")

runden = np.arange(1, 11)
gewinn = [f1lab.undercut_gain(deg_alt, deg_neu, int(n), strafe) for n in runden]
beste_runde, bester_gewinn = f1lab.optimal_undercut_window(
    deg_alt, deg_neu, max_laps=10, out_lap_penalty=strafe)

fig = go.Figure(go.Bar(
    x=runden, y=gewinn,
    marker={"color": [d.POSITIV if g > 0 else d.NEGATIV for g in gewinn],
            "line": {"color": d.BG, "width": 2}},
    hovertemplate="%{x} Runden frueher<br>%{y:+.2f} s<extra></extra>"))
fig.add_hline(y=0, line_color=d.MUTED, line_width=1)
if bester_gewinn > 0:
    fig.add_annotation(x=int(beste_runde), y=bester_gewinn, text="Optimum",
                       showarrow=True, arrowhead=2, ay=-34,
                       arrowcolor=d.FG, font={"color": d.FG})

zeige(fig, hoehe=400, showlegend=False,
      xaxis=achse("Runden frueher gestoppt", dtick=1),
      yaxis=achse("Sekunden, positiv heisst lohnt sich"))

k = st.columns(3)
k[0].metric("Bester Vorlauf", f"{beste_runde} Runden")
k[1].metric("Gewinn dabei", f"{bester_gewinn:+.2f} s")
if verlust is not None:
    k[2].metric("Zum Vergleich: Pitloss", f"{verlust:.2f} s",
                help="Der Pitloss kuerzt sich in der Rechnung heraus, weil "
                     "beide stoppen. Er steht hier nur als Groessenordnung "
                     "daneben.")

hinweis("Nicht modelliert ist Verkehr - und der entscheidet den Undercut in "
        "der Praxis oefter als die Reifen. Wer hinter einem langsameren Auto "
        "herauskommt, gibt den rechnerischen Vorteil in einer einzigen Runde "
        "wieder ab.")

# --- echte undercut-duelle (P42) --------------------------------------------
st.markdown("##### Wie oft gewinnt der Undercut wirklich?")
if not nur_rennen(auswahl, "Die Undercut-Duelle"):
    duelle = f1lab.undercut_duels(ses)
    if duelle.empty:
        st.info("Keine echten Undercut-Duelle in dieser Session gefunden - "
                "das braucht zwei Fahrer, die innerhalb weniger Runden "
                "hintereinander stoppen, waehrend einer den anderen direkt "
                "verfolgt.")
    else:
        n_duelle = len(duelle)
        erfolge = int(duelle["erfolg"].sum())
        k = st.columns(2)
        k[0].metric("Undercut-Duelle in dieser Session", n_duelle)
        k[1].metric("Davon erfolgreich", f"{erfolge}/{n_duelle}",
                    help="Erfolg heisst: der fruehere Stopper liegt zwei "
                         "gruene Runden nach dem spaeteren der beiden "
                         "Stopps vor seinem Rivalen.")
        tabelle(duelle.rename(columns={
            "driver": "Fahrer", "lap": "Stopp-Runde", "rival": "Rivale",
            "rival_lap": "Rivale stoppt Runde", "erfolg": "Erfolg"}))
    hinweis("Gegen den echten Rivalen gemessen (nicht gegen das ganze "
            "Feld, siehe P42) gewinnt der Undercut saison-weit (2024, 24 "
            "Rennen, 161 Duelle) nur 23.6% dieser direkten Duelle "
            "(95%-KI 17.3%-30.9%, p<0.0001 gegen 50%) - die Verteidigung "
            "gewinnt in diesen Daten gut drei von vier. Genau das ist der "
            "Verkehr-Effekt, den das Modell oben nicht sieht: wer frueher "
            "stoppt, muss sich erst durch den Rueckstand zurueckarbeiten, "
            "waehrend der Vordermann seine alte Reifenmischung gezielt "
            "gegen genau diesen einen Rivalen verteidigt.")

with st.expander("Wie die Rechnung aufgebaut ist"):
    st.markdown(
        "Fuer jede Runde im Fenster wird verglichen, was beide Fahrer "
        "verlieren:\n\n"
        "- Der **Vordermann** bleibt draussen und verliert `deg_alt` pro "
        "Runde, kumuliert.\n"
        "- Der **Verfolger** war schon an der Box, verliert `deg_neu` pro "
        "Runde und zusaetzlich einmalig den Aufschlag auf die Out-Lap.\n\n"
        "Die Summe der Differenzen ist der Gewinn. Dass die Kurve ab einem "
        "Punkt wieder faellt, liegt am frischen Reifen: auch er altert, und je "
        "frueher gestoppt wird, desto laenger muss er halten.")

if verlust is not None:
    with st.expander("Stopps in dieser Session"):
        laps = ses.laps
        stopps = (laps[laps["PitInTime"].notna()]
                  [["Driver", "Team", "LapNumber", "Compound", "TyreLife"]]
                  .rename(columns={"Driver": "Fahrer", "Team": "Team",
                                   "LapNumber": "Runde",
                                   "Compound": "Reifen bis dahin",
                                   "TyreLife": "Alter bei Stopp"})
                  .sort_values("Runde"))
        tabelle(stopps)
        st.caption("`Reifen bis dahin` ist die Mischung, mit der die In-Lap "
                   "gefahren wurde - nicht die, auf die gewechselt wird.")
