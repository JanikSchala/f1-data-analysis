"""safety-car-chronik und race-control-meldungen (strafen, track limits).

zwei reiter aus P18/P19. gerechnet wird nirgends hier. jede kennzahl kommt
aus f1lab.track_status_phases()/field_spread()/sc_compaction() bzw.
f1lab.parse_penalties()/parse_track_limits()/track_limit_crosscheck().
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
    nur_rennen,
    setup,
    sidebar_session,
    tabelle,
    zeige,
)

import f1lab
from f1lab import design as d

pfad = setup("Race Control", "Safety-Car-Phasen und Feldkompaktierung, dazu "
                             "Strafen und Track-Limit-Meldungen.")
if kein_cache_hinweis(pfad):
    st.stop()
assert pfad is not None  # kein_cache_hinweis() haette sonst schon abgebrochen

auswahl = sidebar_session(pfad)
assert auswahl is not None
ses = f1lab.load(auswahl.season, auswahl.event, auswahl.ident, messages=True)
st.subheader(auswahl.titel)

tab_sc, tab_rcm = st.tabs(["Safety-Car-Chronik", "Strafen & Track Limits"])

# ============================================================ safety-car
with tab_sc:
    if nur_rennen(auswahl, "Die Safety-Car-Chronik"):
        pass
    else:
        try:
            phasen = f1lab.track_status_phases(ses)
        except Exception:
            phasen = pd.DataFrame()

        if phasen.empty:
            st.info("Fuer diese Session liegen keine Flaggendaten vor.")
        else:
            stoerungen = phasen[phasen["label"] != "gruen"]
            neutral = stoerungen[stoerungen["label"].isin(["safety car", "vsc"])]

            k = st.columns(3)
            k[0].metric("Phasen ausser Gruen", len(stoerungen))
            k[1].metric("Neutralisationen", len(neutral))
            k[2].metric("Zeit unter Neutralisation",
                       f"{stoerungen['duration_s'].sum() / 60:.1f} min")

            st.markdown("##### Feldstreckung und Flaggenphasen")
            spread = f1lab.field_spread(ses)
            fig = go.Figure()
            for p in stoerungen.itertuples():
                fig.add_vrect(x0=p.lap_start,
                             x1=max(p.lap_end, p.lap_start + 0.3),
                             fillcolor=d.PHASE.get(p.label, d.MUTED),
                             opacity=0.35, line_width=0, layer="below")
            fig.add_trace(go.Scatter(x=spread.index, y=spread.to_numpy(),
                                     mode="lines", line={"color": d.FG, "width": 1.4},
                                     showlegend=False))
            zeige(fig, hoehe=380, xaxis=achse("Runde"),
                 yaxis=achse("Feldstreckung P1-Letzter [s]"))
            hinweis("Farbige Baender sind Flaggenphasen (gelb/SC/VSC/rot). Die "
                    "rohe Feldstreckung waehrend einer SC-Phase ist wegen des "
                    "Ausloese-Zwischenfalls oft hoeher als im Gruenschnitt - das "
                    "Feld kompaktiert sich erst im Verlauf der Phase (siehe "
                    "P18).")

            if not neutral.empty:
                st.markdown("##### Kompaktierung: Baseline gegen Minimum "
                           "waehrend der Neutralisation")
                komp = f1lab.sc_compaction(neutral, spread)
                if komp.empty:
                    st.info("Zu wenige gruene Runden vor den Neutralisationen "
                           "fuer eine Baseline.")
                else:
                    fig2 = go.Figure()
                    labels = [f"Runde {int(r.start)}-{int(r.ende)}"
                             for r in komp.itertuples()]
                    fig2.add_trace(go.Bar(x=labels, y=komp["baseline_s"],
                                         name="Baseline (3 gruene Runden davor)",
                                         marker={"color": d.MUTED}))
                    fig2.add_trace(go.Bar(x=labels, y=komp["minimum_s"],
                                         name="Minimum waehrend SC/VSC",
                                         marker={"color": d.SERIEN[1]}))
                    zeige(fig2, hoehe=360, barmode="group",
                         xaxis=namensachse(), yaxis=achse("Feldstreckung [s]"))
                    hinweis("Nicht der Mittelwert waehrend der Phase, sondern "
                            "das Minimum kurz vor dem Restart - der Mittelwert "
                            "waere vom Ausloese-Zwischenfall selbst verzerrt "
                            "(siehe P18).")
                    tabelle(komp.round(1).rename(columns={
                        "start": "Start", "ende": "Ende",
                        "baseline_s": "Baseline [s]", "minimum_s": "Minimum [s]",
                        "kompaktierung_pct": "Kompaktierung [%]"}))

            tabelle(stoerungen[["label", "lap_start", "lap_end", "duration_s"]]
                   .round(1).rename(columns={
                       "label": "Phase", "lap_start": "Start-Runde",
                       "lap_end": "End-Runde", "duration_s": "Dauer [s]"}))

            sek = f1lab.sc_deployment_sectors(ses)
            if not sek.empty:
                st.markdown("##### ZWEITE AUSBAUSTUFE: Feld-Sektorverteilung "
                           "bei SC-Deployment")
                hinweis("`Sector1/2/3SessionTime` (bislang ungenutzt) direkt "
                        "gegen `track_status['Time']` - beide session-relativ, "
                        "kein Telemetrie-Laden noetig (siehe P18).")
                tab_sek = (sek.dropna(subset=["sector"])
                          .groupby(["time", "sector"]).size().unstack(fill_value=0))
                fig3 = go.Figure()
                for i, col in enumerate(tab_sek.columns):
                    fig3.add_trace(go.Bar(
                        x=[f"Deployment {j + 1}" for j in range(len(tab_sek))],
                        y=tab_sek[col], name=f"Sektor {int(col)}",
                        marker={"color": d.SERIEN[i % len(d.SERIEN)]}))
                zeige(fig3, hoehe=340, barmode="group", xaxis=namensachse(),
                     yaxis=achse("Fahrer"))

# ==================================================================== RCM
with tab_rcm:
    try:
        rcm = ses.race_control_messages
    except fastf1.exceptions.DataNotLoadedError:
        # siehe 12_Wetter.py: FastF1 scheitert im Offline-Modus nicht beim
        # Laden, sondern erst beim Zugriff, wenn die Kategorie fuer diese
        # Session nicht im Cache liegt.
        rcm = None
    if rcm is None or rcm.empty:
        st.info("Fuer diese Session liegen keine Race-Control-Meldungen vor.")
    else:
        st.markdown("##### Meldungskategorien")
        kat = rcm["Category"].value_counts()
        fig = go.Figure(go.Bar(x=kat.index, y=kat.to_numpy(),
                               marker={"color": d.SERIEN[0]}))
        zeige(fig, hoehe=280, showlegend=False, xaxis=namensachse(),
             yaxis=achse("Meldungen"))

        pen = f1lab.parse_penalties(rcm)
        lim = f1lab.parse_track_limits(rcm)

        links, rechts = st.columns(2)
        with links:
            st.markdown(f"##### Strafen ({len(pen)})")
            if pen.empty:
                st.info("Keine Strafen erkannt.")
            else:
                je_fahrer = pen.groupby("driver").size().sort_values()
                fig2 = go.Figure(go.Bar(
                    x=je_fahrer.to_numpy(), y=je_fahrer.index, orientation="h",
                    marker={"color": d.SERIEN[1]}))
                zeige(fig2, hoehe=max(220, 28 * len(je_fahrer)), showlegend=False,
                     xaxis=achse("Anzahl Strafen"), yaxis=namensachse())
                tabelle(pen[["lap", "driver", "strafmass", "grund"]].rename(
                    columns={"lap": "Runde", "driver": "Fahrer",
                            "strafmass": "Strafmass", "grund": "Grund"}),
                       height=240)

        with rechts:
            st.markdown(f"##### Track Limits ({len(lim)})")
            if lim.empty:
                st.info("Keine Track-Limit-Meldungen erkannt.")
            else:
                je_kurve = lim.groupby("turn").size().sort_index()
                fig3 = go.Figure(go.Bar(
                    x=[f"T{t}" for t in je_kurve.index], y=je_kurve.to_numpy(),
                    marker={"color": d.SERIEN[0]}))
                zeige(fig3, hoehe=280, showlegend=False, xaxis=namensachse(),
                     yaxis=achse("Meldungen"))
                hinweis("Nicht dasselbe wie eine geloeschte Runde - erst die "
                        "Gegenpruefung unten zeigt, welche Meldungen tatsaechlich "
                        "eine Runde kosten (siehe P19).")

        if not lim.empty:
            st.markdown("##### Gegenpruefung mit Laps.Deleted")
            fehlend, deleted, n_mit_runde = f1lab.track_limit_crosscheck(ses, rcm)
            k = st.columns(3)
            k[0].metric("Meldungen mit eindeutiger Runde", n_mit_runde)
            k[1].metric("Davon als Deleted=True in den Laps", n_mit_runde - len(fehlend))
            k[2].metric("Ohne passende geloeschte Runde", len(fehlend))
            if not fehlend.empty:
                hinweis("FastF1s Deleted-Spalte ist fuer Track-Limit-"
                        "Auswertungen leicht unvollstaendig: Runden ohne eine "
                        "im Timing gefuehrte Rundenzeit (erste Runde, "
                        "Boxenrunden) koennen kein Deleted-Flag tragen, obwohl "
                        "die Meldung existiert (siehe P19).")
                tabelle(fehlend.rename(columns={"driver": "Fahrer",
                                                "runde": "Runde"}))

            st.markdown("##### ZWEITE AUSBAUSTUFE: umgekehrte Gegenpruefung "
                       "(DeletedReason)")
            hinweis("`Laps.DeletedReason` stand lange nur als "
                    "Docstring-Behauptung, nie gelesen. Umgekehrte Richtung: "
                    "findet jede per DeletedReason als Track-Limits erkannte "
                    "Runde auch eine passende Text-Meldung? (siehe P19).")
            umgekehrt = f1lab.deleted_reason_crosscheck(ses, rcm)
            if umgekehrt.empty:
                st.success("Alle DeletedReason-Faelle finden eine passende "
                         "Meldung - keine Regex-Luecke in diese Richtung.",
                         icon="\N{WHITE HEAVY CHECK MARK}")
            else:
                st.warning(f"{len(umgekehrt)} DeletedReason-Faelle ohne "
                         "passende Text-Meldung:")
                tabelle(umgekehrt.rename(columns={"driver": "Fahrer",
                                                  "turn": "Kurve",
                                                  "runde": "Runde"}))

        st.markdown("##### ZWEITE AUSBAUSTUFE: Blaue Flaggen")
        hinweis("`Flag`/`RacingNumber` (bislang ungenutzt) direkt statt "
                "Regex auf den Meldungstext - eine blaue Flagge ist kein "
                "Vergehen, sondern die Aufforderung, einen Ueberrunder "
                "durchzulassen (siehe P19).")
        bf = f1lab.blue_flags(ses, rcm)
        if bf.empty:
            st.info("Keine blauen Flaggen in dieser Session.")
        else:
            je_fahrer = bf.groupby("driver").size().sort_values()
            fig4 = go.Figure(go.Bar(
                x=je_fahrer.to_numpy(), y=je_fahrer.index, orientation="h",
                marker={"color": d.SERIEN[2]}))
            zeige(fig4, hoehe=max(220, 24 * len(je_fahrer)), showlegend=False,
                 xaxis=achse("Blaue Flaggen"), yaxis=namensachse())
