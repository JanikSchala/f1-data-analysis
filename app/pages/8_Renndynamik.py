"""positionsverlauf, ueberholungen, start und verfolgung ueber ein rennen.

vier reiter, alle auf derselben session: wer gewinnt/verliert wo positionen
(P20), wer gewinnt die ersten meter (P31), was ein enger vordermann eine
rundenzeit kostet (P32), und wo auf der strecke ueberholungen passieren,
in einer DRS-zone oder nicht (P39). gerechnet wird nirgends hier, jede
kennzahl kommt aus f1lab.
"""
from __future__ import annotations

import fastf1
import fastf1.plotting as f1plt
import numpy as np
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
    nur_rennen,
    setup,
    sidebar_session,
    tabelle,
    zeige,
)

import f1lab
from f1lab import design as d

LINESTYLE = {"-": "solid", "--": "dash", ":": "dot", "-.": "dashdot"}

pfad = setup("Renndynamik", "Positionen, Ueberholungen, Start und "
                           "Verfolgung ueber ein Rennen.")
if kein_cache_hinweis(pfad):
    st.stop()
assert pfad is not None  # kein_cache_hinweis() haette sonst schon abgebrochen

auswahl = sidebar_session(pfad)
assert auswahl is not None
if auswahl is None:
    st.stop()
if nur_rennen(auswahl, "Renndynamik"):
    st.stop()

ses = lade(auswahl, telemetrie=auswahl.telemetrie)
kopfzeile(ses, auswahl)

tab_position, tab_start, tab_verfolgung, tab_drs = st.tabs(
    ["Position & Ueberholungen", "Start", "Verfolgung (Dirty Air)",
     "Ueberholorte (DRS)"])


# ================================================== position & ueberholungen
with tab_position:
    pos = f1lab.position_progression(ses)
    if pos.empty:
        st.info("Keine Positionsdaten in dieser Session.")
    else:
        st.markdown("##### Positionsverlauf")
        fig = go.Figure()
        for drv in pos.columns:
            style = f1plt.get_driver_style(drv, ["color", "linestyle"],
                                           session=ses)
            fig.add_trace(go.Scatter(
                x=pos.index, y=pos[drv], mode="lines", name=drv,
                line={"color": style["color"], "width": 2,
                     "dash": LINESTYLE.get(style["linestyle"], "solid")},
                hovertemplate=f"<b>{drv}</b><br>Runde %{{x}}<br>"
                              "Position %{y}<extra></extra>"))
        zeige(fig, hoehe=560,
             xaxis=achse("Runde"),
             yaxis=achse("Position", autorange="reversed", dtick=2))

        st.markdown("##### Ueberholungen gesamt")
        with st.spinner("Ueberholmatrix wird berechnet ..."):
            mat = f1lab.overtakes_matrix(ses)
        if mat.empty:
            st.info("Keine auswertbaren Ueberholungen.")
        else:
            gesamt = mat.sum(axis=1).sort_values()
            gesamt = gesamt[gesamt > 0]
            fig = go.Figure(go.Bar(x=gesamt, y=gesamt.index, orientation="h",
                                   marker={"color": d.SERIEN[0]}))
            zeige(fig, hoehe=max(320, 26 * len(gesamt)), showlegend=False,
                 xaxis=achse("Ueberholungen"), yaxis=namensachse())
            hinweis("Ohne Boxenstopp-Effekt und nur auf gruener Flagge - "
                    "ein Safety-Car-Restart oder ein Boxenstopp verschiebt "
                    "Positionen ohne echtes Duell auf der Strecke.")

            stacked = mat.stack()
            stacked = stacked[stacked > 0].sort_values(ascending=False)
            if not stacked.empty:
                top = stacked.head(15).reset_index()
                top.columns = ["Ueberholer", "Ueberholter", "Anzahl"]
                with st.expander("Haeufigste Duelle"):
                    tabelle(top)


# ================================================================== start
with tab_start:
    if not auswahl.telemetrie:
        st.info("Fuer diese Session liegt keine Telemetrie vor - der Start "
               "braucht sie (Geschwindigkeit ueber die ersten Sekunden).")
    else:
        with st.spinner("Startphase wird fuer alle Fahrer ausgewertet ..."):
            sp = f1lab.start_performance(ses)
        if sp.empty:
            st.info("Keine auswertbaren Starts (nur Boxenstarts, oder keine "
                   "Telemetrie fuer Runde 1).")
        else:
            d_plot = sp.dropna(subset=["m_nach_5s", "gewinn"])
            fig = go.Figure(go.Scatter(
                x=d_plot["m_nach_5s"], y=d_plot["gewinn"], mode="markers+text",
                text=d_plot["driver"], textposition="top center",
                textfont={"color": d.FG, "size": 10},
                marker={"color": d.SERIEN[0], "size": 10}))
            fig.add_hline(y=0, line={"color": d.MUTED, "width": 0.8})
            zeige(fig, hoehe=560, showlegend=False,
                 xaxis=achse("Distanz nach 5 s [m]"),
                 yaxis=achse("Positionsgewinn Runde 1"))
            hinweis("Boxenstarts sind ausgeschlossen - komplett andere "
                    "Ausgangsgeschwindigkeit als ein Ampelstart. Wer weit "
                    "rechts UND deutlich im Minus liegt, hat einen guten "
                    "Start verloren (meist Kurve-1-Chaos, nicht die eigene "
                    "Beschleunigung).")
            with st.expander("Tabelle"):
                tabelle(sp.rename(columns={
                    "driver": "Fahrer", "grid": "Start", "ende_r1": "Ende R1",
                    "gewinn": "Gewinn", "t_100": "bis 100 km/h [s]",
                    "t_200": "bis 200 km/h [s]", "m_nach_5s": "Distanz 5s [m]"}))


# ======================================================= verfolgung (Dirty Air)
with tab_verfolgung:
    if not auswahl.telemetrie:
        st.info("Fuer diese Session liegt keine Telemetrie vor - die "
               "Verfolgungsanalyse braucht sie (DistanceToDriverAhead).")
    else:
        fahrer = sorted(ses.laps["Driver"].dropna().unique())
        schnellster = str(ses.laps.loc[ses.laps["LapTime"].idxmin(), "Driver"]) \
            if ses.laps["LapTime"].notna().any() else fahrer[0]
        wahl = st.selectbox("Fahrer", fahrer,
                            index=fahrer.index(schnellster) if schnellster in fahrer
                            else 0, key="dirty_air_fahrer")

        with st.spinner(f"Telemetrie von {wahl} wird ausgewertet ..."):
            roh = f1lab.close_following(ses, wahl)
        if roh.empty:
            st.info(f"Keine auswertbaren gruenen Runden fuer {wahl}.")
        else:
            slope, inter, r2, dd = f1lab.dirty_air_effect(roh)
            if dd.empty or len(dd) < 5:
                st.info("Zu wenige Runden fuer eine Regression.")
            else:
                k = st.columns(2)
                k[0].metric("Dirty-Air-Effekt",
                           f"{slope:+.4f} s je 1% Zeit unter 50 m")
                k[1].metric("R²", f"{r2:.3f}",
                           help="Anteil der Streuung, den der Abstand "
                                "erklaert - niedrig heisst: der Effekt ist "
                                "schwach gegenueber der uebrigen Streuung "
                                "einer Rundenzeit.")

                xs = np.linspace(dd["anteil_nah"].min(), dd["anteil_nah"].max(), 50)
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=dd["anteil_nah"], y=dd["sec_corr"], mode="markers",
                    marker={"color": dd["tyre_life"], "colorscale": d.RAMPE_SCALE,
                           "size": 9, "colorbar": {"title": "Reifenalter"}},
                    hovertemplate="Runde %{customdata}<br>%{x:.0f}% unter "
                                 "50m<br>%{y:.2f}s<extra></extra>",
                    customdata=dd["lap"]))
                fig.add_trace(go.Scatter(
                    x=xs, y=slope * xs + inter, mode="lines",
                    line={"color": d.SERIEN[1], "width": 2.5},
                    showlegend=False))
                zeige(fig, hoehe=480, showlegend=False,
                     xaxis=achse("Zeitanteil unter 50 m Abstand [%]"),
                     yaxis=achse("Treibstoff- und degradationsbereinigte "
                                "Rundenzeit [s]"))
                hinweis("Treibstoff- (P04) und degradationsbereinigt (P13) - "
                        "sonst misst die Regression den Sprit- oder "
                        "Reifeneffekt statt Dirty Air, siehe P32.")


# ============================================================ ueberholorte (DRS)
with tab_drs:
    if not auswahl.telemetrie:
        st.info("Fuer diese Session liegt keine Telemetrie vor - der Ort "
               "eines Ueberholvorgangs braucht sie (DriverAhead-Kanal).")
    else:
        @st.cache_data(show_spinner="Ueberholorte werden bestimmt (eigene "
                                    "Telemetrie je Ereignis, dauert einen "
                                    "Moment) ...")
        def _ueberholorte(cache_pfad: str, season: int, event: str, ident: str):
            s = f1lab.load(season, event, ident, telemetry=True)
            events = f1lab.overtake_events(s)
            try:
                # overtake_locations() laedt intern selbst eine Qualifying-
                # Referenzsession fuer die DRS-Zonen - liegt deren Telemetrie
                # nicht im Cache, scheitert das erst beim Zugriff (siehe
                # 12_Wetter.py/13_RaceControl.py), nicht schon beim Laden.
                # KeyError zusaetzlich zu DataNotLoadedError: FastF1 wirft das,
                # wenn ein einzelner Fahrer im Cache Laufzeitdaten, aber keine
                # car_data hat (session.car_data[DriverNumber] schlaegt fehl) -
                # dieselbe Kategorie Cache-Luecke, nur ein anderer Fehlertyp.
                orte = f1lab.overtake_locations(s)
                q = f1lab.load(season, int(s.event["RoundNumber"]), "Q",
                               telemetry=True)
                lap_q = q.laps.pick_fastest()
                zonen = f1lab.drs_zones(q, str(lap_q["Driver"]))
            except (fastf1.exceptions.DataNotLoadedError, KeyError):
                orte = pd.DataFrame(columns=["gainer", "loser", "lap",
                                             "distance_m", "in_drs_zone"])
                zonen = pd.DataFrame()
            ref_lap = s.laps.pick_fastest()
            tel = ref_lap.get_telemetry().add_distance()
            bremszonen = f1lab.driver_braking_zones(s, str(ref_lap["Driver"]))
            return events, orte, zonen, tel, bremszonen

        events, orte, zonen, tel, bremszonen = _ueberholorte(
            str(pfad), auswahl.season, auswahl.event, auswahl.ident)

        if events.empty:
            st.info("Keine auswertbaren Ueberholungen fuer diese Session.")
        elif orte.empty:
            st.info("Keine der Ueberholungen liess sich in der Telemetrie "
                   "eindeutig lokalisieren.")
        else:
            k = st.columns(3)
            k[0].metric("Lokalisiert", f"{len(orte)}/{len(events)}",
                       help=f"{100 * len(orte) / len(events):.0f}% "
                            "Abdeckung - der Rest hatte im entscheidenden "
                            "Moment keinen sauberen DriverAhead-Treffer "
                            "(siehe P39).")
            k[1].metric("Davon in DRS-Zone",
                       f"{100 * orte['in_drs_zone'].mean():.0f}%")
            k[2].metric("DRS-Zonen", len(zonen),
                       help="Aus der Qualifying-Session desselben Events - "
                            "DRS im Rennen braucht einen Rueckstand unter "
                            "1s, die schnellste Rennrunde hat deshalb oft "
                            "gar keine offene Zone (siehe P39).")

            x = (tel["X"] / 10).to_numpy()
            y = (tel["Y"] / 10).to_numpy()
            dist = tel["Distance"].to_numpy()

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=x, y=y, mode="lines",
                                     line={"color": d.GRID, "width": 8},
                                     hoverinfo="skip", showlegend=False))
            for _, z in zonen.iterrows():
                maske = (dist >= z["start_m"]) & (dist <= z["end_m"])
                fig.add_trace(go.Scatter(
                    x=x[maske], y=y[maske], mode="lines",
                    line={"color": d.SERIEN[0], "width": 8},
                    hoverinfo="skip", showlegend=False))

            idx = [int(np.argmin(np.abs(dist - v))) for v in orte["distance_m"]]
            farben = [d.POSITIV if v else d.SERIEN[1]
                     for v in orte["in_drs_zone"]]
            text = [f"{g} ueberholt {v} (Runde {int(lp)})" for g, v, lp
                   in zip(orte["gainer"], orte["loser"], orte["lap"])]
            fig.add_trace(go.Scatter(
                x=x[idx], y=y[idx], mode="markers",
                marker={"color": farben, "size": 10,
                       "line": {"color": d.FG, "width": 0.5}},
                text=text, hovertemplate="%{text}<extra></extra>",
                showlegend=False))
            zeige(fig, hoehe=560, xaxis={"visible": False},
                 yaxis={"visible": False, "scaleanchor": "x", "scaleratio": 1})
            hinweis("Blau eingefaerbte Streckenabschnitte sind die "
                   "DRS-Zonen. Punktfarbe: gruen ueberholt in einer Zone, "
                   "orange ausserhalb. Nur lokalisierte Ueberholungen sind "
                   "eingezeichnet (siehe Abdeckung oben) - kein Zufalls-"
                   "Bias in eine bestimmte Streckenhaelfte, aber die Zahl "
                   "ist keine vollstaendige Zaehlung (siehe P39).")

            with st.expander("Tabelle"):
                tabelle(orte.rename(columns={
                    "gainer": "Ueberholer", "loser": "Ueberholter",
                    "lap": "Runde", "distance_m": "Distanz [m]",
                    "in_drs_zone": "DRS-Zone"}))

            if not bremszonen.empty:
                st.markdown("##### Dritte AUSBAUSTUFE: Abstand zur naechsten "
                           "Bremszone")
                strecke_m = float(dist.max())
                vorlauf_echt = f1lab.lead_distance_to_zone(
                    orte["distance_m"], bremszonen["start_m"], strecke_m)
                rng = np.random.default_rng(42)
                zufall = rng.uniform(0, strecke_m, 20_000)
                vorlauf_zufall = f1lab.lead_distance_to_zone(
                    zufall, bremszonen["start_m"], strecke_m)

                fig2 = go.Figure()
                fig2.add_trace(go.Histogram(
                    x=vorlauf_zufall, histnorm="probability density",
                    name="Zufalls-Baseline", marker={"color": d.GRID},
                    opacity=0.7, nbinsx=25))
                fig2.add_trace(go.Histogram(
                    x=vorlauf_echt, histnorm="probability density",
                    name="Echte Ueberholorte", marker={"color": d.SERIEN[0]},
                    opacity=0.7, nbinsx=25))
                zeige(fig2, hoehe=360, barmode="overlay",
                     xaxis=achse("Abstand zur naechsten Bremszone [m]"),
                     yaxis=achse("Dichte"))
                hinweis(f"Median Abstand: echt {np.median(vorlauf_echt):.0f}m "
                       f"gegen Zufall {np.median(vorlauf_zufall):.0f}m. "
                       f"Anteil unter 200m: echt "
                       f"{100 * (vorlauf_echt < 200).mean():.0f}% gegen "
                       f"Zufall {100 * (vorlauf_zufall < 200).mean():.0f}% - "
                       "echte Ueberholungen passieren NICHT ueberdurch"
                       "schnittlich direkt vor dem Bremspunkt, aber "
                       "zuverlaessig irgendwo auf der Geraden davor (unter "
                       "800m: siehe P39, dritte AUSBAUSTUFE).")

    st.markdown("##### Vierte AUSBAUSTUFE: DRS-Anteil vor und nach dem "
               "Ground-Effect-Umbruch")
    st.caption("Feste Saisons 2018/2024 (die einzigen mit vollstaendiger "
              "Telemetrie im Cache), unabhaengig von der oben gewaehlten "
              "Session.")

    @st.cache_data(persist="disk", show_spinner="DRS-Anteil 2018 und 2024 "
                                                "wird verglichen (erster "
                                                "Aufruf dauert einige "
                                                "Minuten) ...")
    def _drs_saison_scan(cache_pfad: str, saison: int) -> pd.DataFrame:
        inv_lokal = f1lab.cached_sessions(cache_pfad)
        tel_rennen = sorted(inv_lokal[(inv_lokal["season"] == saison)
                                      & (inv_lokal["ident"] == "R")
                                      & inv_lokal["telemetry"]]["event"].unique())
        zeilen = []
        for gp in tel_rennen:
            try:
                s = f1lab.load(saison, gp, "R", telemetry=True)
                ev = f1lab.overtake_events(s)
                if ev.empty:
                    continue
                o = f1lab.overtake_locations(s)
                if o.empty:
                    continue
            except Exception:
                continue
            zeilen.append({"gp": gp, "lokalisiert": len(o),
                           "drs_pct": round(100 * o["in_drs_zone"].mean(), 1)})
        return pd.DataFrame(zeilen)

    erg_2018 = _drs_saison_scan(str(pfad), 2018)
    erg_2024 = _drs_saison_scan(str(pfad), 2024)

    if erg_2018.empty or erg_2024.empty:
        st.info("Fuer 2018 oder 2024 liegt keine telemetriefaehige Session "
               "im Cache.")
    else:
        n_2018 = int((erg_2018["lokalisiert"] * erg_2018["drs_pct"] / 100)
                    .round().sum())
        n_2024 = int((erg_2024["lokalisiert"] * erg_2024["drs_pct"] / 100)
                    .round().sum())
        kd = st.columns(2)
        kd[0].metric("DRS-Anteil 2018 (gepoolt)",
                    f"{100 * n_2018 / erg_2018['lokalisiert'].sum():.1f}%",
                    help=f"{n_2018}/{erg_2018['lokalisiert'].sum()} "
                         f"lokalisierte Ueberholungen, {len(erg_2018)} "
                         "Strecken")
        kd[1].metric("DRS-Anteil 2024 (gepoolt)",
                    f"{100 * n_2024 / erg_2024['lokalisiert'].sum():.1f}%",
                    help=f"{n_2024}/{erg_2024['lokalisiert'].sum()} "
                         f"lokalisierte Ueberholungen, {len(erg_2024)} "
                         "Strecken")

        fig3 = go.Figure()
        fig3.add_trace(go.Box(y=erg_2018["drs_pct"], name="2018",
                              marker={"color": d.SERIEN[0]}, boxpoints="all"))
        fig3.add_trace(go.Box(y=erg_2024["drs_pct"], name="2024",
                              marker={"color": d.SERIEN[1]}, boxpoints="all"))
        zeige(fig3, hoehe=380, showlegend=False,
             yaxis=achse("DRS-Anteil je Strecke [%]"))
        hinweis("Die 2022er Regeln sollten engere Rennen 'auch ohne DRS' "
               "ermoeglichen - in diesen Daten ist das Gegenteil der Fall: "
               "der DRS-Anteil an Ueberholungen ist 2024 deutlich hoeher als "
               "2018, nicht niedriger. Bei nur zwei Saisons mit Telemetrie "
               "im Cache ist das eine Momentaufnahme zweier Regel-Epochen, "
               "kein belastbarer Trend (siehe P39, vierte AUSBAUSTUFE).")
