"""einstiegspunkt und router: baut die navigation, die startseite zeigt was
auf dem rechner liegt und was die app damit macht.

start mit::

    streamlit run app/Start.py

oder per doppelklick auf "2_Dashboard_starten.command".
"""
from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st
from common import (
    IDENT_ORDER,
    achse,
    hinweis,
    inventar,
    kein_cache_hinweis,
    konfiguriere_app,
    namensachse,
    setup,
    tabelle,
    zeige,
)

from f1lab import design as d

konfiguriere_app()


def start_seite() -> None:
    pfad = setup(
        "\N{CHEQUERED FLAG} Formel-1-Analyse",
        "Renndaten auswerten statt Code lesen. Gerechnet wird ausschliesslich in "
        "f1lab - derselbe Code, den auch die Skripte und die Tests verwenden.")

    if kein_cache_hinweis(pfad):
        st.stop()

    inv = inventar(str(pfad))
    geladen = inv[inv["timing"]]

    # --- eckdaten -----------------------------------------------------------
    k = st.columns(4)
    k[0].metric("Sessions gespeichert", f"{len(geladen):,}".replace(",", "."))
    k[1].metric("davon mit Telemetrie", int(geladen["telemetry"].sum()))
    k[2].metric("Saisons", geladen["season"].nunique())
    k[3].metric("Rennwochenenden", geladen.groupby(["season", "event"]).ngroups)

    # --- abdeckung ------------------------------------------------------------
    st.markdown("##### Was gespeichert ist")

    # die bestandsaufnahme kennt keine rundennummern, die reihenfolge im kalender
    # ergibt sie trotzdem eindeutig, weil testfahrten nicht im cache landen.
    wochenenden = (geladen.groupby(["season", "event", "event_date"])
                   .agg(sessions=("ident", "nunique"))
                   .reset_index())
    wochenenden["runde"] = (wochenenden.sort_values("event_date")
                            .groupby("season").cumcount() + 1)
    raster = wochenenden.pivot(index="season", columns="runde", values="sessions")
    raster = raster.reindex(sorted(raster.index))

    fig = go.Figure(go.Heatmap(
        z=raster.to_numpy(), x=raster.columns.tolist(),
        y=[str(s) for s in raster.index],
        colorscale=[[0, d.BG_HELL]] + [[(i + 1) / len(d.RAMPE), f]
                                       for i, f in enumerate(d.RAMPE)],
        hovertemplate="%{y} Runde %{x}<br>%{z} Sessions<extra></extra>",
        colorbar={"title": {"text": "Sessions", "side": "right"},
                  "tickfont": {"color": d.MUTED}, "outlinewidth": 0},
        xgap=2, ygap=2))
    zeige(fig, hoehe=110 + 34 * len(raster),
          xaxis=achse("Runde", dtick=2),
          yaxis=namensachse(autorange="reversed"))
    hinweis("Leere Felder sind Wochenenden, die noch nie geladen wurden - keine "
            "Ausfaelle, sondern Luecken im Warmup. Sieben Sessions sind das "
            "Maximum, konventionelle Wochenenden haben fuenf.")

    # --- verteilung -----------------------------------------------------------
    links, rechts = st.columns(2)

    with links:
        je_typ = (geladen.groupby("ident")
                  .agg(sessions=("ident", "size"), telemetrie=("telemetry", "sum"))
                  .reindex([i for i in IDENT_ORDER if i in set(geladen["ident"])]))
        fig = go.Figure([
            go.Bar(name="nur Rundendaten",
                   y=[d.IDENT_NAME.get(i, i) for i in je_typ.index],
                   x=je_typ["sessions"] - je_typ["telemetrie"],
                   orientation="h", marker={"color": d.SERIEN[0]}),
            go.Bar(name="mit Telemetrie",
                   y=[d.IDENT_NAME.get(i, i) for i in je_typ.index],
                   x=je_typ["telemetrie"], orientation="h",
                   marker={"color": d.SERIEN[1]}),
        ])
        st.markdown("##### Nach Session-Typ")
        zeige(fig, hoehe=340, barmode="stack",
              xaxis=achse("Sessions"), yaxis=namensachse())

    with rechts:
        je_saison = (geladen.groupby("season")
                     .agg(sessions=("ident", "size"),
                          telemetrie=("telemetry", "sum")).reset_index())
        fig = go.Figure([
            go.Bar(name="nur Rundendaten", x=je_saison["season"],
                   y=je_saison["sessions"] - je_saison["telemetrie"],
                   marker={"color": d.SERIEN[0]}),
            go.Bar(name="mit Telemetrie", x=je_saison["season"],
                   y=je_saison["telemetrie"], marker={"color": d.SERIEN[1]}),
        ])
        st.markdown("##### Nach Saison")
        zeige(fig, hoehe=340, barmode="stack",
              xaxis=achse(type="category"), yaxis=achse("Sessions"))

    hinweis("Telemetrie ist ein eigener, um ein Vielfaches groesserer Download. "
            "Ohne sie laufen Pace, Reifen und Strategie trotzdem - nur "
            "Streckenverlauf und Bremszonen brauchen sie zwingend.")

    # --- wegweiser ------------------------------------------------------------
    st.markdown("##### Die Seiten")
    st.markdown("""
| Seite | Was sie beantwortet | Quelle in f1lab |
| --- | --- | --- |
| **Uebersicht** | Wie lief die Session, und wie entwickelten sich die Rundenzeiten? | `clean_laps` |
| **Pace** | Wer war wirklich schnell, und ab wann ist der Unterschied belegbar? | `pace_table`, `bootstrap_median` |
| **Reifen** | Wie stark baut welche Mischung ab, und welchem Fit ist zu trauen? | `stints`, `degradation` |
| **Strategie** | Was kostet ein Stopp hier, und wann lohnt der Undercut? | `pit_loss`, `undercut_gain` |
| **Rennverlauf** | Wo lagen Safety Car und VSC, und was hat das verschoben? | `track_status_phases` |
| **Telemetrie** | Wo wird gebremst, wie hart, und wie sieht die Strecke aus? | `braking_zones`, `circuit_geometry` |
| **Kalender** | Wie sieht der Kalender aus, und wie vermisst man eine Strecke? | `event_dimension`, `circuit_dimension` |
| **Renndynamik** | Wer ueberholt wen, wie laeuft der Start, und wie stark bremst Verkehr das Feld? | `overtakes_matrix`, `start_performance`, `dirty_air_effect` |
| **Teamkollegen** | Wer gewinnt das teaminterne Duell ueber eine ganze Saison? | `elo_expected`, `elo_update` |
| **Strecke** | Wie sieht die Strecke aus, und hat sich ihr Layout ueber die Jahre veraendert? | `corner_labels`, `corner_speeds`, `circuit_geometry` |
| **Boxenstopps** | Welches Team boxt am schnellsten, ueber eine ganze Saison? | Ergast/jolpica (Boxengassen-Zeiten) |
| **Wetter** | Wie veraendern Regen und Streckentemperatur die Pace? | `weather_join`, `temperature_effect`, `wet_dry_classifier` |
| **Race Control** | Wann griff die Rennleitung ein, und wer wurde bestraft? | `track_status_phases`, `parse_penalties` |
| **Historie** | Wer kann noch Weltmeister werden, und wie hat sich F1 ueber 75 Jahre veraendert? | Ergast/jolpica (WM-Stand, historische Trends) |
| **Machine Learning** | Laesst sich Qualifying, Fahrstil oder ein Ausfall vorhersagen? | sklearn/torch-Modelle auf f1lab-Daten |
| **Strategie-Optimierer** | Was ist der rechnerisch beste Boxenstopp-Plan fuer dieses Rennen? | `optimal_strategy`, `SafetyCarProcess` |
| **Rundenzeit-Simulation** | Laesst sich eine Rundenzeit aus Streckengeometrie und Fahrzeugparametern berechnen? | `simulate_lap`, `calibrate_lap_model` |
| **Ueberholschwierigkeit** | Ist eine Strecke strukturell schwer zu ueberholen? | `overtakes_matrix`, `circuit_dimension` |
| **Startplatz-Paritaet** | Hat die "schmutzige Seite" der Startaufstellung einen echten Effekt? | `grid_lap1_positions` |
| **Verkehr-Simulation** | Wie stark aendert Verkehr eine Boxenstopp-Strategie wirklich? | `gap_evolution`, `traffic_cost` |
| **Streckenentwicklung** | Wird die Strecke von Q1 zu Q3 wirklich schneller? | `qualifying_track_evolution` |
| **Sprint vs. Rennen** | Wird im Sprint wirklich weniger ueberholt als im Rennen? | `overtakes_matrix` (Sprint vs. Rennen) |
""")

    with st.expander("Warum nur gespeicherte Daten?"):
        st.markdown(
            "FastF1 begrenzt sich auf 500 Abfragen pro Stunde. Wuerde die App "
            "fehlende Sessions selbst nachladen, waere die Bedienung von "
            "minutenlangen Wartezeiten und Zwangspausen durchsetzt - mitten im "
            "Klicken.\n\n"
            "Das Laden ist deshalb ein eigener Vorgang: der Warmup aus **p01** "
            "zieht ganze Saisons, faengt das Limit ab und laesst sich jederzeit "
            "abbrechen und fortsetzen. Die App liest nur, was dort angekommen "
            "ist. Sie laeuft damit auch ohne Netz und bietet nur Sessions an, die "
            "tatsaechlich auswertbar sind.")

    with st.expander("Bestand im Detail"):
        t = geladen.copy()
        t["Session"] = [d.IDENT_NAME.get(i, i) for i in t["ident"]]
        t["Telemetrie"] = t["telemetry"].map({True: "ja", False: "\N{EN DASH}"})
        tabelle(
            t[["season", "event_date", "event", "Session", "Telemetrie"]]
            .rename(columns={"season": "Saison", "event_date": "Datum",
                             "event": "Rennen"}),
            column_config={"Datum": st.column_config.DateColumn(
                format="DD.MM.YYYY")})


pg = st.navigation([
    st.Page(start_seite, title="Start", icon=":material/home:", default=True),
    st.Page("pages/1_Uebersicht.py", title="Uebersicht", icon=":material/analytics:"),
    st.Page("pages/2_Pace.py", title="Pace", icon=":material/speed:"),
    st.Page("pages/3_Reifen.py", title="Reifen", icon=":material/donut_large:"),
    st.Page("pages/4_Strategie.py", title="Strategie", icon=":material/route:"),
    st.Page("pages/5_Rennverlauf.py", title="Rennverlauf", icon=":material/timeline:"),
    st.Page("pages/6_Telemetrie.py", title="Telemetrie", icon=":material/sensors:"),
    st.Page("pages/7_Kalender.py", title="Kalender", icon=":material/calendar_month:"),
    st.Page("pages/8_Renndynamik.py", title="Renndynamik", icon=":material/bolt:"),
    st.Page("pages/9_Teamkollegen.py", title="Teamkollegen", icon=":material/groups:"),
    st.Page("pages/10_Strecke.py", title="Strecke", icon=":material/map:"),
    st.Page("pages/11_Boxenstopps.py", title="Boxenstopps", icon=":material/build:"),
    st.Page("pages/12_Wetter.py", title="Wetter", icon=":material/thunderstorm:"),
    st.Page("pages/13_RaceControl.py", title="Race Control", icon=":material/flag:"),
    st.Page("pages/14_Historie.py", title="Historie", icon=":material/history:"),
    st.Page("pages/15_MachineLearning.py", title="Machine Learning", icon=":material/psychology:"),
    st.Page("pages/16_Strategie_Optimierer.py", title="Strategie-Optimierer", icon=":material/tune:"),
    st.Page("pages/17_Rundenzeit_Simulation.py", title="Rundenzeit-Simulation", icon=":material/timer:"),
    st.Page("pages/18_Ueberholschwierigkeit.py", title="Ueberholschwierigkeit", icon=":material/compare_arrows:"),
    st.Page("pages/19_Startplatz_Paritaet.py", title="Startplatz-Paritaet", icon=":material/balance:"),
    st.Page("pages/20_Verkehr_Simulation.py", title="Verkehr-Simulation", icon=":material/traffic:"),
    st.Page("pages/21_Streckenentwicklung.py", title="Streckenentwicklung", icon=":material/trending_up:"),
    st.Page("pages/22_Sprint_vs_Rennen.py", title="Sprint vs. Rennen", icon=":material/sports_score:"),
], expanded=True)
pg.run()
