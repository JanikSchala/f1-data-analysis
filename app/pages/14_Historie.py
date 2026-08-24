"""WM-stand-simulator (fahrer und konstrukteure) fuer die laufende saison,
und 75 jahre historische trends.

drei reiter aus P21/P45/P22. alle brauchen eine echte netzwerkverbindung zu
Ergast/jolpica, wie 11_Boxenstopps.py. die beiden WM-simulatoren sind an die
laufende saison gebunden (die frage "wer kann noch weltmeister werden"
ergibt nur an einer offenen saison sinn) und sind deshalb nicht in der
seitenleiste waehlbar, nur stundenweise gecacht (standings aendern sich
rennweise). die 75-jahre-trends sind dagegen eine feste historische spanne
und liegen dauerhaft im diskcache.
"""
from __future__ import annotations

import time

import fastf1
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from common import achse, hinweis, namensachse, setup, zeige
from fastf1.ergast import Ergast

from f1lab import design as d

YEAR = pd.Timestamp.now().year
PTS_RENNEN = {1: 25, 2: 18, 3: 15, 4: 12, 5: 10, 6: 8, 7: 6, 8: 4, 9: 2, 10: 1}
PTS_SPRINT = {1: 8, 2: 7, 3: 6, 4: 5, 5: 4, 6: 3, 7: 2, 8: 1}
N_SIM = 50_000

ERSTE_KONSTRUKTEURS_WM = 1958
DNF_AB = 1970
STRECKEN = {"monza": "Monza", "spa": "Spa-Francorchamps",
           "silverstone": "Silverstone"}
SCHRITT_STRECKEN = 4

setup("Historie", "WM-Stand-Simulator fuer die laufende Saison, und 75 "
                  "Jahre F1 aus Ergast/jolpica - braucht Netzwerk.")

tab_wm, tab_konstrukteure, tab_trends = st.tabs(
    ["Fahrer-WM-Simulator", "Konstrukteurs-WM-Simulator", "75-Jahre-Trends"])


def _mit_wiederholung(fn, *args, versuche: int = 5, **kwargs):
    for i in range(versuche):
        try:
            return fn(*args, **kwargs)
        except Exception:
            if i == versuche - 1:
                raise
            time.sleep(3 * (i + 1))
    return None


def _punkte_array(tabelle: dict) -> np.ndarray:
    arr = np.zeros(25)
    for platz, pkt in tabelle.items():
        arr[platz] = pkt
    return arr


def _monte_carlo(base_points, drivers, pos_hist, sprint_hist, sprint_flags,
                 dnf_rate, rng) -> pd.DataFrame:
    pts_r, pts_s = _punkte_array(PTS_RENNEN), _punkte_array(PTS_SPRINT)
    totals = {dr: np.full(N_SIM, base_points.get(dr, 0.0)) for dr in drivers}
    for ist_sprint in sprint_flags:
        for dr in drivers:
            hist = pos_hist.get(dr)
            if hist is not None and len(hist) > 0:
                pos = np.clip(rng.choice(hist, size=N_SIM).astype(int), 0, 24)
                pts = pts_r[pos]
                if dnf_rate is not None:
                    ausfall = rng.random(N_SIM) < dnf_rate.get(dr, 0.0)
                    pts = np.where(ausfall, 0.0, pts)
                totals[dr] += pts
            if ist_sprint:
                shist = sprint_hist.get(dr)
                if shist is not None and len(shist) > 0:
                    pos = np.clip(rng.choice(shist, size=N_SIM).astype(int), 0, 24)
                    pts = pts_s[pos]
                    if dnf_rate is not None:
                        ausfall = rng.random(N_SIM) < dnf_rate.get(dr, 0.0)
                        pts = np.where(ausfall, 0.0, pts)
                    totals[dr] += pts
    mat = np.vstack([totals[dr] for dr in drivers])
    return pd.DataFrame(mat.T, columns=drivers)


def _monte_carlo_team(base_points, teams, pos_hist, sprint_hist, sprint_flags,
                      dnf_rate, rng) -> pd.DataFrame:
    """wie _monte_carlo(), aber zwei unabhaengige Positionszuege je Team und
    Event (ein Konstrukteur bringt zwei Autos, beide Punkte zaehlen -
    siehe P45)."""
    pts_r, pts_s = _punkte_array(PTS_RENNEN), _punkte_array(PTS_SPRINT)
    totals = {t: np.full(N_SIM, base_points.get(t, 0.0)) for t in teams}
    for ist_sprint in sprint_flags:
        for t in teams:
            hist = pos_hist.get(t)
            if hist is not None and len(hist) > 0:
                for _auto in range(2):
                    pos = np.clip(rng.choice(hist, size=N_SIM).astype(int), 0, 24)
                    pts = pts_r[pos]
                    if dnf_rate is not None:
                        ausfall = rng.random(N_SIM) < dnf_rate.get(t, 0.0)
                        pts = np.where(ausfall, 0.0, pts)
                    totals[t] += pts
            if ist_sprint:
                shist = sprint_hist.get(t)
                if shist is not None and len(shist) > 0:
                    for _auto in range(2):
                        pos = np.clip(rng.choice(shist, size=N_SIM).astype(int), 0, 24)
                        pts = pts_s[pos]
                        if dnf_rate is not None:
                            ausfall = rng.random(N_SIM) < dnf_rate.get(t, 0.0)
                            pts = np.where(ausfall, 0.0, pts)
                        totals[t] += pts
    mat = np.vstack([totals[t] for t in teams])
    return pd.DataFrame(mat.T, columns=teams)


@st.cache_data(ttl=3600, show_spinner=False)
def _wm_daten(jahr: int):
    erg = Ergast(result_type="pandas", auto_cast=True)
    standings = _mit_wiederholung(erg.get_driver_standings, season=jahr).content[0]
    sched = _mit_wiederholung(erg.get_race_schedule, season=jahr)
    remaining = fastf1.get_events_remaining(include_testing=False)
    sprint_flags = remaining["EventFormat"].str.contains(
        "sprint", case=False).tolist()
    gefahrene_runden = max(0, len(sched) - len(remaining))

    race_frames, sprint_frames = [], []
    for r in range(1, gefahrene_runden + 1):
        res = _mit_wiederholung(erg.get_race_results, season=jahr, round=r).content
        if res:
            race_frames.append(res[0])
        time.sleep(0.2)
        sp = _mit_wiederholung(erg.get_sprint_results, season=jahr, round=r).content
        if sp:
            sprint_frames.append(sp[0])
        time.sleep(0.2)
    races = (pd.concat(race_frames, ignore_index=True) if race_frames
             else pd.DataFrame())
    sprints = (pd.concat(sprint_frames, ignore_index=True) if sprint_frames
              else pd.DataFrame())
    if not races.empty:
        races["dnf"] = ~races["status"].isin(["Finished", "Lapped"])
    return standings, sprint_flags, races, sprints


@st.cache_data(ttl=3600, show_spinner=False)
def _konstrukteurs_standings(jahr: int) -> pd.DataFrame:
    erg = Ergast(result_type="pandas", auto_cast=True)
    return _mit_wiederholung(erg.get_constructor_standings, season=jahr).content[0]


with tab_wm:
    try:
        with st.spinner(f"WM-Stand {YEAR} wird von Ergast/jolpica geladen "
                       "(gebremst wegen Rate-Limit) ..."):
            standings, sprint_flags, races, sprints = _wm_daten(YEAR)
    except Exception as exc:
        st.error(f"Ergast/jolpica war nicht erreichbar: {exc}")
        standings, races = pd.DataFrame(), pd.DataFrame()

    if standings.empty or races.empty:
        st.info(f"Fuer {YEAR} liegen noch keine oder keine auswertbaren "
               "Ergebnisse vor.")
    else:
        n_races, n_sprints = len(sprint_flags), sum(sprint_flags)
        max_points = n_races * 25 + n_sprints * 8
        leader = standings["points"].max()
        standings = standings.copy()
        standings["moeglich"] = standings["points"] + max_points >= leader

        k = st.columns(4)
        k[0].metric("Rennen verbleibend", n_races)
        k[1].metric("Davon Sprints", n_sprints)
        k[2].metric("Max. moegliche Punkte", max_points)
        k[3].metric("Rechnerisch noch im Rennen",
                   f"{int(standings['moeglich'].sum())}/{len(standings)}")

        st.markdown(f"##### Fahrer-WM {YEAR}, aktueller Stand")
        top = standings.head(10).iloc[::-1]
        farben = [d.SERIEN[0] if i == len(top) - 1 else d.MUTED
                 for i in range(len(top))]
        fig = go.Figure(go.Bar(x=top["points"], y=top["familyName"],
                               orientation="h", marker={"color": farben}))
        zeige(fig, hoehe=380, showlegend=False, xaxis=achse("Punkte"),
             yaxis=namensachse())

        base_points = standings.set_index("driverId")["points"].to_dict()
        name_map = standings.set_index("driverId")["familyName"].to_dict()
        drivers = list(base_points.keys())

        pos_hist_alle = (races.groupby("driverId")["position"]
                         .apply(lambda s: s.dropna().to_numpy()).to_dict())
        pos_hist_ohne_dnf = (races[~races["dnf"]].groupby("driverId")["position"]
                             .apply(lambda s: s.dropna().to_numpy()).to_dict())
        dnf_rate = races.groupby("driverId")["dnf"].mean().to_dict()
        sprint_hist = (sprints.groupby("driverId")["position"]
                      .apply(lambda s: s.dropna().to_numpy()).to_dict()
                      if not sprints.empty else {})

        rng = np.random.default_rng(7)
        sim_basis = _monte_carlo(base_points, drivers, pos_hist_alle, sprint_hist,
                                 sprint_flags, None, rng)
        chance = pd.Series(
            {name_map[dr]: (sim_basis.idxmax(axis=1) == dr).mean() * 100
             for dr in drivers}).sort_values(ascending=False)

        st.markdown("##### Titelchance (Monte-Carlo, historische Positions-"
                   "Stichprobe)")
        top_chance = chance[chance > 0.5].head(8)
        if top_chance.empty:
            st.info("Kein Fahrer mit nennenswerter Restchance.")
        else:
            farben2 = [d.SERIEN[0] if i == 0 else d.MUTED
                      for i in range(len(top_chance))]
            fig2 = go.Figure(go.Bar(x=top_chance.index, y=top_chance.to_numpy(),
                                    marker={"color": farben2},
                                    text=[f"{v:.1f}%" for v in top_chance],
                                    textposition="outside"))
            zeige(fig2, hoehe=360, showlegend=False, xaxis=namensachse(),
                 yaxis=achse("Titelchance [%]", range=[0, 108]))
            hinweis(f"n={N_SIM:,} Simulationen: fuer jedes verbleibende Rennen/"
                    "Sprint eine Position aus der bisherigen Saison-Verteilung "
                    "je Fahrer gezogen. Der rechnerische Test oben ist bei "
                    f"{max_points} moeglichen Punkten oft fuer fast das ganze "
                    "Feld wahr - erst die Simulation zeigt die reale Konzentration "
                    "auf wenige Fahrer (siehe P21).".replace(",", "."))

        st.markdown("##### AUSBAUSTUFE: DNF-Quote und ihre Wirkung auf die "
                   "Podiumschance")
        dnf_s = pd.Series({name_map[dr]: r * 100 for dr, r in dnf_rate.items()
                          if r > 0}).sort_values()
        linke, rechte = st.columns(2)
        with linke:
            if dnf_s.empty:
                st.info("Keine Ausfaelle in der bisherigen Saison.")
            else:
                farben3 = [d.SERIEN[1] if v == dnf_s.max() else d.MUTED
                          for v in dnf_s.to_numpy()]
                fig3 = go.Figure(go.Bar(x=dnf_s.to_numpy(), y=dnf_s.index,
                                        orientation="h", marker={"color": farben3}))
                zeige(fig3, hoehe=max(300, 26 * len(dnf_s)), showlegend=False,
                     xaxis=achse("Ausfallquote bisherige Saison [%]"),
                     yaxis=namensachse())

        with rechte:
            rng2 = np.random.default_rng(7)
            sim_dnf = _monte_carlo(base_points, drivers, pos_hist_ohne_dnf,
                                   sprint_hist, sprint_flags, dnf_rate, rng2)

            def top3_quote(sim: pd.DataFrame) -> pd.Series:
                rang = (-sim).rank(axis=1, method="first")
                return pd.Series({name_map[dr]: (rang[dr] <= 3).mean() * 100
                                  for dr in drivers})

            vgl = pd.DataFrame({"ohne_dnf": top3_quote(sim_basis),
                               "mit_dnf": top3_quote(sim_dnf)})
            vgl = vgl[(vgl["ohne_dnf"] > 0.1) | (vgl["mit_dnf"] > 0.1)]
            vgl = vgl.sort_values("mit_dnf", ascending=False).head(8)
            if vgl.empty:
                st.info("Zu wenige Faelle mit nennenswerter Top-3-Chance.")
            else:
                fig4 = go.Figure()
                fig4.add_trace(go.Bar(x=vgl.index, y=vgl["ohne_dnf"],
                                      name="ohne DNF-Modell",
                                      marker={"color": d.MUTED}))
                fig4.add_trace(go.Bar(x=vgl.index, y=vgl["mit_dnf"],
                                      name="mit DNF-Modell",
                                      marker={"color": d.SERIEN[1]}))
                zeige(fig4, hoehe=max(300, 26 * len(dnf_s)), barmode="group",
                     xaxis=namensachse(), yaxis=achse("P(Top 3 Endstand) [%]"))
        hinweis("Das Punktesystem saettigt ausserhalb der Top 10 bei 0 - ob "
                "eine schlechte Runde ein Ausfall oder ein regulaerer 15. "
                "Platz war, steckt in der reinen Positions-Stichprobe schon "
                "drin. Die Verschiebung durchs DNF-Modell bleibt deshalb klein "
                "(siehe P21).")

# ========================================================= konstrukteurs-wm
with tab_konstrukteure:
    try:
        with st.spinner(f"Konstrukteurs-Stand {YEAR} wird von Ergast/jolpica "
                       "geladen (gebremst wegen Rate-Limit) ..."):
            # _wm_daten ist st.cache_data-gecacht - ein zweiter Aufruf hier
            # kostet kein zusaetzliches Netzwerk, races/sprints/sprint_flags
            # sind je Auto schon genau das, was auch die Konstrukteure
            # brauchen (nur anders gruppiert, siehe P45).
            _standings_f, sprint_flags_t, races_t, sprints_t = _wm_daten(YEAR)
            standings_t = _konstrukteurs_standings(YEAR)
    except Exception as exc:
        st.error(f"Ergast/jolpica war nicht erreichbar: {exc}")
        standings_t, races_t = pd.DataFrame(), pd.DataFrame()

    if standings_t.empty or races_t.empty:
        st.info(f"Fuer {YEAR} liegen noch keine oder keine auswertbaren "
               "Ergebnisse vor.")
    else:
        n_races_t, n_sprints_t = len(sprint_flags_t), sum(sprint_flags_t)
        # ein team bringt zwei autos an den start - platz 1+2 statt nur 1.
        max_points_t = n_races_t * (25 + 18) + n_sprints_t * (8 + 7)
        leader_t = standings_t["points"].max()
        standings_t = standings_t.copy()
        standings_t["moeglich"] = standings_t["points"] + max_points_t >= leader_t

        k = st.columns(4)
        k[0].metric("Rennen verbleibend", n_races_t)
        k[1].metric("Davon Sprints", n_sprints_t)
        k[2].metric("Max. moegliche Punkte", max_points_t)
        k[3].metric("Rechnerisch noch im Rennen",
                   f"{int(standings_t['moeglich'].sum())}/{len(standings_t)}")

        st.markdown(f"##### Konstrukteurs-WM {YEAR}, aktueller Stand")
        top_t = standings_t.head(10).iloc[::-1]
        farben_t = [d.SERIEN[0] if i == len(top_t) - 1 else d.MUTED
                   for i in range(len(top_t))]
        fig_t = go.Figure(go.Bar(x=top_t["points"], y=top_t["constructorName"],
                                 orientation="h", marker={"color": farben_t}))
        zeige(fig_t, hoehe=380, showlegend=False, xaxis=achse("Punkte"),
             yaxis=namensachse())

        base_points_t = standings_t.set_index("constructorId")["points"].to_dict()
        name_map_t = standings_t.set_index("constructorId")["constructorName"].to_dict()
        teams_t = list(base_points_t.keys())

        pos_hist_alle_t = (races_t.groupby("constructorId")["position"]
                          .apply(lambda s: s.dropna().to_numpy()).to_dict())
        pos_hist_ohne_dnf_t = (
            races_t[~races_t["dnf"]].groupby("constructorId")["position"]
            .apply(lambda s: s.dropna().to_numpy()).to_dict())
        dnf_rate_t = races_t.groupby("constructorId")["dnf"].mean().to_dict()
        sprint_hist_t = (sprints_t.groupby("constructorId")["position"]
                        .apply(lambda s: s.dropna().to_numpy()).to_dict()
                        if not sprints_t.empty else {})

        rng_t = np.random.default_rng(7)
        sim_basis_t = _monte_carlo_team(base_points_t, teams_t, pos_hist_alle_t,
                                        sprint_hist_t, sprint_flags_t, None, rng_t)
        chance_t = pd.Series(
            {name_map_t[t]: (sim_basis_t.idxmax(axis=1) == t).mean() * 100
             for t in teams_t}).sort_values(ascending=False)

        st.markdown("##### Titelchance (Monte-Carlo, zwei Autos je Team und "
                   "Event)")
        top_chance_t = chance_t[chance_t > 0.5].head(8)
        if top_chance_t.empty:
            st.info("Kein Team mit nennenswerter Restchance.")
        else:
            farben2_t = [d.SERIEN[0] if i == 0 else d.MUTED
                        for i in range(len(top_chance_t))]
            fig2_t = go.Figure(go.Bar(
                x=top_chance_t.index, y=top_chance_t.to_numpy(),
                marker={"color": farben2_t},
                text=[f"{v:.1f}%" for v in top_chance_t], textposition="outside"))
            zeige(fig2_t, hoehe=360, showlegend=False, xaxis=namensachse(),
                 yaxis=achse("Titelchance [%]", range=[0, 108]))
            hinweis("Positions-Stichprobe je Team ist ueber beide Autos "
                    "gepoolt, pro Event werden zwei unabhaengige Positionen "
                    "gezogen - Team-interne Korrelation (Teamorder, "
                    "gemeinsamer Bauteilausfall) ist damit nicht modelliert "
                    "(siehe P45).")

        st.markdown("##### Ausfallquote je Konstrukteur (beide Autos gepoolt)")
        dnf_s_t = pd.Series({name_map_t[t]: r * 100 for t, r in dnf_rate_t.items()
                            if r > 0}).sort_values()
        if dnf_s_t.empty:
            st.info("Keine Ausfaelle in der bisherigen Saison.")
        else:
            farben3_t = [d.SERIEN[1] if v == dnf_s_t.max() else d.MUTED
                       for v in dnf_s_t.to_numpy()]
            fig3_t = go.Figure(go.Bar(x=dnf_s_t.to_numpy(), y=dnf_s_t.index,
                                      orientation="h", marker={"color": farben3_t}))
            zeige(fig3_t, hoehe=max(300, 26 * len(dnf_s_t)), showlegend=False,
                 xaxis=achse("Ausfallquote je Auto, bisherige Saison [%]"),
                 yaxis=namensachse())

# ============================================================= 75-jahre-trends
@st.cache_data(persist="disk", show_spinner=False)
def _dominanz_und_nationen() -> pd.DataFrame:
    erg = Ergast(result_type="pandas", auto_cast=True)
    rows = []
    for year in range(ERSTE_KONSTRUKTEURS_WM, 2025):
        cs = _mit_wiederholung(erg.get_constructor_standings, season=year)
        if cs is None or not cs.content or cs.content[0].empty:
            continue
        cs = cs.content[0]
        total = cs["points"].sum()
        if total == 0:
            continue
        nat_anteil = cs["constructorNationality"].value_counts(
            normalize=True).iloc[0]
        rows.append({
            "season": year, "champion": cs.iloc[0]["constructorName"],
            "anteil": cs.iloc[0]["points"] / total, "teams": len(cs),
            "haeufigste_nation": cs["constructorNationality"].mode().iloc[0],
            "nation_anteil": nat_anteil,
        })
        time.sleep(0.15)
    return pd.DataFrame(rows)


@st.cache_data(persist="disk", show_spinner=False)
def _dnf_rate() -> pd.DataFrame:
    erg = Ergast(result_type="pandas", auto_cast=True)
    rows = []
    for year in range(DNF_AB, 2025):
        s = _mit_wiederholung(erg.get_finishing_status, season=year)
        if s is None or s.empty:
            continue
        total = s["count"].sum()
        finished = s.loc[s["status"].str.contains(
            "Finished|Lap", na=False), "count"].sum()
        rows.append({"season": year, "dnf_rate": 1 - finished / total})
        time.sleep(0.15)
    return pd.DataFrame(rows)


@st.cache_data(persist="disk", show_spinner=False)
def _kalendergroesse() -> pd.DataFrame:
    erg = Ergast(result_type="pandas", auto_cast=True)
    rows = []
    for year in range(ERSTE_KONSTRUKTEURS_WM, 2025):
        sched = _mit_wiederholung(erg.get_race_schedule, season=year)
        if sched is None or sched.empty:
            continue
        rows.append({"season": year, "rennen": len(sched),
                    "laender": sched["country"].nunique()})
        time.sleep(0.15)
    return pd.DataFrame(rows)


@st.cache_data(persist="disk", show_spinner=False)
def _rundenzeit_je_strecke() -> pd.DataFrame:
    erg = Ergast(result_type="pandas", auto_cast=True)
    rows = []
    for circuit_id, name in STRECKEN.items():
        for year in range(1950, 2025, SCHRITT_STRECKEN):
            res = _mit_wiederholung(erg.get_race_results, season=year,
                                    circuit=circuit_id)
            if res is None or not res.content or res.content[0].empty:
                continue
            df = res.content[0]
            sieger = df[df["position"] == 1]
            if sieger.empty:
                continue
            sieger = sieger.iloc[0]
            if pd.isna(sieger["totalRaceTime"]) or not sieger["laps"]:
                continue
            sekunden = sieger["totalRaceTime"].total_seconds() / sieger["laps"]
            if sekunden < 30:
                continue
            rows.append({"strecke": name, "season": year, "rundenzeit_s": sekunden})
            time.sleep(0.15)
    return pd.DataFrame(rows)


with tab_trends:
    try:
        with st.spinner("75 Jahre F1-Historie werden von Ergast/jolpica "
                       "geladen (nur beim allerersten Aufruf - danach liegt "
                       "es dauerhaft im Diskcache, das dauert mehrere "
                       "Minuten) ..."):
            dom = _dominanz_und_nationen()
            dnf = _dnf_rate()
            kal = _kalendergroesse()
    except Exception as exc:
        st.error(f"Ergast/jolpica war nicht erreichbar: {exc}")
        dom = pd.DataFrame()

    if dom.empty:
        st.info("Keine historischen Daten geladen.")
    else:
        oben1, oben2 = st.columns(2)
        with oben1:
            st.markdown("##### Dominanzphasen: Konstrukteurs-WM")
            fig = go.Figure(go.Bar(x=dom["season"], y=dom["anteil"] * 100,
                                   marker={"color": d.SERIEN[0]}))
            zeige(fig, hoehe=340, showlegend=False, xaxis=achse("Saison"),
                 yaxis=achse("Punkteanteil Champion [%]"))
        with oben2:
            st.markdown("##### Nationen: Konzentration im Konstrukteursfeld")
            fig2 = go.Figure(go.Scatter(x=dom["season"], y=dom["nation_anteil"] * 100,
                                        mode="lines",
                                        line={"color": d.SERIEN[2], "width": 1.8}))
            zeige(fig2, hoehe=340, showlegend=False, xaxis=achse("Saison"),
                 yaxis=achse("Anteil haeufigste Nation [%]"))

        unten1, unten2 = st.columns(2)
        with unten1:
            st.markdown("##### Zuverlaessigkeit ueber die Zeit")
            if dnf.empty:
                st.info("Keine DNF-Daten geladen.")
            else:
                fig3 = go.Figure(go.Scatter(x=dnf["season"], y=dnf["dnf_rate"] * 100,
                                            mode="lines",
                                            line={"color": d.SERIEN[1], "width": 1.8}))
                zeige(fig3, hoehe=340, showlegend=False, xaxis=achse("Saison"),
                     yaxis=achse("DNF-Rate [%]"))
        with unten2:
            st.markdown("##### Streckenwandel: Kalendergroesse")
            if kal.empty:
                st.info("Keine Kalenderdaten geladen.")
            else:
                fig4 = go.Figure()
                fig4.add_trace(go.Scatter(x=kal["season"], y=kal["rennen"],
                                          mode="lines", name="Rennen",
                                          line={"color": d.SERIEN[0], "width": 1.8}))
                fig4.add_trace(go.Scatter(x=kal["season"], y=kal["laender"],
                                          mode="lines", name="Laender",
                                          line={"color": d.MUTED, "width": 1.4,
                                               "dash": "dot"}))
                zeige(fig4, hoehe=340, xaxis=achse("Saison"), yaxis=achse("Anzahl"))

        hinweis("Punkteanteil, DNF-Rate und Nationenkonzentration stammen aus "
                "denselben get_constructor_standings-Aufrufen; Ergasts "
                "totalRaceTime ist fuer sehr alte Rennen teils fehlerhaft, "
                "hier aber ohne Wirkung (siehe P22).")

        st.markdown("##### AUSBAUSTUFE: Rundenzeit auf angeblich unveraenderten "
                   "Strecken")
        rz = _rundenzeit_je_strecke()
        if rz.empty:
            st.info("Keine Rundenzeit-Daten geladen.")
        else:
            fig5 = go.Figure()
            for i, (strecke, g) in enumerate(rz.groupby("strecke")):
                g = g.sort_values("season")
                fig5.add_trace(go.Scatter(
                    x=g["season"], y=g["rundenzeit_s"], mode="lines+markers",
                    name=strecke, line={"color": d.SERIEN[i % len(d.SERIEN)]}))
            fig5.add_vline(x=1979, line_color=d.MUTED, line_dash="dot")
            fig5.add_annotation(x=1979, y=1, yref="paper", showarrow=False,
                               text="Spa fehlt 1971-82,<br>kehrt verkuerzt zurueck",
                               font={"color": d.MUTED, "size": 10},
                               align="left", xanchor="left", yanchor="top")
            zeige(fig5, hoehe=400, xaxis=achse("Saison"),
                 yaxis=achse("Rundenzeit Sieger [s]"))
            hinweis("Monza, Spa und Silverstone sind KEINE 75 Jahre lang "
                    "unveraendert geblieben - Spas Sprung 1983 ist eine auf "
                    "weniger als die Haelfte gekuerzte Strecke, keine "
                    "40 Jahre Geschwindigkeitsgewinn. totalRaceTime/laps ist "
                    "ausserdem die Renn-Durchschnittszeit, nicht die reine "
                    "Pace - Safety-Car- und Regenrennen ziehen sie nach oben "
                    "(siehe P22).")
