"""Stints und Reifendegradation.

Die Steigung je Stint ist mit herausgerechnetem Treibstoffeffekt geschaetzt.
Ohne diese Korrektur wird die Degradation systematisch unterschaetzt: das Auto
wird leichter, waehrend der Reifen abbaut, und beide Effekte heben sich
teilweise auf.
"""
from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st
from common import (
    achse,
    hinweis,
    kein_cache_hinweis,
    kopfzeile,
    lade,
    namensachse,
    setup,
    sidebar_session,
    tabelle,
    zeige,
)

import f1lab
from f1lab import design as d

pfad = setup("Reifen", "Wer faehrt welche Mischung wie lange - und was kostet "
                       "sie pro Runde?")
if kein_cache_hinweis(pfad):
    st.stop()

auswahl = sidebar_session(pfad)
ses = lade(auswahl)
kopfzeile(ses, auswahl)

with st.sidebar:
    st.header("Filter")
    schwelle = st.slider(
        "Ausreisser-Schwelle", 1.02, 1.25, 1.10, 0.01,
        help="Fuer Degradation bewusst weiter gefasst als bei der Pace: der "
             "Abbau am Stint-Ende soll drinbleiben, sonst schaetzt der Fit "
             "ihn weg.")
    min_runden = st.slider("Mindestrunden je Stint", 4, 20, 6, 1)

# --- Stints ---------------------------------------------------------------
st.markdown("##### Reifenwechsel je Fahrer")

stints = f1lab.stints(ses)
if stints.empty:
    st.info("Fuer diese Session liegen keine Stint-Daten vor.")
    st.stop()

fig = go.Figure()
gezeigt = set()
for row in stints.itertuples():
    mischung = str(row.Compound).upper()
    fig.add_trace(go.Bar(
        y=[row.Driver], x=[row.laps], base=[row.start - 1], orientation="h",
        name=mischung, legendgroup=mischung,
        showlegend=mischung not in gezeigt,
        marker={"color": d.COMPOUND.get(mischung, d.MUTED),
                "line": {"color": d.BG, "width": 2}},
        hovertemplate=f"<b>{row.Driver}</b><br>{mischung.title()}<br>"
                      f"Runde {row.start}\N{EN DASH}{row.end} "
                      f"({row.laps} Runden)<extra></extra>"))
    gezeigt.add(mischung)

zeige(fig, hoehe=max(360, 24 * stints["Driver"].nunique()), barmode="stack",
      xaxis=achse("Runde"), yaxis=namensachse())
hinweis("Ein Balken je Stint, eingefaerbt nach Mischung. Jeder Wechsel ist "
        "ein Boxenstopp. Auffaellig kurze Stints am Anfang sind meist "
        "Schadensbegrenzung nach einem Zwischenfall, nicht Strategie.")

# --- Degradation je Stint -------------------------------------------------
st.markdown("##### Reifenabbau je Stint")

deg = f1lab.degradation(ses, threshold=schwelle, min_laps=min_runden)
if deg.empty:
    st.info("Kein Stint hat genug saubere Runden fuer eine Schaetzung. "
            "Mindestrunden senken oder Schwelle anheben.")
    st.stop()

belastbar = deg[deg["reliable"]]
k = st.columns(3)
k[0].metric("Geschaetzte Stints", len(deg))
k[1].metric("Davon belastbar", len(belastbar),
            help="Plausibilitaetspruefung aus f1lab: unplausible Steigungen "
                 "und zu schwache Fits fallen raus.")
if not belastbar.empty:
    schlimmster = belastbar.iloc[-1]
    k[2].metric("Staerkster Abbau", f"{schlimmster['deg_s_per_lap']:.3f} s/Runde",
                f"{schlimmster['driver']} auf "
                f"{str(schlimmster['compound']).title()}")

fig = go.Figure()
for mischung, teil in deg.groupby("compound"):
    name = str(mischung).upper()
    fig.add_trace(go.Scatter(
        x=teil["laps"], y=teil["deg_s_per_lap"], mode="markers", name=name,
        customdata=teil[["driver", "stint", "r2"]],
        marker={"size": 13, "color": d.COMPOUND.get(name, d.MUTED),
                "line": {"width": 1.5, "color": d.BG},
                "symbol": ["circle" if r else "x" for r in teil["reliable"]]},
        hovertemplate="<b>%{customdata[0]}</b> Stint %{customdata[1]}<br>"
                      "%{y:.3f} s pro Runde<br>%{x} Runden, "
                      "R2 %{customdata[2]:.2f}<extra></extra>"))

fig.add_hline(y=0, line_color=d.MUTED, line_width=1, line_dash="dot")
zeige(fig, xaxis=achse("Ausgewertete Runden im Stint"),
      yaxis=achse("Sekunden pro Runde Reifenalter"))
hinweis("Kreuze sind Fits, die die Plausibilitaetspruefung nicht bestanden "
        "haben - meist zu wenige Runden oder ein Stint, in dem Verkehr die "
        "Rundenzeit staerker bestimmt hat als der Reifen. Werte unter null "
        "heissen nicht, dass der Reifen besser wird: dort ist der Fit schlicht "
        "nicht identifiziert.")

# --- Je Mischung ----------------------------------------------------------
je_mischung = f1lab.degradation_by_compound(ses, threshold=schwelle,
                                            min_laps=min_runden)
links, rechts = st.columns([1, 1])

with links:
    st.markdown("##### Mittel je Mischung")
    if je_mischung.empty:
        st.info("Kein belastbarer Fit uebrig.")
    else:
        fig = go.Figure(go.Bar(
            x=[str(c).title() for c in je_mischung.index],
            y=je_mischung["mean"],
            marker={"color": [d.COMPOUND.get(str(c).upper(), d.MUTED)
                              for c in je_mischung.index],
                    "line": {"color": d.BG, "width": 2}},
            error_y={"type": "data", "array": je_mischung["std"].fillna(0),
                     "color": d.MUTED, "thickness": 1.4},
            customdata=je_mischung[["stints"]],
            hovertemplate="%{x}<br>%{y:.3f} s pro Runde<br>"
                          "%{customdata[0]} Stints<extra></extra>"))
        zeige(fig, hoehe=380, showlegend=False,
              xaxis=namensachse(), yaxis=achse("s pro Runde"))
        hinweis("Nur belastbare Fits. Die Fehlerbalken sind "
                "Standardabweichungen zwischen den Stints, keine "
                "Konfidenzintervalle - bei zwei oder drei Stints je Mischung "
                "sagen sie wenig.")

with rechts:
    st.markdown("##### Alle Fits")
    tabelle(deg.rename(columns={
        "driver": "Fahrer", "team": "Team", "stint": "Stint",
        "compound": "Mischung", "fresh": "frischer Reifen", "laps": "Runden",
        "deg_s_per_lap": "Abbau [s/Runde]", "base_s": "Basiszeit [s]",
        "r2": "R2", "reliable": "belastbar"}), height=380)

# --- Frisch vs. wiederverwendet (P13 ZWEITE AUSBAUSTUFE) ------------------
st.markdown("##### Frischer gegen wiederverwendeter Reifen")
hinweis("`FreshTyre` stand lange nur als Docstring-Behauptung, nie als "
        "gelesene Spalte - jetzt Teil von `f1lab.degradation()`. Nur "
        "belastbare Fits, nur Mischungen mit Faellen in beiden Gruppen "
        "(siehe P13).")

rel = belastbar
tab_frisch = rel.groupby(["compound", "fresh"])["deg_s_per_lap"].agg(
    ["mean", "count"])
mischungen = [c for c in tab_frisch.index.get_level_values(0).unique()
             if {True, False} <= set(tab_frisch.loc[c].index)]
if not mischungen:
    st.info("Keine Mischung mit belastbaren Fits in beiden Gruppen in "
           "dieser Session.")
else:
    fig = go.Figure()
    for fresh, label, farbe in ((False, "wiederverwendet", d.MUTED),
                                (True, "frisch", d.SERIEN[0])):
        werte, ns = [], []
        for c in mischungen:
            if (c, fresh) in tab_frisch.index:
                werte.append(tab_frisch.loc[(c, fresh), "mean"])
                ns.append(int(tab_frisch.loc[(c, fresh), "count"]))
            else:
                werte.append(None)
                ns.append(0)
        fig.add_trace(go.Bar(
            x=[m.title() for m in mischungen], y=werte, name=label,
            marker={"color": farbe},
            customdata=ns, hovertemplate=f"{label}<br>" + "%{x}: %{y:.3f} "
                          "s/Runde<br>n=%{customdata}<extra></extra>"))
    zeige(fig, hoehe=360, barmode="group", xaxis=namensachse(),
         yaxis=achse("Mittlere Degradation [s/Runde]"))
    hinweis("Ein wiederverwendeter Satz hat Vorverschleiss aus einer "
            "frueheren Session - dass er hier trotzdem nicht automatisch "
            "staerker abbaut, ist eher ein Strategie-Auswahleffekt "
            "(wiederverwendete Saetze werden oft konservativer gefahren) "
            "als ein Reifeneffekt (siehe P13).")

with st.expander("Warum die Treibstoffkorrektur noetig ist"):
    st.markdown(
        "Ein Auto verliert ueber die Renndistanz rund 100 kg Sprit und wird "
        "dadurch kontinuierlich schneller - Groessenordnung 0,03 Sekunden je "
        "Kilogramm. Ueber einen Stint von 20 Runden ist das gut eine Sekunde, "
        "die der Reifen scheinbar gutmacht.\n\n"
        "`fuel_correct()` normiert alle Rundenzeiten auf leeren Tank, bevor "
        "die Steigung geschaetzt wird. Die verwendeten Faustwerte "
        "(1,8 kg pro Runde, 0,03 s pro kg) sind Literaturwerte - die "
        "Groessenordnung stimmt, exakte Messwerte sind es nicht.")

# --- Cliff-Erkennung (P13 AUSBAUSTUFE) -------------------------------------
st.markdown("##### Bricht der Reifen irgendwo ein? (Cliff-Erkennung)")
hinweis("Reifen degradieren nicht immer linear - ab einem Knickpunkt kann der "
        "Abbau ploetzlich steiler werden. f1lab.find_cliff() probiert jeden "
        "moeglichen Bruchpunkt durch und akzeptiert ihn nur, wenn die zweite "
        "Steigung die erste deutlich uebertrifft (siehe P13).")

MIN_RUNDEN_CLIFF = 10
laps_sauber = f1lab.clean_laps(ses, threshold=schwelle).copy()
laps_sauber["sec"] = laps_sauber["LapTime"].dt.total_seconds()
laps_sauber["corrected"] = f1lab.fuel_correct(
    laps_sauber["sec"], laps_sauber["LapNumber"], ses.total_laps)

cliffs = []
kandidaten = []
for (drv, stint), g in laps_sauber.groupby(["Driver", "Stint"]):
    g = g.sort_values("TyreLife")
    if len(g) < MIN_RUNDEN_CLIFF:
        continue
    kandidaten.append((drv, stint))
    knick, links, rechts = f1lab.find_cliff(g["TyreLife"], g["corrected"])
    if knick is not None:
        cliffs.append({"driver": drv, "stint": int(stint), "laps": g,
                       "knick": knick, "vor": links.slope, "nach": rechts.slope})

if not kandidaten:
    st.info(f"Kein Stint mit mindestens {MIN_RUNDEN_CLIFF} sauberen Runden in "
           "dieser Session.")
elif not cliffs:
    st.success(f"Kein Cliff erkannt: alle {len(kandidaten)} langen Stints "
              "(>= 10 Runden) sind linear genug fuer einen einfachen Fit.",
              icon="\N{WHITE HEAVY CHECK MARK}")
else:
    k = st.columns(2)
    k[0].metric("Stints mit erkanntem Cliff", f"{len(cliffs)}/{len(kandidaten)}")
    staerkster = max(cliffs, key=lambda c: c["nach"] - c["vor"])
    k[1].metric("Deutlichster Fall",
               f"{staerkster['driver']} Stint {staerkster['stint']}",
               f"{staerkster['vor']:+.2f} -> {staerkster['nach']:+.2f} s/Runde")

    wahl = st.selectbox(
        "Stint anzeigen", cliffs,
        format_func=lambda c: f"{c['driver']} Stint {c['stint']} "
                              f"(Knick bei Reifenalter {c['knick']})",
        index=cliffs.index(staerkster))

    g = wahl["laps"]
    vor = g[g["TyreLife"] <= wahl["knick"]]
    nach = g[g["TyreLife"] >= wahl["knick"]]
    b_vor = vor["corrected"].mean() - wahl["vor"] * vor["TyreLife"].mean()
    b_nach = nach["corrected"].mean() - wahl["nach"] * nach["TyreLife"].mean()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=g["TyreLife"], y=g["corrected"], mode="markers",
                             marker={"color": d.MUTED, "size": 8},
                             name="Runden"))
    pb = g[g["IsPersonalBest"]]
    if not pb.empty:
        fig.add_trace(go.Scatter(
            x=pb["TyreLife"], y=pb["corrected"], mode="markers",
            marker={"symbol": "star", "size": 14, "color": "rgba(0,0,0,0)",
                   "line": {"color": d.FG, "width": 1.5}},
            name="Personal Best"))
    xv = [vor["TyreLife"].min(), wahl["knick"]]
    xn = [wahl["knick"], nach["TyreLife"].max()]
    fig.add_trace(go.Scatter(x=xv, y=[wahl["vor"] * x + b_vor for x in xv],
                             line={"color": d.SERIEN[0], "width": 3},
                             name=f"vor Knick: {wahl['vor']:+.2f} s/Runde"))
    fig.add_trace(go.Scatter(x=xn, y=[wahl["nach"] * x + b_nach for x in xn],
                             line={"color": d.SERIEN[1], "width": 3},
                             name=f"nach Knick: {wahl['nach']:+.2f} s/Runde"))
    fig.add_vline(x=wahl["knick"], line_color=d.MUTED, line_dash="dot")
    zeige(fig, hoehe=380, xaxis=achse("Reifenalter [Runden]"),
         yaxis=achse("Fuel-korrigierte Rundenzeit [s]"))
