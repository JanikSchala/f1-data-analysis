"""Live-Timing (P30): waehrend einer echten laufenden F1-Session aufzeichnen
und die rollierende Pace live mitverfolgen.

anders als jede andere seite hier bewusst NICHT im offline-modus: eine
live-aufzeichnung fuer ein ganz neues event/jahr, das lokal noch nirgends
im cache steht, braucht das saisonkalender fuer die reine sitzungsaufloesung
(fastf1.get_session()) - im offline-modus schlaegt das bei einem frischen,
leeren cache fehl (echt gegengeprueft), waehrend jede andere seite bewusst
NIE selbst nachlaedt (siehe common.setup()). live-timing ist per definition
netzwerkabhaengig, dieselbe ausnahme wie 11_Boxenstopps.py fuer Ergast, nur
fuer FastF1s eigenen cache.

die aufzeichnung selbst laeuft als eigener subprocess
(python -m fastf1.livetiming save ..., siehe P30-skript), weil nur ein
subprocess sich zuverlaessig per terminate() stoppen laesst - ein
blockierender SignalRClient.start() in einem thread nicht. die live-ansicht
liest die wachsende aufzeichnungsdatei periodisch neu (st.fragment) und
wertet sie ueber dieselbe rolling_pace()/aktuelle_rangliste()-logik wie das
skript aus.
"""
from __future__ import annotations

import subprocess
import sys
from datetime import datetime
from pathlib import Path

import fastf1
import plotly.graph_objects as go
import streamlit as st
from common import achse, namensachse, setup, zeige
from fastf1.livetiming.data import LiveTimingData

import f1lab
from f1lab import design as d

OUT = Path(__file__).resolve().parents[2] / "12_live_timing" / "out"
OUT.mkdir(exist_ok=True)
IDENTS = ["R", "Q", "S", "SQ", "FP1", "FP2", "FP3"]


def _p30():
    """dasselbe muster wie make_assets.py._skript_importieren(): das
    P30-skript dynamisch importieren statt seine rolling_pace()/
    aktuelle_rangliste()-logik hier zu kopieren."""
    pfad = (Path(__file__).resolve().parents[2] / "12_live_timing"
           / "p30_live_timing_aufzeichnen_und_in_echtzeit_auswerte.py")
    import importlib.util
    spec = importlib.util.spec_from_file_location(pfad.stem, pfad)
    assert spec is not None and spec.loader is not None
    modul = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modul)
    return modul


setup("Live-Timing", "Zeichnet den echten FastF1-Live-Timing-Stream auf, "
                     "waehrend eine F1-Session laeuft (siehe P30). Bewusst "
                     "online statt offline: eine brandneue Session braucht "
                     "den Saisonkalender, den keine andere Seite hier je "
                     "selbst nachlaedt.")
# setup() aktiviert offline-modus wie jede andere seite - hier bewusst
# wieder aufgehoben, siehe modul-docstring oben.
f1lab.enable_cache(offline=False)

if "live_proc" not in st.session_state:
    st.session_state.live_proc = None
    st.session_state.live_pfad = None
    st.session_state.live_meta = None

with st.sidebar:
    st.header("Aufzeichnung")
    jahr = st.number_input("Jahr", min_value=2018, max_value=2100,
                           value=datetime.now().year, step=1)
    event = st.text_input("Event", value="", placeholder="z.B. Italy")
    ident = st.selectbox("Session", IDENTS)
    timeout_min = st.number_input("Timeout [Minuten]", min_value=5,
                                  max_value=360, value=180, step=5,
                                  help="SignalRClient beendet sich selbst, "
                                       "wenn so lange keine neue Nachricht "
                                       "ankommt - nicht die Renndauer.")

laeuft = (st.session_state.live_proc is not None
         and st.session_state.live_proc.poll() is None)

start_col, stop_col = st.sidebar.columns(2)
if start_col.button("Start", disabled=laeuft or not event, width="stretch"):
    zeitstempel = datetime.now().strftime("%Y%m%d_%H%M%S")
    slug = event.strip().replace(" ", "_")
    pfad = OUT / f"live_{int(jahr)}_{slug}_{ident}_{zeitstempel}.txt"
    proc = subprocess.Popen(
        [sys.executable, "-m", "fastf1.livetiming", "save", str(pfad),
         "--timeout", str(int(timeout_min) * 60)],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    st.session_state.live_proc = proc
    st.session_state.live_pfad = pfad
    st.session_state.live_meta = {"jahr": int(jahr), "event": event,
                                  "ident": ident}
    st.rerun()

if stop_col.button("Stopp", disabled=not laeuft, width="stretch"):
    st.session_state.live_proc.terminate()
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.caption("Eine gestartete Aufzeichnung laeuft im Hintergrund "
                   "weiter, auch wenn diese Seite verlassen wird - der "
                   "Subprocess ist unabhaengig von der Streamlit-Session.")

alte_aufzeichnungen = sorted(OUT.glob("live_*.txt"), reverse=True)
gewaehlte_datei = None
if alte_aufzeichnungen:
    st.sidebar.markdown("---")
    st.sidebar.subheader("Vorhandene Aufzeichnungen")
    namen = [p.name for p in alte_aufzeichnungen]
    auswahl_name = st.sidebar.selectbox(
        "Ansehen (ohne laufende Aufzeichnung)", ["(aktuelle)", *namen])
    if auswahl_name != "(aktuelle)":
        gewaehlte_datei = OUT / auswahl_name

aktive_datei = st.session_state.live_pfad if laeuft else gewaehlte_datei
aktive_meta = st.session_state.live_meta

if not laeuft and aktive_datei is None:
    st.info("Keine aktive oder ausgewaehlte Aufzeichnung. Waehrend einer "
           "echten Session links Jahr/Event/Session eintragen und starten - "
           "ausserhalb eines Rennfensters verbindet sich FastF1 zwar "
           "sauber, schreibt aber 0 Bytes (schon mehrfach in diesem "
           "Projekt beobachtet, kein Fehler).")
    st.stop()

if aktive_meta is None and aktive_datei is not None:
    # eine alte datei wurde ausgewaehlt, kein session_state-eintrag dafuer -
    # metadaten aus dem dateinamen zurueckgewinnen (live_<jahr>_<event>_<ident>_<zeitstempel>.txt)
    teile = aktive_datei.stem.split("_")
    aktive_meta = {"jahr": int(teile[1]), "event": teile[2], "ident": teile[3]}

# datei kann kurz nach dem Start noch fehlen (subprocess-anlaufzeit) - nicht
# abbrechen, sonst kommt das fragment unten (mit seinem eigenen 5s-refresh)
# nie zum laufen und die seite bleibt auf diesem stand haengen.
assert aktive_datei is not None
datei_da = aktive_datei.exists()
groesse_kb = aktive_datei.stat().st_size / 1024 if datei_da else 0.0
k = st.columns(4)
k[0].metric("Status", "Aktiv" if laeuft else "Beendet")
k[1].metric("Event", f"{aktive_meta['event']} {aktive_meta['jahr']}")
k[2].metric("Session", aktive_meta["ident"])
k[3].metric("Aufzeichnungsgroesse",
           f"{groesse_kb:.1f} KB" if datei_da else "wird erstellt ...")


@st.fragment(run_every="5s" if laeuft else None)
def live_ansicht():
    if not aktive_datei.exists() or aktive_datei.stat().st_size < 200:
        st.info("Warte auf Daten ... (die Datei waechst erst, sobald "
               "wirklich eine Session laeuft)")
        return
    try:
        live = LiveTimingData(str(aktive_datei))
        live.load()
        ses = fastf1.get_session(aktive_meta["jahr"], aktive_meta["event"],
                                 aktive_meta["ident"])
        ses.load(livedata=live, telemetry=False, weather=False)
        if ses.laps.empty:
            st.info("Verbindung steht, aber noch keine vollstaendige Runde "
                   "aufgezeichnet.")
            return
        p30 = _p30()
        mit_rolling = p30.rolling_pace(ses.laps)
        rang = p30.aktuelle_rangliste(mit_rolling)
    except Exception as exc:
        st.info(f"Noch nicht genug Daten fuer eine Auswertung ({exc}).")
        return

    if rang.empty:
        st.info("Noch keine rollierende Pace berechenbar (mindestens 3 "
               "Runden je Fahrer noetig).")
        return

    fig = go.Figure(go.Bar(
        x=rang["rolling"], y=rang["Driver"], orientation="h",
        marker={"color": d.SERIEN[0]}))
    zeige(fig, hoehe=max(340, 26 * len(rang)), showlegend=False,
         xaxis=achse("Rollierende Pace, letzte 5 Runden [s]"),
         yaxis=namensachse())
    st.caption(f"Stand: Runde {int(rang['LapNumber'].max())}, "
              f"{len(rang)} Fahrer mit auswertbarer Pace. "
              f"Aktualisiert alle 5s." if laeuft else
              f"Letzter Stand dieser Aufzeichnung: Runde "
              f"{int(rang['LapNumber'].max())}.")


live_ansicht()
