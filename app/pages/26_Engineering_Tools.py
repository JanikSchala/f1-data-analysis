"""die vier Projekte, die reine Infrastruktur/CLI-Werkzeuge sind statt einer
Analyse mit eigener Storyline (P26 Data Warehouse, P27 REST-API, P29
PDF-Bericht, P34 f1analyze-CLI) - deshalb hier als Reiter statt vier eigene
Sidebar-Eintraege, dasselbe Buendelungs-Muster wie 14_Historie.py.

alle vier rufen bestehende Skripte/Pakete direkt auf statt ihre Logik zu
kopieren: P26 fragt die schon gebaute DuckDB-Datei read-only ab, P27 startet
denselben In-Prozess-Uvicorn-Thread, den P27s eigener Smoke-Test schon
benutzt, P29 ruft dessen build()-Funktion, P34 ruft den echten installierten
f1analyze-Befehl als Subprocess auf.
"""
from __future__ import annotations

import importlib.util
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import duckdb
import plotly.graph_objects as go
import requests
import streamlit as st
from common import achse, hinweis, namensachse, setup, tabelle, zeige

from f1lab import design as d

ROOT = Path(__file__).resolve().parents[2]
WAREHOUSE = ROOT / "10_data_engineering" / "f1_warehouse" / "f1.duckdb"


def _skript_importieren(rel_pfad: str):
    """dasselbe muster wie make_assets.py._skript_importieren()."""
    pfad = ROOT / rel_pfad
    spec = importlib.util.spec_from_file_location(pfad.stem, pfad)
    assert spec is not None and spec.loader is not None
    modul = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modul)
    return modul


setup("Engineering-Tools", "Die vier Infrastruktur-/CLI-Projekte, die keine "
                           "eigene Analyse-Seite brauchen, aber trotzdem "
                           "direkt aus dem Dashboard nutzbar sein sollen: "
                           "Data Warehouse (P26), REST-API (P27), "
                           "PDF-Bericht (P29), f1analyze-CLI (P34).")

tab_wh, tab_api, tab_pdf, tab_cli = st.tabs(
    ["Data Warehouse", "REST-API", "PDF-Bericht", "f1analyze-CLI"])

# ==================================================================== P26
with tab_wh:
    if not WAREHOUSE.exists():
        st.info(f"Noch kein Warehouse unter `{WAREHOUSE.relative_to(ROOT)}` "
               "gefunden. Einmal bauen mit:\n\n"
               "```\npython 10_data_engineering/"
               "p26_f1_data_warehouse_sternschema_in_duckdb.py\n```")
    else:
        con = duckdb.connect(str(WAREHOUSE), read_only=True)
        views = con.execute(
            "SELECT view_name FROM duckdb_views() WHERE NOT internal "
            "ORDER BY 1").fetchdf()["view_name"].tolist()
        wahl = st.selectbox("Tabelle/View", views,
                            index=views.index("mart_driver_pace")
                            if "mart_driver_pace" in views else 0)
        df = con.execute(f"SELECT * FROM {wahl} LIMIT 500").fetchdf()
        st.caption(f"{len(df)} Zeilen (auf 500 begrenzt)")
        tabelle(df)

        if wahl == "mart_driver_pace" and not df.empty:
            rang = (df.groupby("Driver")["rel_pace"].mean()
                   .sort_values().head(15))
            delta_pct = (rang - 1.0) * 100
            fig = go.Figure(go.Bar(x=delta_pct.to_numpy(), y=rang.index,
                                   orientation="h",
                                   marker={"color": d.SERIEN[0]}))
            zeige(fig, hoehe=max(340, 26 * len(rang)), showlegend=False,
                 xaxis=achse("Mittlere relative Pace ggue. Event-Schnellster [%]"),
                 yaxis=namensachse())
            hinweis("Dieselbe Kennzahl wie das Warehouse-Highlight im "
                   "README (siehe P26/`make_assets.py`s `warehouse_pace()`) "
                   "- hier direkt live aus der DuckDB-Datei statt "
                   "vorgerechnet.")
        con.close()

# ==================================================================== P27
with tab_api:
    if "api_server" not in st.session_state:
        st.session_state.api_server = None
        st.session_state.api_basis = None

    laeuft = st.session_state.api_server is not None
    col1, col2 = st.columns(2)
    if col1.button("Server starten", disabled=laeuft):
        p27 = _skript_importieren(
            "10_data_engineering/p27_telemetrie_api_mit_fastapi.py")
        with st.spinner("uvicorn wird lokal gestartet ..."):
            server, basis = p27._server_starten()
        st.session_state.api_server = server
        st.session_state.api_basis = basis
        st.rerun()
    if col2.button("Server stoppen", disabled=not laeuft):
        st.session_state.api_server.should_exit = True
        st.session_state.api_server = None
        st.session_state.api_basis = None
        st.rerun()

    if not laeuft:
        st.info("Startet denselben lokalen FastAPI-Server, den P27s "
               "Smoke-Test benutzt (In-Prozess-Uvicorn-Thread, Port 8321, "
               "nur auf `127.0.0.1`).")
    else:
        basis = st.session_state.api_basis
        st.success(f"Server laeuft unter `{basis}`.")
        beispiele = {
            "GET /sessions/2024": "/sessions/2024",
            "GET /laps/2024/Bahrain/R": "/laps/2024/Bahrain/R",
        }
        wahl = st.radio("Beispiel-Endpunkt", list(beispiele), horizontal=True)
        if st.button("Anfrage senden"):
            url = basis + beispiele[wahl]
            try:
                r = requests.get(url, timeout=15)
                st.caption(f"`GET {beispiele[wahl]}` -> {r.status_code} in "
                          f"{r.elapsed.total_seconds() * 1000:.0f} ms")
                st.json(r.json())
            except Exception as exc:
                st.error(f"Anfrage fehlgeschlagen: {exc}")

# ==================================================================== P29
with tab_pdf:
    c1, c2 = st.columns(2)
    jahr_pdf = c1.number_input("Jahr", min_value=2018, max_value=2100,
                               value=2024, step=1, key="pdf_jahr")
    event_pdf = c2.text_input("Event (leer = juengstes Rennen der Saison)",
                              value="", key="pdf_event")
    if st.button("PDF erzeugen"):
        p29 = _skript_importieren(
            "11_visualisierung/p29_automatischer_rennbericht_als_pdf.py")
        with st.spinner("Bericht wird erzeugt (laedt bei Bedarf die "
                       "Session) ..."):
            try:
                with tempfile.TemporaryDirectory() as tmp:
                    out = Path(tmp) / "race_report.pdf"
                    p29.build(int(jahr_pdf), event_pdf or None, out)
                    pdf_bytes = out.read_bytes()
                st.success("Bericht erzeugt.")
                st.download_button("PDF herunterladen", data=pdf_bytes,
                                   file_name=f"f1_report_{int(jahr_pdf)}.pdf",
                                   mime="application/pdf")
            except Exception as exc:
                st.error(f"Bericht konnte nicht erzeugt werden: {exc}")

# ==================================================================== P34
with tab_cli:
    f1analyze_pfad = shutil.which("f1analyze") or str(
        Path(sys.executable).parent / "f1analyze")
    if not Path(f1analyze_pfad).exists():
        st.info("`f1analyze` ist nicht installiert. Einmal einrichten mit:"
               "\n\n```\npip install -e "
               "10_data_engineering/f1analyze\n```")
    else:
        befehle = {
            "pace": "bereinigte Race Pace als Rangliste",
            "strategy": "Stints und Compounds je Fahrer",
            "weekend": "komplette Wochenendanalyse als Text",
        }
        c1, c2, c3 = st.columns(3)
        befehl = c1.selectbox("Befehl", list(befehle),
                              format_func=lambda b: f"{b} - {befehle[b]}")
        jahr_cli = c2.number_input("Jahr", min_value=2018, max_value=2100,
                                   value=2024, step=1, key="cli_jahr")
        event_cli = c3.text_input("Event", value="Bahrain", key="cli_event")
        if st.button("Ausfuehren"):
            cmd = [f1analyze_pfad, befehl, str(int(jahr_cli)), event_cli]
            with st.spinner(f"`{' '.join(cmd)}` laeuft ..."):
                erg = subprocess.run(cmd, capture_output=True, text=True,
                                     timeout=120)
            if erg.returncode == 0:
                st.code(erg.stdout or "(keine Ausgabe)")
            else:
                st.error("Befehl ist fehlgeschlagen.")
                st.code(erg.stderr)
