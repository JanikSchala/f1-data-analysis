"""Gemeinsame Bausteine der Oberflaeche.

Die App rechnet nichts selbst. Jede Kennzahl kommt aus :mod:`f1lab`, jede
Farbe aus :mod:`f1lab.design` - hier steht nur, was Anordnung ist. Sobald in
einer Seite eine Formel oder ein Farbwert auftaucht, gehoert er dorthin und
nicht hierher, sonst driften Skripte und Oberflaeche auseinander.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

# "streamlit run" legt nur den Ordner des Startskripts auf den Pfad. Die App
# liegt in app/, das Paket eine Ebene darueber - ohne diese Zeilen scheitert
# "import f1lab" aus einem frischen Checkout heraus.
REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

import f1lab  # noqa: E402
from f1lab import design as d  # noqa: E402

# Reihenfolge eines Rennwochenendes, nicht alphabetisch.
IDENT_ORDER = ["R", "S", "Q", "SQ", "FP3", "FP2", "FP1"]

_CSS = f"""<style>
  .block-container {{padding-top: 2.2rem; padding-bottom: 3rem;}}
  [data-testid="stMetric"] {{
      background: {d.BG_HELL}; border: 1px solid {d.GRID};
      border-radius: 12px; padding: 14px 16px;}}
  [data-testid="stMetricLabel"] p {{color: {d.MUTED}; font-size: 0.82rem;}}
  h1, h2, h3 {{letter-spacing: -0.01em;}}
  .hinweis {{color: {d.MUTED}; font-size: 0.88rem; margin: -6px 0 14px 0;}}
</style>"""


@dataclass(frozen=True)
class Auswahl:
    """Die in der Seitenleiste gewaehlte Session."""
    season: int
    event: str
    ident: str
    telemetrie: bool

    @property
    def titel(self) -> str:
        return f"{self.event} {self.season} \N{MIDDLE DOT} " \
               f"{d.IDENT_NAME.get(self.ident, self.ident)}"


# --- Aufbau ---------------------------------------------------------------
def setup(titel: str, beschreibung: str = "") -> Path | None:
    """Seitenkonfiguration, Stil, Cache - der immer gleiche Seitenkopf.

    Der Cache wird im Offline-Modus aktiviert: die App zieht nie selbst Daten
    nach. Was fehlt, holt der Warmup aus p01 - dort gehoert es hin, weil es
    Stunden dauern kann und ins Rate-Limit der API laeuft.
    """
    st.set_page_config(page_title=f"F1 Analyse - {titel}", page_icon="\N{CHEQUERED FLAG}",
                       layout="wide")
    st.markdown(_CSS, unsafe_allow_html=True)

    pfad = f1lab.find_cache()
    if pfad is not None:
        f1lab.enable_cache(pfad, offline=True)

    st.title(titel)
    if beschreibung:
        hinweis(beschreibung)
    return pfad


@st.cache_data(show_spinner=False)
def inventar(cache_pfad: str) -> pd.DataFrame:
    """Was liegt im Cache? Liest nur Ordnernamen, braucht also kein Netz.

    Der Pfad steht im Argument, damit Streamlit den Zwischenspeicher verwirft,
    sobald ein anderer Ordner angebunden wird.
    """
    return f1lab.cached_sessions(cache_pfad)


def kein_cache_hinweis(pfad: Path | None) -> bool:
    """True, wenn nichts auswertbar ist - die Seite bricht dann ab.

    Ein leerer Cache ist kein Fehler, sondern der Zustand vor dem ersten
    Warmup. Die Meldung sagt deshalb in einem Satz, was zu tun ist, und nicht,
    was schiefgegangen ist.
    """
    if pfad is not None and not inventar(str(pfad)).empty:
        return False

    st.error("**Es liegen noch keine Renndaten auf diesem Rechner.**")
    st.markdown(
        "Damit die Auswertung ohne Wartezeit laeuft, werden die Daten einmalig "
        "heruntergeladen und gespeichert. Starte dafuer einmal:\n\n"
        "```\npython 01_grundlagen/p01_*.py\n```\n\n"
        "Das dauert je nach Umfang einige Stunden - der Rechner darf dabei "
        "laufen, du musst nichts weiter tun. Danach diese Seite neu laden.")
    if pfad is None:
        st.caption("Gesucht wurde in dieser Reihenfolge: `$F1_CACHE`, "
                   "`~/f1_cache`, und ein `f1_cache` neben dem Repository.")
    else:
        st.caption(f"Gefundener, aber leerer Cache: `{pfad}`")
    return True


# --- Auswahl --------------------------------------------------------------
def _wahl(label, optionen, state_key, hilfe=None, formatierer=str):
    """Auswahlfeld, das seinen Wert ueber Seitenwechsel behaelt.

    Bewusst ohne Streamlits ``key=``: die Listen haengen voneinander ab, und
    ein gespeicherter Wert, der in der neuen Liste nicht mehr vorkommt, wuerde
    dort einen Fehler ausloesen statt einfach zurueckzufallen.
    """
    vorher = st.session_state.get(state_key)
    index = optionen.index(vorher) if vorher in optionen else 0
    wert = st.sidebar.selectbox(label, optionen, index=index,
                                format_func=formatierer, help=hilfe)
    st.session_state[state_key] = wert
    return wert


def sidebar_session(pfad: Path, nur_mit_telemetrie: bool = False) -> Auswahl | None:
    """Session-Auswahl in der Seitenleiste, allein aus dem Cache-Bestand.

    Angeboten wird ausschliesslich, was auch auswertbar ist. Der Umweg ueber
    die Bestandsaufnahme ist noetig, weil FastF1 im Offline-Modus nicht
    scheitert, wenn Daten fehlen - der Fehler faellt erst beim Zugriff auf die
    Runden an, und dann steht die Oberflaeche schon.
    """
    inv = inventar(str(pfad))
    inv = inv[inv["timing"]]
    if nur_mit_telemetrie:
        inv = inv[inv["telemetry"]]
        if inv.empty:
            st.info("**Fuer keine gespeicherte Session liegt Telemetrie vor.**\n\n"
                    "Telemetrie ist ein eigener, deutlich groesserer Download. "
                    "Im Warmup aus p01 laesst sie sich mit `telemetry=True` "
                    "anfordern.")
            return None

    with st.sidebar:
        st.header("Auswahl")
        saison = _wahl("Saison", sorted(inv["season"].unique(), reverse=True),
                       "f1_season")

        events = inv[inv["season"] == saison].sort_values("event_date")
        event = _wahl("Rennen", list(dict.fromkeys(events["event"])), "f1_event")

        passend = events[events["event"] == event]
        idents = [i for i in IDENT_ORDER if i in set(passend["ident"])]
        ident = _wahl("Session", idents, "f1_ident",
                      formatierer=lambda i: d.IDENT_NAME.get(i, i))

        zeile = passend[passend["ident"] == ident].iloc[0]
        auswahl = Auswahl(int(saison), str(event), str(ident),
                          bool(zeile["telemetry"]))

        st.divider()
        if auswahl.telemetrie:
            st.success("Telemetrie vorhanden", icon="\N{WHITE HEAVY CHECK MARK}")
        else:
            st.info("Nur Rundendaten - fuer diese Session wurde keine "
                    "Telemetrie geladen. Streckenverlauf und Bremszonen "
                    "bleiben deshalb leer.")
        gesamt = inventar(str(pfad))
        st.caption(f"{int(gesamt['timing'].sum())} Sessions gespeichert, "
                   f"{gesamt['season'].min()}\N{EN DASH}{gesamt['season'].max()}")

    return auswahl


def lade(auswahl: Auswahl, telemetrie: bool = False):
    """Session aus dem Cache holen.

    ``f1lab.load`` haelt die zuletzt geladenen Sessions im Speicher, ein
    Seitenwechsel kostet also nichts.
    """
    try:
        with st.spinner(f"{auswahl.titel} wird geladen ..."):
            ses = f1lab.load(auswahl.season, auswahl.event, auswahl.ident,
                             telemetry=telemetrie)
            _ = ses.laps          # erzwingt den Zugriff, siehe unten
            return ses
    except Exception as exc:
        # Im Offline-Modus laedt FastF1 klaglos eine leere Session; der Fehler
        # faellt erst beim Zugriff auf die Runden an. Deshalb oben der
        # erzwungene Zugriff - lieber hier eine klare Meldung als spaeter eine
        # Seite voller Ausnahmen.
        st.error("**Diese Session laesst sich nicht oeffnen.**\n\n"
                 "Meist fehlt ein Teil der Daten im Speicher. Waehle links "
                 "eine andere Session, oder lade sie mit p01 nach.")
        st.caption(f"Technischer Hinweis: `{type(exc).__name__}`")
        st.stop()


def kopfzeile(ses, auswahl: Auswahl) -> None:
    """Ueberschrift und vier Eckdaten - auf jeder Session-Seite gleich."""
    st.subheader(auswahl.titel)
    laps = ses.laps
    k = st.columns(4)
    k[0].metric("Runden gefahren", f"{len(laps):,}".replace(",", "."))
    k[1].metric("Fahrer", int(laps["Driver"].nunique()))

    schnellste = laps["LapTime"].min()
    k[2].metric("Schnellste Runde",
                f"{schnellste.total_seconds():.3f} s"
                if pd.notna(schnellste) else "\N{EN DASH}")
    try:
        sieger = ses.results.iloc[0]
        k[3].metric("Vorne", str(sieger["Abbreviation"]),
                    str(sieger["TeamName"])[:18])
    except Exception:
        k[3].metric("Vorne", "\N{EN DASH}")


def nur_rennen(auswahl: Auswahl, was: str) -> bool:
    """Manche Auswertungen ergeben nur im Rennen oder Sprint einen Sinn."""
    if auswahl.ident in ("R", "S"):
        return False
    st.info(f"{was} braucht ein Rennen oder einen Sprint - "
            f"{d.IDENT_NAME.get(auswahl.ident, auswahl.ident)} hat weder "
            "Boxenstopps noch eine belastbare Renndistanz.")
    return True


# --- Darstellung ----------------------------------------------------------
def zeige(fig: go.Figure, hoehe: int = 420, **layout) -> None:
    """Diagramm im Hausstil rendern.

    Das Grundlayout kommt aus :func:`f1lab.design.plotly_layout`, damit App
    und die von den Skripten erzeugten Grafiken zusammenpassen.
    """
    fig.update_layout(**d.plotly_layout(hoehe=hoehe, **layout))
    st.plotly_chart(fig, width="stretch")


def achse(titel: str | None = None, **kw) -> dict:
    """Achsendefinition im Hausstil, fuer die Uebergabe an :func:`zeige`."""
    achse = {"title": titel or "", "gridcolor": d.GRID, "zeroline": False,
             "linecolor": d.GRID, "tickfont": {"color": d.MUTED}}
    achse.update(kw)
    return achse


def namensachse(titel: str | None = None, **kw) -> dict:
    """Achse fuer Beschriftungen statt Zahlen - kein Gitter, heller Text."""
    return achse(titel, gridcolor="rgba(0,0,0,0)",
                 tickfont={"color": d.FG}, **kw)


def hinweis(text: str) -> None:
    """Eine Zeile darunter, die erklaert, wie die Zahlen zu lesen sind.

    Steht bewusst unter jeder Auswertung: eine Zahl ohne ihre Grenzen ist in
    diesem Datensatz regelmaessig irrefuehrend.
    """
    st.markdown(f'<p class="hinweis">{text}</p>', unsafe_allow_html=True)


def tabelle(df: pd.DataFrame, **kw) -> None:
    """DataFrame in voller Breite, ohne Index."""
    st.dataframe(df, width="stretch", hide_index=True, **kw)
