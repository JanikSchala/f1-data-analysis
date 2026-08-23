"""gemeinsame bausteine der oberflaeche.

die app rechnet nichts selbst. jede kennzahl kommt aus :mod:`f1lab`, jede
farbe aus :mod:`f1lab.design`. hier steht nur anordnung, sonst driften
skripte und oberflaeche auseinander.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

# "streamlit run" legt nur den ordner des startskripts auf den pfad. die app
# liegt in app/, das paket eine ebene darueber. ohne diese zeilen scheitert
# "import f1lab" aus einem frischen checkout.
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

# futuristische telemetrie-optik: farben ausschliesslich aus f1lab.design
# (ACCENT/TELEMETRY sind reine chrome-deko, diagrammfarben bleiben
# unberuehrt). fonts kommen primaer aus .streamlit/config.toml
# (Orbitron/Rajdhani/Share Tech Mono). das @import hier sichert nur die
# verfuegbarkeit fuer selektoren, die Streamlits eigene font-zuweisung nicht
# erreicht (z.B. ::before-pseudoelemente).
_CSS = f"""<style>
  @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@600;700;800&family=Share+Tech+Mono&display=swap');

  .block-container {{padding-top: 2.4rem; padding-bottom: 3rem;}}

  /* Scanline oben - eine schmale, leicht pulsierende Akzentleiste als
     wiederkehrendes Motiv, kein Dekor pro Seite von Hand gepflegt. */
  [data-testid="stAppViewContainer"]::before {{
      content: ""; position: fixed; top: 0; left: 0; right: 0; height: 2px;
      z-index: 999; background: linear-gradient(90deg,
          transparent 0%, {d.ACCENT} 20%, {d.TELEMETRY} 50%,
          {d.ACCENT} 80%, transparent 100%);
      background-size: 200% 100%; opacity: 0.85;
      animation: f1-scan 7s linear infinite;}}
  @keyframes f1-scan {{
      0% {{background-position: 200% 0;}}
      100% {{background-position: -200% 0;}}}}

  /* Sehr feine Diagonalstreifen im Hintergrund - Andeutung von Speedlines,
     niedrige Opazitaet, damit Text/Diagramme nicht darunter leiden. */
  [data-testid="stAppViewContainer"] {{
      background-image: repeating-linear-gradient(-45deg,
          rgba(255,255,255,0.012) 0px, rgba(255,255,255,0.012) 1px,
          transparent 1px, transparent 42px);}}

  h1, h2, h3 {{letter-spacing: 0.01em;}}
  h1 {{text-transform: uppercase; letter-spacing: 0.03em;}}

  /* ##### -Zwischenueberschriften (h5, die de-facto-Sektionsmarke dieser
     App) bekommen einen HUD-Tag: Akzentstrich plus Uppercase-Tracking. */
  h5 {{
      text-transform: uppercase; letter-spacing: 0.08em; font-size: 0.95rem !important;
      color: {d.FG}; border-left: 3px solid {d.ACCENT};
      padding-left: 10px; margin-top: 1.4rem !important;}}

  [data-testid="stMetric"] {{
      background: linear-gradient(160deg, {d.BG_HELL} 0%, {d.BG} 100%);
      border: 1px solid {d.GRID}; border-top: 2px solid {d.ACCENT};
      border-radius: 4px; padding: 14px 16px;
      box-shadow: 0 0 18px -6px {d.ACCENT_GLOW};
      clip-path: polygon(0 0, 100% 0, 100% 100%, 12px 100%, 0 calc(100% - 12px));}}
  [data-testid="stMetricLabel"] p {{
      color: {d.MUTED}; font-size: 0.76rem; text-transform: uppercase;
      letter-spacing: 0.06em;}}
  [data-testid="stMetricValue"] {{
      font-family: 'Share Tech Mono', monospace; color: {d.TELEMETRY};}}

  /* Sidebar: eigener, noch dunklerer Ton mit gluehender Trennkante. */
  [data-testid="stSidebar"] {{
      border-right: 1px solid {d.GRID};
      box-shadow: 4px 0 24px -12px {d.ACCENT_GLOW};}}
  [data-testid="stSidebar"] h1 {{font-size: 1.1rem !important;}}

  /* Tabs: aktiver Reiter mit Akzentlinie statt der Standardfarbe. */
  button[data-baseweb="tab"][aria-selected="true"] {{
      color: {d.ACCENT} !important;}}
  div[data-baseweb="tab-highlight"] {{
      background-color: {d.ACCENT} !important;
      box-shadow: 0 0 8px 0 {d.ACCENT_GLOW};}}

  /* Duenner, dunkler Scrollbalken statt der Browser-Voreinstellung. */
  ::-webkit-scrollbar {{width: 10px; height: 10px;}}
  ::-webkit-scrollbar-track {{background: {d.BG};}}
  ::-webkit-scrollbar-thumb {{
      background: {d.GRID}; border-radius: 5px;}}
  ::-webkit-scrollbar-thumb:hover {{background: {d.ACCENT};}}

  .hinweis {{
      color: {d.MUTED}; font-size: 0.88rem; margin: -6px 0 14px 0;
      border-left: 2px solid {d.GRID}; padding-left: 10px;}}
</style>"""


@dataclass(frozen=True)
class Auswahl:
    """die in der seitenleiste gewaehlte session."""
    season: int
    event: str
    ident: str
    telemetrie: bool

    @property
    def titel(self) -> str:
        return f"{self.event} {self.season} \N{MIDDLE DOT} " \
               f"{d.IDENT_NAME.get(self.ident, self.ident)}"


# --- aufbau ---------------------------------------------------------------
def konfiguriere_app() -> None:
    """seitenkonfiguration und stil, einmal fuer die ganze app.

    muss vor st.navigation(...) im einstiegspunkt laufen, nicht mehr pro
    seite wie vor der umstellung auf st.navigation - set_page_config()
    darf nur einmal je skriptlauf aufgerufen werden, der einstiegspunkt
    laeuft aber bei jedem rerun mit, jede ausgewaehlte seite haengt sich
    daran nur noch per .run() an.
    """
    st.set_page_config(page_title="F1 Analyse", page_icon="\N{CHEQUERED FLAG}",
                       layout="wide")
    st.markdown(_CSS, unsafe_allow_html=True)


def setup(titel: str, beschreibung: str = "") -> Path | None:
    """cache aktivieren und den seitenkopf zeigen: der immer gleiche teil
    jeder einzelnen seite.

    der cache laeuft im offline-modus, die app zieht nie selbst daten nach.
    das erledigt der warmup aus p01, weil es stunden dauern kann und ins
    rate-limit der API laeuft.
    """
    pfad = f1lab.find_cache()
    if pfad is not None:
        f1lab.enable_cache(pfad, offline=True)

    st.title(titel)
    if beschreibung:
        hinweis(beschreibung)
    return pfad


@st.cache_data(show_spinner=False)
def inventar(cache_pfad: str) -> pd.DataFrame:
    """was liegt im cache? liest nur ordnernamen, braucht also kein netz.

    der pfad steht im argument, damit Streamlit den zwischenspeicher
    verwirft, sobald ein anderer ordner angebunden wird.
    """
    return f1lab.cached_sessions(cache_pfad)


def kein_cache_hinweis(pfad: Path | None) -> bool:
    """true, wenn nichts auswertbar ist, die seite bricht dann ab.

    ein leerer cache ist kein fehler, sondern der zustand vor dem ersten
    warmup. die meldung sagt in einem satz, was zu tun ist, nicht was
    schiefgegangen ist.
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


# --- auswahl --------------------------------------------------------------
def _wahl(label, optionen, state_key, hilfe=None, formatierer=str):
    """auswahlfeld, das seinen wert ueber seitenwechsel behaelt.

    bewusst ohne Streamlits ``key=``: die listen haengen voneinander ab. ein
    gespeicherter wert, der in der neuen liste nicht mehr vorkommt, wuerde
    sonst einen fehler ausloesen statt zurueckzufallen.
    """
    vorher = st.session_state.get(state_key)
    index = optionen.index(vorher) if vorher in optionen else 0
    wert = st.sidebar.selectbox(label, optionen, index=index,
                                format_func=formatierer, help=hilfe)
    st.session_state[state_key] = wert
    return wert


def sidebar_session(pfad: Path, nur_mit_telemetrie: bool = False) -> Auswahl | None:
    """session-auswahl in der seitenleiste, allein aus dem cache-bestand.

    angeboten wird nur, was auch auswertbar ist. der umweg ueber die
    bestandsaufnahme ist noetig, weil FastF1 im offline-modus nicht
    scheitert, wenn daten fehlen. der fehler faellt erst beim zugriff auf
    die runden an, wenn die oberflaeche schon steht.
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
    """session aus dem cache holen.

    ``f1lab.load`` haelt die zuletzt geladenen sessions im speicher, ein
    seitenwechsel kostet also nichts.
    """
    try:
        with st.spinner(f"{auswahl.titel} wird geladen ..."):
            ses = f1lab.load(auswahl.season, auswahl.event, auswahl.ident,
                             telemetry=telemetrie)
            _ = ses.laps          # erzwingt den zugriff, siehe unten
            return ses
    except Exception as exc:
        # im offline-modus laedt FastF1 klaglos eine leere session, der fehler
        # faellt erst beim zugriff auf die runden an. deshalb oben der
        # erzwungene zugriff, lieber hier eine klare meldung als spaeter eine
        # seite voller ausnahmen.
        st.error("**Diese Session laesst sich nicht oeffnen.**\n\n"
                 "Meist fehlt ein Teil der Daten im Speicher. Waehle links "
                 "eine andere Session, oder lade sie mit p01 nach.")
        st.caption(f"Technischer Hinweis: `{type(exc).__name__}`")
        st.stop()


def kopfzeile(ses, auswahl: Auswahl) -> None:
    """ueberschrift und vier eckdaten, auf jeder session-seite gleich."""
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
    """manche auswertungen ergeben nur im rennen oder sprint einen sinn."""
    if auswahl.ident in ("R", "S"):
        return False
    st.info(f"{was} braucht ein Rennen oder einen Sprint - "
            f"{d.IDENT_NAME.get(auswahl.ident, auswahl.ident)} hat weder "
            "Boxenstopps noch eine belastbare Renndistanz.")
    return True


# --- darstellung ----------------------------------------------------------
def zeige(fig: go.Figure, hoehe: int = 420, **layout) -> None:
    """diagramm im hausstil rendern.

    das grundlayout kommt aus :func:`f1lab.design.plotly_layout`, damit app
    und die von den skripten erzeugten grafiken zusammenpassen.
    """
    fig.update_layout(**d.plotly_layout(hoehe=hoehe, **layout))
    st.plotly_chart(fig, width="stretch")


def achse(titel: str | None = None, **kw) -> dict:
    """achsendefinition im hausstil, fuer die uebergabe an :func:`zeige`."""
    achse = {"title": titel or "", "gridcolor": d.GRID, "zeroline": False,
             "linecolor": d.GRID, "tickfont": {"color": d.MUTED}}
    achse.update(kw)
    return achse


def namensachse(titel: str | None = None, **kw) -> dict:
    """achse fuer beschriftungen statt zahlen, kein gitter, heller text."""
    return achse(titel, gridcolor="rgba(0,0,0,0)",
                 tickfont={"color": d.FG}, **kw)


def hinweis(text: str) -> None:
    """eine zeile darunter, die erklaert, wie die zahlen zu lesen sind.

    steht bewusst unter jeder auswertung: eine zahl ohne ihre grenzen ist in
    diesem datensatz regelmaessig irrefuehrend.
    """
    st.markdown(f'<p class="hinweis">{text}</p>', unsafe_allow_html=True)


def tabelle(df: pd.DataFrame, **kw) -> None:
    """dataframe in voller breite, ohne index."""
    st.dataframe(df, width="stretch", hide_index=True, **kw)
