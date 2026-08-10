"""
P30 - Live-Timing aufzeichnen und in Echtzeit auswerten
=======================================================

Waehrend einer laufenden Session den Live-Timing-Stream mitschneiden und anschliessend wie eine normale Session laden.

Kategorie:   Live Timing
Niveau:      Profi
Aufwand:     6-8 h
Schwerpunkt: Engineering, Strategie

WARUM DAS LOHNT
Echtzeitdatenverarbeitung ist technisch anspruchsvoll und macht Spass: waehrend das Rennen laeuft, entsteht dein eigenes Timing-Board. Wer das gebaut hat, hat den F1-Datenstack wirklich verstanden.

VORGEHEN
  1. Recorder per CLI starten: python -m fastf1.livetiming save output.txt
  2. Alternativ SignalRClient programmatisch mit asyncio starten
  3. Aufzeichnung in LiveTimingData laden
  4. Session mit livedata=... laden und normal analysieren
  5. Rolling-Auswertung: Pace der letzten 5 Runden je Fahrer

GENUTZTE FASTF1-BAUSTEINE
  - fastf1.livetiming.client.SignalRClient
  - fastf1.livetiming.data.LiveTimingData
  - Session.load(livedata=...)

AUSBAUSTUFE  [umgesetzt]
Schreibe die Live-Daten in eine Zeitreihen-DB (InfluxDB oder TimescaleDB) und
baue ein Grafana-Board, das waehrend des Rennens aktualisiert.

Dieses Projekt ist anders als die 29 davor: es gibt keine Cache-Datei, gegen
die man testen kann - Live-Timing existiert nur waehrend eine Session
laeuft. Am 2026-08-10 (Sommerpause, naechstes Rennen erst am 23.08.) laeuft
keine. Deshalb wurde, statt das zu ignorieren, tatsaechlich probiert:

VORGEHEN 1 real ausgefuehrt (`python -m fastf1.livetiming save
/tmp/live_test.txt`, 30s laufen lassen): der Client verbindet sich
("Starting FastF1 live timing client") und blockiert korrekt auf die
Verbindung, schreibt aber 0 Bytes - erwartbar ausserhalb eines Sessionfensters,
jetzt tatsaechlich beobachtet statt nur behauptet. Dabei zwei echte,
versionsbedingte Bugs in der Vorlage gefunden:

  - `debug=False` an SignalRClient() zu uebergeben laeuft noch, aber ein
    `--debug`-Flag ueber die CLI oder debug=True direkt wirft
    `ValueError: Debug mode is no longer supported.` - der Modus wurde aus
    fastf1 entfernt.
  - VORGEHEN 2 ("mit asyncio starten") ist mit der installierten fastf1-
    Version (3.8.3) nicht mehr moeglich: `SignalRClient.async_start()`
    wirft `NotImplementedError("... no longer uses asyncio! Please use
    .start instead.")`. Die Vorlage importierte `asyncio`, benutzte es aber
    nirgends - vermutlich ein Rest aus einer aelteren fastf1-Version, in der
    async_start() der empfohlene Weg war. `record()` nutzt deshalb
    ausschliesslich das synchrone `.start()`.

VORGEHEN 3-5 (LiveTimingData laden, rollierende Pace) lassen sich ohne
laufende Session nicht ueber den echten Pfad testen. Um trotzdem gegen
echte Daten zu pruefen: die Rechenlogik (`rolling_pace()`) ist von der
Ladefunktion getrennt und wird gegen eine ECHTE, bereits geladene Session
(Bahrain 2024 R) verifiziert - `ses.laps` hat nach dem Laden dieselbe Form,
egal ob die Runden aus dem Live-Feed oder aus dem historischen Archiv
stammen. Das prueft die Berechnung ehrlich, nicht das Laden selbst.

AUSBAUSTUFE: weder influxdb-client noch psycopg2/sqlalchemy stehen in
requirements.txt, kein lokaler InfluxDB/TimescaleDB/Grafana-Prozess laeuft.
Umgesetzt mit stdlib sqlite3 als Zeitreihen-Speicher (append-only, ein
Datenpunkt je Fahrer und Runde) und einem PNG-"Board" statt eines
Grafana-Dashboards - dieselbe Idee, ohne Infrastruktur, die hier niemand
betreiben wuerde. Gefuellt wird die Zeitreihe per Replay: dieselbe echte
Bahrain-2024-R-Session, Runde fuer Runde in ihrer tatsaechlichen
Startzeit-Reihenfolge eingespielt (nicht die fertige Endauswertung auf
einen Schlag) - genau der Verarbeitungsablauf, den ein echter Live-Feed
Runde fuer Runde erzeugen wuerde, nur mit echten historischen statt echten
Live-Daten gefuettert. Der erste Entwurf des Replays baute den
"bisher eingetroffenen" Datensatz per wiederholtem pd.concat() auf ein
leeres DataFrame auf - dabei degradiert LapTime von timedelta64 zu object,
und der .dt-Accessor in rolling_pace() bricht ("Can only use .dt accessor
with datetimelike values"). Gefixt durch Slicing der schon korrekt
typisierten Originaltabelle (laps.iloc[:i+1]) statt inkrementellem Konkat.
"""
from __future__ import annotations

import sqlite3
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")                      # kein Fenster, nur Dateien

import fastf1
import matplotlib.pyplot as plt
import pandas as pd
from fastf1.livetiming.client import SignalRClient
from fastf1.livetiming.data import LiveTimingData

import f1lab
from f1lab.design import FG, GRID, MUTED, SERIEN, matplotlib_stil

warnings.filterwarnings("ignore")
fastf1.set_log_level("ERROR")

OUT = Path(__file__).parent / "out"
OUT.mkdir(exist_ok=True)

FENSTER = 5     # Runden fuer die rollierende Pace, VORGEHEN 5

plt.rcParams.update(matplotlib_stil())


# --------------------------------------------------------------- VORGEHEN 1-2
def record(path: str = "saved_data.txt", timeout_min: int = 120) -> None:
    """Waehrend einer laufenden Session aufrufen. Blockiert bis zum Ende der
    Session oder Ctrl+C - kein Rueckgabewert, das File waechst waehrenddessen.

    Nur .start() (synchron) - async_start() wirft seit fastf1 3.x
    NotImplementedError, siehe Docstring. debug bleibt bewusst weg: der
    Parameter existiert noch, aber jeder Wert ausser False wirft seit
    Kurzem einen ValueError.
    """
    client = SignalRClient(filename=path, timeout=timeout_min * 60)
    client.start()


# ------------------------------------------------------------------ VORGEHEN 5
def rolling_pace(laps: pd.DataFrame, fenster: int = FENSTER) -> pd.DataFrame:
    """Rollierende Median-Pace je Fahrer - unabhaengig davon, ob `laps` aus
    einem Live-Feed oder einem historischen Archiv stammt (siehe Docstring)."""
    df = laps.copy()
    df["sec"] = df["LapTime"].dt.total_seconds()
    df = df.sort_values(["Driver", "LapNumber"])
    df["rolling"] = (df.groupby("Driver")["sec"]
                     .transform(lambda s: s.rolling(fenster, min_periods=3).median()))
    return df


def aktuelle_rangliste(df_mit_rolling: pd.DataFrame) -> pd.DataFrame:
    """Neuester Rolling-Wert je Fahrer, schnellster zuerst."""
    return (df_mit_rolling.dropna(subset=["rolling"])
           .sort_values("LapNumber").groupby("Driver").tail(1)
           .sort_values("rolling"))


# ------------------------------------------------------------------ VORGEHEN 3-4
def analyse(path: str, year: int, gp: str, ident: str) -> pd.DataFrame:
    """Aufzeichnung laden und wie eine normale Session behandeln."""
    live = LiveTimingData(path)
    live.load()

    ses = fastf1.get_session(year, gp, ident)
    ses.load(livedata=live, telemetry=False, weather=False)

    df = rolling_pace(ses.laps)
    rang = aktuelle_rangliste(df)
    print("Aktuelle Pace (Median letzte 5 Runden):")
    print(rang[["Driver", "LapNumber", "Compound", "TyreLife", "rolling"]]
         .round(3).to_string(index=False))
    return df


# ------------------------------------------------------------------- AUSBAUSTUFE
def zeitreihe_anlegen(db_pfad: Path) -> sqlite3.Connection:
    con = sqlite3.connect(db_pfad)
    con.execute("""CREATE TABLE IF NOT EXISTS pace_snapshots(
        ts TEXT, driver TEXT, lap INTEGER, compound TEXT,
        tyre_life REAL, rolling_pace_s REAL)""")
    return con


def replay_als_livefeed(ses, con: sqlite3.Connection) -> None:
    """Ersetzt den echten Live-Feed: dieselbe Session, Runde fuer Runde in
    ihrer tatsaechlichen Reihenfolge eingespielt, nicht als fertige
    Endauswertung. Nach jeder neu "eintreffenden" Runde wird die
    rollierende Pace neu berechnet und genau der neue Punkt geschrieben -
    das ist der Teil, der bei einem echten Feed inkrementell passieren
    wuerde."""
    laps = ses.laps.dropna(subset=["LapStartTime", "LapTime"]).copy()
    laps = laps.sort_values("LapStartTime").reset_index(drop=True)

    for i in range(len(laps)):
        # Slice der schon-korrekt-typisierten Runden statt wiederholtem
        # pd.concat() auf ein leeres DataFrame - letzteres degradiert
        # LapTime von timedelta64 zu object (.dt-Accessor bricht dann).
        bislang = laps.iloc[:i + 1]
        neue_runde = laps.iloc[i]
        mit_rolling = rolling_pace(bislang)
        aktuelle = mit_rolling[
            (mit_rolling["Driver"] == neue_runde["Driver"])
            & (mit_rolling["LapNumber"] == neue_runde["LapNumber"])]
        if aktuelle.empty or pd.isna(aktuelle.iloc[0]["rolling"]):
            continue
        r = aktuelle.iloc[0]
        con.execute(
            "INSERT INTO pace_snapshots VALUES (?,?,?,?,?,?)",
            (str(r["LapStartTime"]), r["Driver"], int(r["LapNumber"]),
             str(r["Compound"]), float(r["TyreLife"]) if pd.notna(r["TyreLife"]) else None,
             float(r["rolling"])))
    con.commit()


def dashboard_rendern(con: sqlite3.Connection, ses, out_png: Path) -> None:
    """"Grafana-Board" als PNG: rollierende Pace ueber die Renndistanz,
    Podium hervorgehoben - dieselbe Zeitreihe, die ein echtes Board live
    nachgezogen haette."""
    df = pd.read_sql("SELECT * FROM pace_snapshots", con)
    order = ses.results.sort_values("Position")["Abbreviation"].tolist()
    top3 = [d for d in order if d in df["driver"].unique()][:3]

    fig, ax = plt.subplots(figsize=(12, 6))
    for drv, g in df.groupby("driver"):
        if drv in top3:
            continue
        g = g.sort_values("lap")
        ax.plot(g["lap"], g["rolling_pace_s"], color=MUTED, lw=0.8, alpha=0.4)
    for i, drv in enumerate(top3):
        g = df[df["driver"] == drv].sort_values("lap")
        ax.plot(g["lap"], g["rolling_pace_s"], color=SERIEN[i], lw=2.2, label=drv)
    ax.set_xlabel("Runde")
    ax.set_ylabel(f"Rollierende Pace, {FENSTER} Runden [s]")
    ax.set_title(f"{ses.event['EventName']} {ses.event.year} - Live-Pace-Board "
                f"(Replay, siehe Docstring)", loc="left", color=FG, fontsize=13,
                pad=10)
    ax.legend(loc="upper right", frameon=False, labelcolor=FG, fontsize=9)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(alpha=0.3, linewidth=0.8, color=GRID)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(out_png, dpi=130, bbox_inches="tight")
    plt.close(fig)


def main():
    f1lab.enable_cache()

    print("[1/3] Echte Verbindung pruefen (VORGEHEN 1) ...")
    print("      Bereits separat ausgefuehrt (siehe Docstring): "
         "python -m fastf1.livetiming save /tmp/live_test.txt, 30s, "
         "0 Bytes - erwartbar ausserhalb eines Sessionfensters.")

    print("\n[2/3] Rolling-Pace-Logik gegen echte Session pruefen "
         "(VORGEHEN 5, Ersatz fuer VORGEHEN 3-4 ohne Live-Daten) ...")
    ses = f1lab.load(2024, "Bahrain", "R", telemetry=False)
    df = rolling_pace(ses.laps)
    rang = aktuelle_rangliste(df)
    print(rang[["Driver", "LapNumber", "Compound", "TyreLife", "rolling"]]
         .head(10).round(3).to_string(index=False))

    print(f"\n[3/3] AUSBAUSTUFE: Replay in SQLite-Zeitreihe, PNG-Board "
         f"({ses.event['EventName']} {ses.event.year}) ...")
    db_pfad = OUT / "live_timeseries.sqlite"
    db_pfad.unlink(missing_ok=True)
    con = zeitreihe_anlegen(db_pfad)
    replay_als_livefeed(ses, con)
    n = con.execute("SELECT count(*) FROM pace_snapshots").fetchone()[0]
    print(f"      {n} Datenpunkte in {db_pfad.name} geschrieben")

    png_pfad = OUT / "live_board.png"
    dashboard_rendern(con, ses, png_pfad)
    con.close()
    print(f"      -> {png_pfad}")


if __name__ == "__main__":
    main()
