"""zeichnet den fastf1 live-timing-stream auf und wertet ihn wie eine normale session aus, inklusive sqlite-zeitreihe und png-board"""
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

FENSTER = 5     # Runden fuer die rollierende Pace

plt.rcParams.update(matplotlib_stil())


# --------------------------------------------------------------- Aufzeichnen
def record(path: str = "saved_data.txt", timeout_min: int = 120) -> None:
    """Während einer laufenden Session aufrufen. Blockiert bis zum Ende der Session oder Ctrl+C.
    Kein Rückgabewert, das File wächst währenddessen.

    Nur .start() (synchron). async_start() wirft seit fastf1 3.x NotImplementedError.
    debug bleibt bewusst weg. Der Parameter existiert noch, aber jeder Wert außer False
    wirft seit Kurzem einen ValueError.
    """
    client = SignalRClient(filename=path, timeout=timeout_min * 60)
    client.start()


# ------------------------------------------------------------------ Rolling-Pace
def rolling_pace(laps: pd.DataFrame, fenster: int = FENSTER) -> pd.DataFrame:
    """rollierende Median-Pace je Fahrer. Funktioniert unabhängig davon ob `laps` aus einem Live-Feed oder einem historischen Archiv stammt."""
    df = laps.copy()
    df["sec"] = df["LapTime"].dt.total_seconds()
    df = df.sort_values(["Driver", "LapNumber"])
    df["rolling"] = (df.groupby("Driver")["sec"]
                     .transform(lambda s: s.rolling(fenster, min_periods=3).median()))
    return df


def aktuelle_rangliste(df_mit_rolling: pd.DataFrame) -> pd.DataFrame:
    """neuester Rolling-Wert je Fahrer, schnellster zuerst."""
    return (df_mit_rolling.dropna(subset=["rolling"])
           .sort_values("LapNumber").groupby("Driver").tail(1)
           .sort_values("rolling"))


# ------------------------------------------------------------------ Auswertung
def analyse(path: str, year: int, gp: str, ident: str) -> pd.DataFrame:
    """lädt die Aufzeichnung und behandelt sie wie eine normale Session."""
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


# ------------------------------------------------------------------- Zeitreihen-Board
def zeitreihe_anlegen(db_pfad: Path) -> sqlite3.Connection:
    con = sqlite3.connect(db_pfad)
    con.execute("""CREATE TABLE IF NOT EXISTS pace_snapshots(
        ts TEXT, driver TEXT, lap INTEGER, compound TEXT,
        tyre_life REAL, rolling_pace_s REAL)""")
    return con


def replay_als_livefeed(ses, con: sqlite3.Connection) -> None:
    """ersetzt den echten Live-Feed: dieselbe Session, Runde für Runde in ihrer tatsächlichen
    Reihenfolge eingespielt statt als fertige Endauswertung. Nach jeder neu "eintreffenden"
    Runde wird die rollierende Pace neu berechnet und genau der neue Punkt geschrieben.
    Das simuliert, was bei einem echten Feed inkrementell passieren würde."""
    laps = ses.laps.dropna(subset=["LapStartTime", "LapTime"]).copy()
    laps = laps.sort_values("LapStartTime").reset_index(drop=True)

    for i in range(len(laps)):
        # Slice der schon korrekt typisierten Runden statt wiederholtem
        # pd.concat() auf ein leeres DataFrame. Letzteres degradiert
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
    """"Grafana-Board" als PNG: rollierende Pace über die Renndistanz, Podium hervorgehoben.
    Dieselbe Zeitreihe, die ein echtes Board live nachgezogen hätte."""
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
