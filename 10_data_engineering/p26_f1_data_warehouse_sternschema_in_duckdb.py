"""baut ein DuckDB data warehouse mit fakten- und dimensionstabellen aus F1-sessions"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import duckdb
import fastf1
import pandas as pd

import f1lab

warnings.filterwarnings("ignore")
fastf1.set_log_level("ERROR")

WH = Path(__file__).parent / "f1_warehouse"
MODELS_DIR = Path(__file__).parent / "models"
SEASON = 2024
SESSION_IDENT = "R"

for tabelle in ("fact_lap", "fact_pitstop", "fact_overtake", "dim_event",
               "dim_driver", "dim_team"):
    (WH / tabelle).mkdir(parents=True, exist_ok=True)


def verbindung() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect(str(WH / "f1.duckdb"))
    con.execute("""CREATE TABLE IF NOT EXISTS load_manifest(
        season INT, round INT, session VARCHAR, loaded_at TIMESTAMP,
        lap_rows BIGINT, pitstop_rows BIGINT,
        PRIMARY KEY(season, round, session))""")
    return con


def bereits_geladen(con, season: int, rnd: int, sess: str) -> bool:
    return con.execute(
        "SELECT count(*) FROM load_manifest WHERE season=? AND round=? AND session=?",
        [season, rnd, sess]).fetchone()[0] > 0


def pitstops_extrahieren(laps: pd.DataFrame) -> pd.DataFrame:
    """fact_pitstop: PitInTime steht auf der runde vor der box.
    PitOutTime steht auf der direkt folgenden runde. ein stopp ist deshalb
    ein paar aufeinanderfolgender zeilen desselben fahrers."""
    zeilen = []
    ein = laps[laps["PitInTime"].notna()]
    for r in ein.itertuples():
        aus = laps[(laps["Driver"] == r.Driver) & (laps["LapNumber"] == r.LapNumber + 1)]
        pit_out = aus.iloc[0]["PitOutTime"] if len(aus) else pd.NaT
        dauer = ((pit_out - r.PitInTime).total_seconds()
                if pd.notna(pit_out) and pd.notna(r.PitInTime) else None)
        zeilen.append({
            "Driver": r.Driver, "Team": r.Team, "InLap": r.LapNumber,
            "StintVor": r.Stint, "CompoundVor": r.Compound,
            "CompoundNach": (aus.iloc[0]["Compound"] if len(aus) else None),
            "DurationS": dauer,
        })
    return pd.DataFrame(zeilen)


def overtakes_extrahieren(session, season: int, rnd: int, sess: str) -> pd.DataFrame:
    """fact_overtake: ein datensatz je ueberholvorgang aus
    f1lab.overtake_events(). braucht wie fact_lap/fact_pitstop keine telemetrie."""
    events = f1lab.overtake_events(session)
    events = events.rename(columns={"gainer": "Gainer", "loser": "Loser",
                                    "lap": "Lap"})
    events["Season"], events["Round"], events["Session"] = season, rnd, sess
    return events


def fact_overtake_pfad(season: int, rnd: int, sess: str) -> Path:
    return WH / "fact_overtake" / f"season={season}" / f"round={rnd}_{sess}.parquet"


def event_laden(season: int, rnd: int, sess: str):
    con = verbindung()
    schon_geladen = bereits_geladen(con, season, rnd, sess)
    ov_pfad = fact_overtake_pfad(season, rnd, sess)
    if schon_geladen and ov_pfad.exists():
        print(f"      skip {season}-{rnd}-{sess} (schon geladen)")
        con.close()
        return None
    con.close()

    s = f1lab.load(season, rnd, sess, telemetry=False, weather=False, messages=False)

    if schon_geladen:
        # fact_overtake kam spaeter dazu als fact_lap/fact_pitstop.
        # fuer laengst geladene runden wird nur das fehlende nachgezogen.
        # das laeuft ueber denselben lokal gecachten FastF1-load ohne netzzugriff.
        events = overtakes_extrahieren(s, season, rnd, sess)
        ov_pfad.parent.mkdir(parents=True, exist_ok=True)
        events.to_parquet(ov_pfad, index=False)
        print(f"      nachgezogen {season}-{rnd}-{sess}: "
             f"{len(events)} Ueberholungen (fact_overtake)")
        return None
    laps = s.laps.copy()
    for c in ("LapTime", "Sector1Time", "Sector2Time", "Sector3Time"):
        laps[c + "_s"] = laps[c].dt.total_seconds()

    keep = ["Driver", "DriverNumber", "Team", "LapNumber", "Stint",
           "Compound", "TyreLife", "FreshTyre", "TrackStatus", "Position",
           "IsAccurate", "SpeedST", "SpeedFL",
           "LapTime_s", "Sector1Time_s", "Sector2Time_s", "Sector3Time_s"]
    fact_lap = laps[keep].copy()
    # Deleted ist nullable. ein fehlender wert heisst nicht gestrichen.
    # als non-nullable bool schreiben. sonst stolpert SQL bei NULL ueber die dreiwertlogik.
    fact_lap["Deleted"] = ~f1lab.not_deleted_mask(laps["Deleted"]).to_numpy()
    fact_lap["Season"], fact_lap["Round"], fact_lap["Session"] = season, rnd, sess
    fact_lap["EventName"] = s.event["EventName"]

    fact_pitstop = pitstops_extrahieren(laps)
    fact_pitstop["Season"], fact_pitstop["Round"], fact_pitstop["Session"] = season, rnd, sess

    fact_overtake = overtakes_extrahieren(s, season, rnd, sess)

    lap_pfad = WH / "fact_lap" / f"season={season}" / f"round={rnd}_{sess}.parquet"
    lap_pfad.parent.mkdir(parents=True, exist_ok=True)
    fact_lap.to_parquet(lap_pfad, index=False)

    pit_pfad = WH / "fact_pitstop" / f"season={season}" / f"round={rnd}_{sess}.parquet"
    pit_pfad.parent.mkdir(parents=True, exist_ok=True)
    fact_pitstop.to_parquet(pit_pfad, index=False)

    ov_pfad.parent.mkdir(parents=True, exist_ok=True)
    fact_overtake.to_parquet(ov_pfad, index=False)

    con = verbindung()
    con.execute("INSERT INTO load_manifest VALUES (?,?,?,now(),?,?)",
               [season, rnd, sess, len(fact_lap), len(fact_pitstop)])
    con.close()
    print(f"      geladen {season}-{rnd}-{sess}: {len(fact_lap)} Runden, "
         f"{len(fact_pitstop)} Boxenstopps, {len(fact_overtake)} Ueberholungen")

    res = s.results[["Abbreviation", "DriverNumber", "FullName", "CountryCode",
                     "TeamName"]].copy()
    return res


def dims_schreiben(season: int, driver_frames: list[pd.DataFrame]) -> None:
    """dim_driver/dim_team: kleine tabellen. bei jedem lauf komplett neu
    geschrieben statt ueber ein manifest verwaltet. neuberechnung ist billig genug."""
    alle = pd.concat(driver_frames, ignore_index=True).drop_duplicates(
        subset=["Abbreviation"])

    dim_driver = alle.rename(columns={"Abbreviation": "Driver"})[
        ["Driver", "DriverNumber", "FullName", "CountryCode"]]
    dim_driver["Season"] = season
    (WH / "dim_driver" / f"season={season}.parquet").parent.mkdir(
        parents=True, exist_ok=True)
    dim_driver.to_parquet(WH / "dim_driver" / f"season={season}.parquet",
                          index=False)

    dim_team = pd.DataFrame({"Team": alle["TeamName"].dropna().unique()})
    dim_team["Season"] = season
    dim_team.to_parquet(WH / "dim_team" / f"season={season}.parquet", index=False)

    # dim_event behaelt event_dimension()s eigene kleingeschriebene spaltennamen.
    # dieselbe funktion liefert P02 und dem kalender-dashboard identische spalten.
    # zwei schreibweisen fuer dieselbe funktion waeren die schlechtere inkonsistenz.
    dim_event = f1lab.event_dimension([season])
    dim_event.to_parquet(WH / "dim_event" / f"season={season}.parquet", index=False)


def views_anlegen(con) -> None:
    for tabelle in ("fact_lap", "fact_pitstop", "fact_overtake"):
        con.execute(f"""CREATE OR REPLACE VIEW {tabelle} AS
            SELECT * FROM read_parquet('{WH}/{tabelle}/*/*.parquet',
                                       hive_partitioning=1)""")
    for tabelle in ("dim_event", "dim_driver", "dim_team"):
        con.execute(f"""CREATE OR REPLACE VIEW {tabelle} AS
            SELECT * FROM read_parquet('{WH}/{tabelle}/*.parquet')""")


def modelle_ausfuehren(con) -> list[str]:
    """legt .sql-Dateien unter models/ als views an. reihenfolge stg_ vor mart_.
    namenskonvention statt echtem DAG."""
    dateien = sorted(MODELS_DIR.glob("*.sql"),
                     key=lambda p: (not p.stem.startswith("stg_"), p.stem))
    namen = []
    for pfad in dateien:
        sql = pfad.read_text()
        con.execute(f"CREATE OR REPLACE VIEW {pfad.stem} AS {sql}")
        namen.append(pfad.stem)
    return namen


def tests_ausfuehren(con) -> None:
    """assertion-queries statt dbt-yaml. jede muss 0 liefern.

    "nicht_leer" ist kein randfall. eine leere tabelle erfuellt jede
    "0 zeilen verletzen die regel"-pruefung automatisch. ohne diesen
    test wuerde eine kaputte pipeline unbemerkt bleiben."""
    tests = {
        "stg_fact_lap_clean_nicht_leer":
            "SELECT CASE WHEN count(*) < 1000 THEN 1 ELSE 0 END "
            "FROM stg_fact_lap_clean",
        "keine_negativen_rundenzeiten":
            "SELECT count(*) FROM stg_fact_lap_clean WHERE LapTime_s <= 0",
        "keine_nullwerte_im_pace_ranking":
            "SELECT count(*) FROM mart_driver_pace WHERE median_pace_s IS NULL",
        "jeder_boxenstopp_hat_ein_team":
            "SELECT count(*) FROM fact_pitstop WHERE Team IS NULL",
        "degradation_nur_bekannte_mischungen":
            "SELECT count(*) FROM mart_degradation WHERE Compound = 'UNKNOWN'",
        "jede_ueberholung_hat_gainer_und_loser":
            "SELECT count(*) FROM fact_overtake "
            "WHERE Gainer IS NULL OR Loser IS NULL OR Gainer = Loser",
    }
    for name, sql in tests.items():
        n = con.execute(sql).fetchone()[0]
        status = "OK" if n == 0 else f"FEHLGESCHLAGEN ({n} Zeilen)"
        print(f"      [{'✓' if n == 0 else '✗'}] {name}: {status}")


def main():
    f1lab.enable_cache()
    con = verbindung()

    print(f"[1/4] Saison {SEASON}, Session {SESSION_IDENT}: Extraktion "
         f"(VORGEHEN 1-3) ...")
    driver_frames = []
    for rnd in range(1, 25):
        try:
            res = event_laden(SEASON, rnd, SESSION_IDENT)
            if res is not None:
                driver_frames.append(res)
        except Exception as exc:
            print(f"      R{rnd} Fehler: {exc}")
    con.close()

    if driver_frames:
        print("\n      dim_driver/dim_team/dim_event schreiben ...")
        dims_schreiben(SEASON, driver_frames)
    else:
        print("\n      keine neuen Sessions - dim_driver/dim_team/dim_event "
             "nur neu schreiben, wenn dieser Lauf tatsaechlich geladen hat "
             "(sie haengen an den geladenen Ergebnissen).")

    con = verbindung()
    print("\n[2/4] DuckDB-Views ueber die Parquet-Partitionen (VORGEHEN 4) ...")
    views_anlegen(con)
    n_laps, n_pits, n_ov = con.execute(
        "SELECT (SELECT count(*) FROM fact_lap), (SELECT count(*) FROM fact_pitstop), "
        "(SELECT count(*) FROM fact_overtake)"
    ).fetchone()
    print(f"      fact_lap: {n_laps} Zeilen, fact_pitstop: {n_pits} Zeilen, "
         f"fact_overtake: {n_ov} Zeilen")

    print("\n[3/4] dbt-Ersatz: Modelle und Tests (AUSBAUSTUFE) ...")
    modelle = modelle_ausfuehren(con)
    print(f"      Modelle angelegt: {modelle}")
    tests_ausfuehren(con)

    print("\n[4/4] Analytische Queries (VORGEHEN 5) ...")
    print("\n      Pace-Ranking (relative Pace ueber die Saison, 1.000 = "
         "an diesem Event schnellster Median):")
    print(con.execute("""
        SELECT d.FullName, count(*) AS events,
               round(avg(p.rel_pace), 4) AS mittlere_rel_pace
        FROM mart_driver_pace p
        JOIN dim_driver d ON d.Driver = p.Driver AND d.Season = p.Season
        GROUP BY 1
        HAVING count(*) >= 10
        ORDER BY mittlere_rel_pace
        LIMIT 10
    """).df().to_string(index=False))

    print("\n      Degradation je Compound und Strecke (Median, s):")
    print(con.execute("""
        SELECT * FROM mart_degradation
        ORDER BY EventName, median_s
        LIMIT 20
    """).df().to_string(index=False))

    print("\n      Boxenstopp-Ranking je Team (Median-Dauer, s):")
    # DurationS ist PitOutTime minus PitInTime. das ist die volle boxengassenzeit.
    # nicht die reine standzeit. filterbereich entsprechend weiter gefasst.
    print(con.execute("""
        SELECT Team, count(*) AS stopps, round(median(DurationS), 2) AS median_s
        FROM fact_pitstop
        WHERE DurationS BETWEEN 15 AND 40
        GROUP BY 1
        ORDER BY median_s
    """).df().to_string(index=False))

    print("\n      Ueberholungen je Rennen (P38/P39-Baustein, "
         "Nachtrag 2026-08-17):")
    print(con.execute("""
        SELECT * FROM mart_overtakes_by_event
        ORDER BY ueberholungen DESC
        LIMIT 10
    """).df().to_string(index=False))

    con.close()


if __name__ == "__main__":
    main()
