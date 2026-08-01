"""
P26 - F1-Data-Warehouse: Sternschema in DuckDB
==============================================

Eine echte ETL-Pipeline: FastF1 -> Parquet -> DuckDB mit Fakten- und Dimensionstabellen, idempotent und inkrementell.

Kategorie:   Data Engineering
Niveau:      Profi
Aufwand:     8-10 h
Rollen:      ENG
Zusaetzliche Pakete: duckdb, pyarrow

WARUM DAS ZAEHLT
Das staerkste Projekt fuer eine Data-Engineering-Rolle. Zeigt Schema-Design, Idempotenz, inkrementelle Loads und analytisches SQL - alles, was in Produktion zaehlt.

VORGEHEN
  1. Schema entwerfen: dim_event, dim_driver, dim_team, fact_lap, fact_pitstop
  2. Extractor je Session schreiben, der Parquet-Partitionen erzeugt
  3. Idempotenz: bereits geladene Sessions ueberspringen (Manifest-Tabelle)
  4. DuckDB-Views ueber die Parquet-Dateien anlegen
  5. Analytische Queries: Pace-Ranking, Deg je Compound und Strecke

GENUTZTE FASTF1-BAUSTEINE
  - fastf1 gesamt
  - pandas.to_parquet
  - duckdb

AUSBAUSTUFE
Haenge dbt davor und definiere die Transformationen als Modelle mit Tests - dann hast du eine Pipeline, die man wirklich deployen wuerde.
"""

from pathlib import Path
import pandas as pd
import duckdb
import fastf1

fastf1.Cache.enable_cache("~/f1_cache")
WH = Path("f1_warehouse")
(WH / "fact_lap").mkdir(parents=True, exist_ok=True)
con = duckdb.connect(str(WH / "f1.duckdb"))

con.execute("""CREATE TABLE IF NOT EXISTS load_manifest(
    season INT, round INT, session VARCHAR, loaded_at TIMESTAMP,
    rows BIGINT, PRIMARY KEY(season, round, session))""")


def already_loaded(season, rnd, sess):
    return con.execute(
        "SELECT count(*) FROM load_manifest WHERE season=? AND round=? AND session=?",
        [season, rnd, sess]).fetchone()[0] > 0


def extract_laps(season, rnd, sess="R"):
    if already_loaded(season, rnd, sess):
        print(f"skip {season}-{rnd}-{sess}")
        return
    s = fastf1.get_session(season, rnd, sess)
    s.load(telemetry=False, weather=False, messages=False)

    laps = s.laps.copy()
    for c in ["LapTime", "Sector1Time", "Sector2Time", "Sector3Time"]:
        laps[c + "_s"] = laps[c].dt.total_seconds()

    keep = ["Driver", "DriverNumber", "Team", "LapNumber", "Stint",
            "Compound", "TyreLife", "FreshTyre", "TrackStatus", "Position",
            "IsAccurate", "Deleted", "SpeedST", "SpeedFL",
            "LapTime_s", "Sector1Time_s", "Sector2Time_s", "Sector3Time_s"]
    out = laps[keep].copy()
    out["Season"], out["Round"], out["Session"] = season, rnd, sess
    out["EventName"] = s.event["EventName"]

    path = WH / "fact_lap" / f"season={season}" / f"round={rnd}_{sess}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(path, index=False)

    con.execute("INSERT INTO load_manifest VALUES (?,?,?,now(),?)",
                [season, rnd, sess, len(out)])
    print(f"loaded {season}-{rnd}-{sess}: {len(out)} Runden")


for rnd in range(1, 25):
    try:
        extract_laps(2024, rnd, "R")
    except Exception as exc:
        print(f"R{rnd} Fehler: {exc}")

con.execute(f"""CREATE OR REPLACE VIEW fact_lap AS
                SELECT * FROM read_parquet('{WH}/fact_lap/*/*.parquet',
                                           hive_partitioning=1)""")

print(con.execute("""
    SELECT EventName, Compound,
           count(*) AS runden,
           round(median(LapTime_s), 3) AS median_s
    FROM fact_lap
    WHERE IsAccurate AND TrackStatus = '1' AND Compound <> 'UNKNOWN'
    GROUP BY 1, 2
    HAVING count(*) > 50
    ORDER BY EventName, median_s
    LIMIT 25
""").df().to_string(index=False))
