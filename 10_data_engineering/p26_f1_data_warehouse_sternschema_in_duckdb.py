"""
P26 - F1-Data-Warehouse: Sternschema in DuckDB
==============================================

Eine echte ETL-Pipeline: FastF1 -> Parquet -> DuckDB mit Fakten- und Dimensionstabellen, idempotent und inkrementell.

Kategorie:   Data Engineering
Niveau:      Profi
Aufwand:     8-10 h
Schwerpunkt: Engineering
Zusaetzliche Pakete: duckdb, pyarrow

WARUM DAS LOHNT
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

AUSBAUSTUFE  [umgesetzt]
Haenge dbt davor und definiere die Transformationen als Modelle mit Tests -
dann hast du eine Pipeline, die man wirklich deployen wuerde.

VORGEHEN 1 versprach fuenf Tabellen, gebaut wurde nur fact_lap. Vier fehlten
komplett: dim_event (wiederverwendet aus f1lab.event_dimension() - dieselbe
Funktion, die P02 und die Kalender-Dashboardseite fuellt, nicht noch einmal
neu geschrieben), dim_driver und dim_team (aus session.results extrahiert,
kein zusaetzlicher API-Call - die Ergebnisse liegen beim Laden der Runden
ohnehin vor) und fact_pitstop (neu: PitInTime steht auf der Runde VOR der
Box, PitOutTime auf der Runde DANACH - ein Stopp ist deshalb ein Paar
aufeinanderfolgender Zeilen desselben Fahrers, keine einzelne). DurationS
ist damit die volle Boxengassenzeit (~20-25s), nicht die reine Standzeit
(~2s) - derselbe Unterschied, den P16 schon bei Ergasts duration-Feld
gefunden hat, hier nur aus Timing-Daten statt der Ergast-API.

VORGEHEN 5 nannte zwei Analysen, die Vorlage baute nur eine (Degradation je
Compound/Strecke). Pace-Ranking ergaenzt: bereinigte Median-Rundenzeit je
Fahrer und Event, ueber dim_driver auf den vollen Namen gejoint. Rohe
Sekunden ueber ein ganzes Jahr zu mitteln waere aber blind fuer
Streckenlaenge (Monaco ~75s Runden, Spa ~105s) - mart_driver_pace fuehrt
deshalb eine "rel_pace"-Spalte (Anteil ueber der schnellsten Median-Runde
desselben Events), erst darueber ist eine saisonweite Rangliste sinnvoll.

AUSBAUSTUFE: kein echtes dbt (nicht in requirements.txt, und ein echtes
dbt-Projekt bringt eigene Projektstruktur, Adapter-Konfiguration und
Macro-Sprache mit, die hier den Rahmen sprengen wuerde) - stattdessen eine
minimale Umsetzung derselben Idee: SQL-Modelldateien unter models/, ein
kleiner Runner, der sie in Abhaengigkeitsreihenfolge als Views anlegt (per
Namenskonvention stg_ vor mart_ - echtes dbt loest das ueber {{ ref() }} und
einen echten DAG, hier nur simuliert), und Tests als Assertion-Queries
(erwartete Zeilenzahl 0), die vor jedem Report laufen und bei Verstoss laut
scheitern statt still falsche Zahlen weiterzureichen.

Die Tests haben beim ersten Lauf sofort einen echten Bug gefangen, nicht nur
demonstriert: "Deleted" ist wie in f1lab.session.not_deleted_mask() bereits
bekannt eine nullable Spalte (fehlender Wert = nicht gestrichen), aber die
SQL-Bedingung "NOT Deleted" verwirft unter SQLs Dreiwertlogik jede Zeile mit
Deleted=NULL statt sie zu behalten - stg_fact_lap_clean war dadurch komplett
leer. Uebler: jeder einzelne der urspruenglichen Tests bestand trotzdem
("0 Zeilen verletzen die Regel" ist auf einer leeren Tabelle immer wahr),
bis ein expliziter Mindestgroessen-Test ergaenzt wurde. Gefixt durch
Wiederverwendung von f1lab.not_deleted_mask() direkt bei der Extraktion,
statt das NULL-Verhalten in SQL zu wiederholen.
"""
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

for tabelle in ("fact_lap", "fact_pitstop", "dim_event", "dim_driver", "dim_team"):
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
    """fact_pitstop: PitInTime steht auf der Runde vor der Box, PitOutTime
    auf der direkt folgenden - ein Stopp ist ein Paar aufeinanderfolgender
    Zeilen desselben Fahrers, siehe Docstring."""
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


def event_laden(season: int, rnd: int, sess: str):
    con = verbindung()
    if bereits_geladen(con, season, rnd, sess):
        print(f"      skip {season}-{rnd}-{sess} (schon geladen)")
        con.close()
        return None

    s = f1lab.load(season, rnd, sess, telemetry=False, weather=False, messages=False)
    laps = s.laps.copy()
    for c in ("LapTime", "Sector1Time", "Sector2Time", "Sector3Time"):
        laps[c + "_s"] = laps[c].dt.total_seconds()

    keep = ["Driver", "DriverNumber", "Team", "LapNumber", "Stint",
           "Compound", "TyreLife", "FreshTyre", "TrackStatus", "Position",
           "IsAccurate", "SpeedST", "SpeedFL",
           "LapTime_s", "Sector1Time_s", "Sector2Time_s", "Sector3Time_s"]
    fact_lap = laps[keep].copy()
    # "Deleted" ist nullable: fehlender Wert heisst nicht gestrichen, nicht
    # unbekannt (siehe f1lab.session.not_deleted_mask()). Als sauberes,
    # nicht-nullable Bool in den Fakt schreiben statt das NULL-Verhalten in
    # jedem SQL-Modell neu falsch zu machen - genau das ist beim ersten Lauf
    # dieses Skripts passiert: "WHERE NOT Deleted" verwarf via SQL-Dreiwertlogik
    # jede Runde mit Deleted=NULL, stg_fact_lap_clean war leer.
    fact_lap["Deleted"] = ~f1lab.not_deleted_mask(laps["Deleted"]).to_numpy()
    fact_lap["Season"], fact_lap["Round"], fact_lap["Session"] = season, rnd, sess
    fact_lap["EventName"] = s.event["EventName"]

    fact_pitstop = pitstops_extrahieren(laps)
    fact_pitstop["Season"], fact_pitstop["Round"], fact_pitstop["Session"] = season, rnd, sess

    lap_pfad = WH / "fact_lap" / f"season={season}" / f"round={rnd}_{sess}.parquet"
    lap_pfad.parent.mkdir(parents=True, exist_ok=True)
    fact_lap.to_parquet(lap_pfad, index=False)

    pit_pfad = WH / "fact_pitstop" / f"season={season}" / f"round={rnd}_{sess}.parquet"
    pit_pfad.parent.mkdir(parents=True, exist_ok=True)
    fact_pitstop.to_parquet(pit_pfad, index=False)

    con.execute("INSERT INTO load_manifest VALUES (?,?,?,now(),?,?)",
               [season, rnd, sess, len(fact_lap), len(fact_pitstop)])
    con.close()
    print(f"      geladen {season}-{rnd}-{sess}: {len(fact_lap)} Runden, "
         f"{len(fact_pitstop)} Boxenstopps")

    res = s.results[["Abbreviation", "DriverNumber", "FullName", "CountryCode",
                     "TeamName"]].copy()
    return res


def dims_schreiben(season: int, driver_frames: list[pd.DataFrame]) -> None:
    """dim_driver/dim_team: kleine Tabellen, bei jedem Lauf komplett neu
    geschrieben (kein Manifest noetig - Neuberechnung ist billig genug,
    um Idempotenz durch schlichtes Ueberschreiben zu bekommen)."""
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

    # dim_event behaelt f1lab.event_dimension()s eigene (kleingeschriebene)
    # Spaltennamen bei, statt sie der PascalCase-Konvention der anderen
    # Tabellen anzupassen - dieselbe Funktion liefert P02 und dem Kalender-
    # Dashboard identische Spalten; zwei Schreibweisen fuer dieselbe
    # Funktion waeren die schlechtere Inkonsistenz.
    dim_event = f1lab.event_dimension([season])
    dim_event.to_parquet(WH / "dim_event" / f"season={season}.parquet", index=False)


def views_anlegen(con) -> None:
    """VORGEHEN 4."""
    for tabelle in ("fact_lap", "fact_pitstop"):
        con.execute(f"""CREATE OR REPLACE VIEW {tabelle} AS
            SELECT * FROM read_parquet('{WH}/{tabelle}/*/*.parquet',
                                       hive_partitioning=1)""")
    for tabelle in ("dim_event", "dim_driver", "dim_team"):
        con.execute(f"""CREATE OR REPLACE VIEW {tabelle} AS
            SELECT * FROM read_parquet('{WH}/{tabelle}/*.parquet')""")


def modelle_ausfuehren(con) -> list[str]:
    """AUSBAUSTUFE: .sql-Dateien unter models/ als Views anlegen, stg_ vor
    mart_ (Namenskonvention statt echtem DAG, siehe Docstring)."""
    dateien = sorted(MODELS_DIR.glob("*.sql"),
                     key=lambda p: (not p.stem.startswith("stg_"), p.stem))
    namen = []
    for pfad in dateien:
        sql = pfad.read_text()
        con.execute(f"CREATE OR REPLACE VIEW {pfad.stem} AS {sql}")
        namen.append(pfad.stem)
    return namen


def tests_ausfuehren(con) -> None:
    """AUSBAUSTUFE: Assertion-Queries statt dbt-YAML - jede muss 0 liefern.

    "nicht_leer" ist kein Randfall: beim ersten Lauf war stg_fact_lap_clean
    durch den NULL-Bug bei Deleted (siehe Docstring oben) komplett leer, und
    JEDER der anderen Tests bestand trotzdem - "0 Zeilen verletzen die
    Regel" ist auf einer leeren Tabelle immer wahr. Ohne einen expliziten
    Mindestgroessen-Test haetten die "gruenen" Tests eine kaputte Pipeline
    stillschweigend als korrekt bestaetigt.
    """
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
    n_laps, n_pits = con.execute(
        "SELECT (SELECT count(*) FROM fact_lap), (SELECT count(*) FROM fact_pitstop)"
    ).fetchone()
    print(f"      fact_lap: {n_laps} Zeilen, fact_pitstop: {n_pits} Zeilen")

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
    # DurationS ist PitOutTime-PitInTime, also die volle Boxengassenzeit
    # (~20-25s), nicht die reine Standzeit (~2s) - siehe P16, derselbe
    # Unterschied. Filter entsprechend, nicht auf die dort schon widerlegte
    # 1.5-6s-Annahme.
    print(con.execute("""
        SELECT Team, count(*) AS stopps, round(median(DurationS), 2) AS median_s
        FROM fact_pitstop
        WHERE DurationS BETWEEN 15 AND 40
        GROUP BY 1
        ORDER BY median_s
    """).df().to_string(index=False))

    con.close()


if __name__ == "__main__":
    main()
