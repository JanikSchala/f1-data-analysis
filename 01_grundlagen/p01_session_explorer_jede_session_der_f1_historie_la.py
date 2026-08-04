"""
P01 - Session-Explorer: Jede Session der F1-Historie laden
==========================================================

Das Fundament. Cache aufsetzen, beliebige Session laden, verstehen was load() eigentlich holt.

Kategorie:   Grundlagen & Datenzugriff
Niveau:      Einsteiger
Aufwand:     1-2 h
Schwerpunkt: Datenanalyse, Engineering

WARUM DAS LOHNT
Ohne sauberes Caching wartest du bei jeder Analyse Minuten. Und wer den Datenfluss einmal durchschaut hat - Live-Timing-API, Parser, DataFrame - weiss bei jedem spaeteren Problem, wo er suchen muss.

VORGEHEN
  1. Cache-Ordner anlegen und global aktivieren
  2. Session ueber (Jahr, GP, Identifier) laden - 'R', 'Q', 'FP1', 'S', 'SQ'
  3. Selektiv laden: laps/telemetry/weather/messages einzeln steuern
  4. session_info, drivers und results inspizieren
  5. Ladezeiten mit und ohne Cache messen

GENUTZTE FASTF1-BAUSTEINE
  - fastf1.Cache.enable_cache
  - fastf1.get_session
  - Session.load
  - Session.session_info

AUSBAUSTUFE
Baue eine Funktion, die eine ganze Saison in einem Rutsch in den Cache zieht und den Fortschritt in der Konsole anzeigt.
"""

import csv
import time
from datetime import datetime
from pathlib import Path

import fastf1
import matplotlib.pyplot as plt
import pandas as pd
from fastf1.exceptions import RateLimitExceededError

CACHE = Path.home() / "f1_cache"
CACHE.mkdir(exist_ok=True)
fastf1.Cache.enable_cache(str(CACHE))

CACHE_LOG = Path(__file__).parent / "cache_warmup_log.csv"
LOG_FIELDS = ["year", "round", "event", "ident", "status", "duration_s", "error"]
DEFAULT_SESSIONS = ("FP1", "FP2", "FP3", "Q", "S", "SQ", "R")

def load_session(year, gp, ident, **kw):
    t0 = time.perf_counter()
    ses = fastf1.get_session(year, gp, ident)
    ses.load(**kw)
    print(f"{year} {gp} {ident} geladen in {time.perf_counter()-t0:.1f}s")
    return ses

# Volles Rennen inkl. Telemetrie
race = load_session(2024, "Monza", "R")

print("Event:      ", race.event["EventName"])
print("Datum:      ", race.event["EventDate"].date())
print("Runden:     ", race.total_laps)
print("Fahrer:     ", len(race.drivers))
print("Lap-Zeilen: ", len(race.laps))

# Nur Timing, ohne Telemetrie -> deutlich schneller
quali = load_session(2024, "Monza", "Q",
                     telemetry=False, weather=False, messages=False)

print(quali.results[["Abbreviation", "TeamName", "Position", "Q1", "Q2", "Q3"]])


def _log_row(path, row):
    """Haengt einen Datensatz an die CSV an - auch bei Abbruch bleibt der
    bisherige Fortschritt erhalten, ein Neustart faengt dank FastF1s
    eigenem Cache nicht von vorne an (bereits geladene Sessions kommen
    beim naechsten Versuch sofort von der Platte statt aus dem Netz)."""
    is_new = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=LOG_FIELDS)
        if is_new:
            w.writeheader()
        w.writerow(row)


def _with_retry(fn, *args, base_wait=30, max_wait=600, **kwargs):
    """FastF1 begrenzt sich selbst auf 500 API-Calls/h (rollierendes
    Fenster) und wirft dann RateLimitExceededError, statt den Server zu
    riskieren. Bei einem mehrstuendigen Warmup ist das kein Fehler,
    sondern der Normalfall - also abwarten und denselben Call erneut
    versuchen, mit wachsender Pause statt fixer Sleeps."""
    wait = base_wait
    while True:
        try:
            return fn(*args, **kwargs)
        except RateLimitExceededError:
            print(f"    Rate-Limit erreicht, warte {wait}s ...")
            time.sleep(wait)
            wait = min(wait * 2, max_wait)


def _done_set(log_path):
    """(year, round, ident)-Kombinationen, die laut Log bereits mit
    status=ok geladen wurden - macht den Warmup neustartsicher, ohne
    Sessions doppelt zu zaehlen oder erneut anzufragen."""
    if not log_path.exists():
        return set()
    log = pd.read_csv(log_path)
    ok = log[log["status"] == "ok"]
    return set(zip(ok["year"], ok["round"], ok["ident"]))


def warm_season_cache(year, sessions=DEFAULT_SESSIONS, log_path=CACHE_LOG,
                      telemetry=False, weather=False, messages=False, **kw):
    """Zieht eine ganze Saison in den Cache, mit Fortschrittsanzeige.

    Laeuft einmal durch den Kalender und laedt jede angegebene Session-
    Kennung fuer jedes Event. Nicht existierende Kombinationen (z.B.
    FP2/FP3 an einem Sprint-Wochenende, oder S/SQ an einem konventionellen)
    werden uebersprungen statt das Skript abzubrechen. Standardmaessig nur
    Timing/Ergebnisse (siehe VORGEHEN Punkt 3: selektiv laden ist deutlich
    schneller und schont das Rate-Limit) - volle Telemetrie geht per
    telemetry=True. Bereits erfolgreich geladene Kombinationen aus einem
    frueheren Lauf werden uebersprungen. Jeder neue Versuch landet als
    Zeile in log_path, damit sich der Fortschritt spaeter auswerten und
    plotten laesst.
    """
    schedule = _with_retry(fastf1.get_event_schedule, year, include_testing=False)
    already = _done_set(log_path)
    total = len(schedule) * len(sessions)
    done, skipped, failed, reused = 0, 0, 0, 0
    t_start = time.perf_counter()

    for _, event in schedule.iterrows():
        for ident in sessions:
            done += 1
            rnd = int(event["RoundNumber"])
            tag = f"[{done}/{total}] {year} {event['EventName']} {ident}"
            if (year, rnd, ident) in already:
                reused += 1
                continue
            row = {"year": year, "round": rnd,
                   "event": event["EventName"], "ident": ident,
                   "status": None, "duration_s": None, "error": ""}
            try:
                ses = _with_retry(fastf1.get_session, year, rnd, ident)
            except Exception:
                skipped += 1
                row["status"] = "skipped"
                print(f"{tag} - keine Session (uebersprungen)")
                _log_row(log_path, row)
                continue
            try:
                t0 = time.perf_counter()
                _with_retry(ses.load, telemetry=telemetry, weather=weather,
                           messages=messages, **kw)
                dt = time.perf_counter() - t0
                row["status"], row["duration_s"] = "ok", round(dt, 2)
                print(f"{tag} - {dt:.1f}s")
            except Exception as exc:
                failed += 1
                row["status"], row["error"] = "failed", f"{type(exc).__name__}: {exc}"
                print(f"{tag} - Fehler: {exc}")
            _log_row(log_path, row)

    elapsed = time.perf_counter() - t_start
    print(f"\nSaison {year} im Cache: {done - skipped - failed - reused} neu geladen, "
          f"{reused} schon vorhanden, {skipped} uebersprungen, {failed} "
          f"fehlgeschlagen, in {elapsed/60:.1f} min")


def warm_all_seasons(start_year=2018, end_year=None, sessions=DEFAULT_SESSIONS,
                      log_path=CACHE_LOG, **kw):
    """Zieht jede Saison der Telemetrie-Aera (ab 2018, siehe README/Grenzen)
    in den Cache - Jahr fuer Jahr, damit ein Fehler in einer Saison nicht
    die anderen mitreisst. Dank _done_set() und dem Retry auf Rate-Limits
    ist ein Neustart (nach Abbruch, oder um die naechste Cache-Traeger zu
    holen) jederzeit gefahrlos - vollstaendig geladene Sessions werden
    nicht erneut angefragt."""
    end_year = end_year or datetime.now().year
    for yr in range(start_year, end_year + 1):
        print(f"\n=== Saison {yr} ===")
        try:
            warm_season_cache(yr, sessions=sessions, log_path=log_path, **kw)
        except Exception as exc:
            print(f"Saison {yr} abgebrochen: {exc}")


def plot_cache_warmup(log_path=CACHE_LOG):
    """Visualisiert den Fortschritt aus warm_all_seasons(): wie viele
    Sessions sind je Saison schon im Cache, und welcher Session-Typ ist
    am teuersten zu laden? Laesst sich jederzeit aufrufen, auch waehrend
    ein warm_all_seasons()-Lauf im Hintergrund noch weiterlaeuft."""
    if not log_path.exists():
        print(f"Noch kein Log unter {log_path} - erst warm_season_cache() "
              f"oder warm_all_seasons() laufen lassen.")
        return

    log = pd.read_csv(log_path)

    fig, ax = plt.subplots(1, 2, figsize=(13, 5.5))

    colors = {"ok": "#2ca02c", "skipped": "#7f7f7f", "failed": "#d62728"}
    by_year = log.groupby(["year", "status"]).size().unstack(fill_value=0)
    by_year = by_year[[c for c in colors if c in by_year.columns]]
    by_year.plot(kind="bar", stacked=True, ax=ax[0],
                color=[colors[c] for c in by_year.columns], width=0.7)
    ax[0].set_xlabel("Saison")
    ax[0].set_ylabel("Sessions")
    ax[0].set_title("Cache-Abdeckung je Saison")
    ax[0].legend()
    ax[0].grid(axis="y", alpha=0.3)

    ok = log[log["status"] == "ok"]
    mean_dur = ok.groupby("ident")["duration_s"].mean().reindex(DEFAULT_SESSIONS).dropna()
    ax[1].bar(mean_dur.index, mean_dur.to_numpy(), color="#1f77b4")
    ax[1].set_xlabel("Session-Typ")
    ax[1].set_ylabel("Mittlere Ladezeit [s]")
    ax[1].set_title(f"Ladezeit nach Session-Typ (n={len(ok)} geladen)")
    ax[1].grid(axis="y", alpha=0.3)

    fig.suptitle(f"Cache-Warmup {int(log['year'].min())}-{int(log['year'].max())} "
                f"({(log['status'] == 'ok').sum()} Sessions geladen)")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    warm_all_seasons()
    plot_cache_warmup()
