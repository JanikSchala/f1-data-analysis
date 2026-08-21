"""lädt f1-sessions aus dem cache und zieht bei bedarf eine ganze saison hinein"""
from __future__ import annotations

import csv
import shutil
import tempfile
import time
from datetime import datetime
from pathlib import Path

import fastf1
import matplotlib

matplotlib.use("Agg")                      # nur dateien statt fenster

import matplotlib.pyplot as plt
import pandas as pd
from fastf1.exceptions import RateLimitExceededError

import f1lab
from f1lab.design import FG, GRID, MUTED, NEGATIV, POSITIV, SERIEN, matplotlib_stil

OUT = Path(__file__).parent / "out"
OUT.mkdir(exist_ok=True)

CACHE = f1lab.enable_cache()

CACHE_LOG = Path(__file__).parent / "cache_warmup_log.csv"
LOG_FIELDS = ["year", "round", "event", "ident", "status", "duration_s", "error"]
DEFAULT_SESSIONS = ("FP1", "FP2", "FP3", "Q", "S", "SQ", "R")

plt.rcParams.update(matplotlib_stil())


def load_session(year, gp, ident, **kw):
    t0 = time.perf_counter()
    ses = fastf1.get_session(year, gp, ident)
    ses.load(**kw)
    print(f"{year} {gp} {ident} geladen in {time.perf_counter()-t0:.1f}s")
    return ses


# laedt standardmaessig auch telemetrie mit
race = load_session(2024, "Monza", "R")

print("Event:      ", race.event["EventName"])
print("Datum:      ", race.event["EventDate"].date())
print("Runden:     ", race.total_laps)
print("Fahrer:     ", len(race.drivers))
print("Lap-Zeilen: ", len(race.laps))

# nur timing ist deutlich schneller als telemetrie mitzuladen
quali = load_session(2024, "Monza", "Q",
                     telemetry=False, weather=False, messages=False)

print(quali.results[["Abbreviation", "TeamName", "Position", "Q1", "Q2", "Q3"]])


def demo_ladezeiten(year: int = 2023, gp: str = "Monza",
                    ident: str = "Q") -> None:
    """misst kaltstart gegen cache-treffer in einem temporaeren cache-ordner.
    der hauptcache bleibt dabei unberuehrt. bricht ohne netzzugriff sauber ab
    statt das ganze skript zu stoppen."""
    tmp = Path(tempfile.mkdtemp(prefix="f1_cache_demo_"))
    try:
        fastf1.Cache.enable_cache(str(tmp))

        t0 = time.perf_counter()
        fastf1.get_session(year, gp, ident).load(
            telemetry=False, weather=False, messages=False)
        kalt = time.perf_counter() - t0

        t0 = time.perf_counter()
        fastf1.get_session(year, gp, ident).load(
            telemetry=False, weather=False, messages=False)
        warm = time.perf_counter() - t0

        print(f"\nLadezeiten {gp} {year} {ident} (frischer, isolierter Cache):")
        print(f"  Kaltstart (Netz):  {kalt:5.2f}s")
        print(f"  Aus dem Cache:     {warm:5.2f}s")
        print(f"  Beschleunigung:    {kalt / warm:5.1f}x")
    except Exception as exc:
        print(f"\nLadezeiten-Demo uebersprungen (kein Netz oder Rate-Limit): "
             f"{type(exc).__name__}: {exc}")
    finally:
        f1lab.enable_cache(CACHE)             # zurueck auf den hauptcache
        shutil.rmtree(tmp, ignore_errors=True)


demo_ladezeiten()


def _log_row(path, row):
    """haengt einen datensatz an die csv an. so bleibt der fortschritt auch
    bei abbruch erhalten."""
    is_new = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=LOG_FIELDS)
        if is_new:
            w.writeheader()
        w.writerow(row)


def _with_retry(fn, *args, base_wait=30, max_wait=600, **kwargs):
    """faengt fastf1s RateLimitExceededError ab und wiederholt den call mit
    wachsender wartezeit statt fixer sleeps."""
    wait = base_wait
    while True:
        try:
            return fn(*args, **kwargs)
        except RateLimitExceededError:
            print(f"    Rate-Limit erreicht, warte {wait}s ...")
            time.sleep(wait)
            wait = min(wait * 2, max_wait)


def _done_set(log_path):
    """(year, round, ident)-kombinationen die laut log bereits erfolgreich
    geladen wurden."""
    if not log_path.exists():
        return set()
    log = pd.read_csv(log_path)
    ok = log[log["status"] == "ok"]
    return set(zip(ok["year"], ok["round"], ok["ident"]))


def warm_season_cache(year, sessions=DEFAULT_SESSIONS, log_path=CACHE_LOG,
                      telemetry=False, weather=False, messages=False, **kw):
    """zieht eine ganze saison in den cache. nicht existierende kombinationen
    (z.b. FP2/FP3 an einem sprint-wochenende) werden uebersprungen statt das
    skript abzubrechen. bereits geladene kombinationen werden nicht erneut
    angefragt."""
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
    """zieht jede saison ab start_year jahr fuer jahr in den cache. ein
    fehler in einer saison bricht die anderen nicht ab."""
    end_year = end_year or datetime.now().year
    for yr in range(start_year, end_year + 1):
        print(f"\n=== Saison {yr} ===")
        try:
            warm_season_cache(yr, sessions=sessions, log_path=log_path, **kw)
        except Exception as exc:
            print(f"Saison {yr} abgebrochen: {exc}")


# gruen = gelungen. grau = uebersprungen. orange = misslungen.
STATUS_FARBE = {"ok": POSITIV, "skipped": MUTED, "failed": NEGATIV}


def plot_cache_warmup(log_path=CACHE_LOG):
    """visualisiert cache-abdeckung je saison und ladezeit je session-typ.
    laesst sich auch aufrufen waehrend warm_all_seasons() noch laeuft."""
    if not log_path.exists():
        print(f"Noch kein Log unter {log_path} - erst warm_season_cache() "
              f"oder warm_all_seasons() laufen lassen.")
        return

    log = pd.read_csv(log_path)

    fig, ax = plt.subplots(1, 2, figsize=(13, 5.5))

    by_year = log.groupby(["year", "status"]).size().unstack(fill_value=0)
    by_year = by_year[[c for c in STATUS_FARBE if c in by_year.columns]]
    by_year.plot(kind="bar", stacked=True, ax=ax[0],
                color=[STATUS_FARBE[c] for c in by_year.columns], width=0.7)
    ax[0].set_xlabel("Saison")
    ax[0].set_ylabel("Sessions")
    ax[0].set_title("Cache-Abdeckung je Saison", loc="left", color=FG,
                    fontsize=13, pad=12)
    ax[0].legend(frameon=False, labelcolor=FG)
    ax[0].grid(axis="y", alpha=0.35, linewidth=0.8, color=GRID)
    ax[0].set_axisbelow(True)
    ax[0].tick_params(axis="x", rotation=0)

    # median statt mittelwert. ein einzelner rate-limit-retry zaehlt seine
    # wartezeit mit in die gemessene dauer und wuerde den mittelwert bei
    # den seltenen session-typen (SQ, S) stark verzerren.
    ok = log[log["status"] == "ok"]
    med_dur = ok.groupby("ident")["duration_s"].median().reindex(DEFAULT_SESSIONS).dropna()
    ax[1].bar(med_dur.index, med_dur.to_numpy(), color=SERIEN[0])
    ax[1].set_xlabel("Session-Typ")
    ax[1].set_ylabel("Mediane Ladezeit [s]")
    ax[1].set_title(f"Ladezeit nach Session-Typ (n={len(ok)} geladen)",
                    loc="left", color=FG, fontsize=13, pad=12)
    ax[1].grid(axis="y", alpha=0.35, linewidth=0.8, color=GRID)
    ax[1].set_axisbelow(True)

    for a in ax:
        for side in ("top", "right"):
            a.spines[side].set_visible(False)

    fig.suptitle(f"Cache-Warmup {int(log['year'].min())}-{int(log['year'].max())} "
                f"({(log['status'] == 'ok').sum()} Sessions geladen)", x=0.125,
                ha="left", color=FG, fontsize=16, y=1.02)
    plt.tight_layout()
    path = OUT / "cache_warmup.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"\n      -> {path}")


if __name__ == "__main__":
    warm_all_seasons()
    plot_cache_warmup()
