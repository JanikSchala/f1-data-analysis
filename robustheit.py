#!/usr/bin/env python3
"""faehrt jedes Analyseskript gegen eine echte Grenzfall-Session.

Die Skripte haben Saison und Event fest oben stehen und laufen deshalb
immer gegen dieselben, gut gefuellten Daten. Was passiert, wenn ein
Skript auf ein Rennen trifft, das es so nicht erwartet, faellt im
Normalbetrieb nie auf - und ist genau die Sorte Fehler, die dieses
Repository wiederholt getroffen hat: KeyError auf einem spaltenlosen
DataFrame, IndexError auf ``.iloc[0]``, Division durch eine leere Menge.

Statt synthetischer Leere wird hier jeder ``f1lab.load()``-Aufruf auf
eine tatsaechlich existierende Session umgeleitet. Nur echte Sessions:
eine ausgedachte leere Session wuerde Fehler melden, die es nie gibt,
und die echten Luecken (halb gefuellte Spalten, fehlende Kanaele)
verfehlen.

Szenarien:

``duenn``
    Spa 2021 R - das Rennen wurde nach zwei Runden hinter dem Safety Car
    gewertet. 60 Runden liegen in den Daten, keine einzige ist als
    persoenliche Bestzeit gewertet, es gibt keine Boxenstopps und kaum
    Race-Control-Meldungen. ``pick_fastest()`` gibt hier ``None``
    zurueck - der haeufigste Ausloeser.

``alt``
    Oesterreich 2018 - die aelteste unterstuetzte Saison. Spalten, die
    heute selbstverstaendlich sind, sind nur teilweise gefuellt: 31 % der
    Stints haben keine Stint-Nummer (2024: 0 %).

``ergast``
    Ergast/jolpica antwortet nicht. Trifft die Historie-Skripte, die
    ihre Daten nicht aus dem Live-Timing-Feed holen.

Bewertet wird nach Absturzart, nicht nach Erfolg: ein Skript darf
abbrechen, wenn es nichts auszuwerten gibt - es muss dann nur sagen,
*dass* nichts da ist. Umfaellt heisst KeyError, IndexError, ValueError
und Verwandte; das sind die Faelle, in denen der Nutzer einen Traceback
sieht statt einer Aussage.

Braucht einen warmen FastF1-Cache mit den Zielsessions und laeuft rund
20 Minuten - deshalb kein Test und nicht Teil der CI. Aufruf::

    python robustheit.py duenn
    python robustheit.py alt 05_reifen_strategie/p13_*.py
"""
from __future__ import annotations

import argparse
import contextlib
import io
import multiprocessing as mp
import runpy
import sys
import traceback
import warnings
from pathlib import Path

REPO = Path(__file__).resolve().parent

# Ein Skript, das hiermit umfaellt, zeigt dem Nutzer einen Traceback
# statt einer Aussage. Alles andere (SystemExit, fachliche Ausnahmen wie
# DataNotLoadedError) gilt als sauberer Abbruch.
ROH = (KeyError, IndexError, ValueError, AttributeError, TypeError,
       ZeroDivisionError)

SZENARIEN = {
    "duenn": (2021, "Belgium", "R"),
    "alt": (2018, "Austria", "R"),
    "ergast": None,
}

ZEITLIMIT_S = 240


def _lauf(pfad: str, szenario: str, q) -> None:
    """ein Skript in einem eigenen Prozess, mit umgebogenem f1lab."""
    warnings.filterwarnings("ignore")
    sys.path.insert(0, str(REPO))
    sys.argv = [pfad]

    import matplotlib
    matplotlib.use("Agg")
    import fastf1
    fastf1.set_log_level("CRITICAL")
    import f1lab
    import f1lab.session as sm

    f1lab.enable_cache()
    ziel = SZENARIEN[szenario]

    if ziel is not None:
        echt_load = f1lab.load

        def stub_load(jahr, event, ident="R", **kw):
            # Saison und Event verwerfen, alles andere durchreichen: ein
            # Aufrufer, der messages=True braucht, muss sie auch bekommen.
            # Kein eigener Zwischenspeicher noetig - f1lab.load traegt ein
            # lru_cache, ein Saison-Scan laedt also trotzdem nur einmal.
            return echt_load(
                *ziel,
                telemetry=kw.get("telemetry", False),
                weather=kw.get("weather", False),
                messages=kw.get("messages", False))

        # bewusstes Monkeypatching: beide Namen zeigen auf dieselbe
        # Funktion, und die Skripte greifen mal ueber f1lab.load, mal
        # ueber das Untermodul darauf zu.
        f1lab.load = stub_load                          # type: ignore[assignment]
        sm.load = stub_load                             # type: ignore[assignment]
    else:
        def stub_ergast(fn, *a, versuche=5, pause=3.0,
                        leer_bei_fehlschlag=False, **kw):
            if leer_bei_fehlschlag:
                return None
            raise ConnectionError("Ergast nicht erreichbar (simuliert)")

        f1lab.ergast_retry = stub_ergast
        sm.ergast_retry = stub_ergast

    try:
        with contextlib.redirect_stdout(io.StringIO()), \
             contextlib.redirect_stderr(io.StringIO()):
            runpy.run_path(str(REPO / pfad), run_name="__main__")
        q.put(("ok", ""))
    except ROH as exc:
        tb = traceback.extract_tb(sys.exc_info()[2])
        # .venv liegt im Repo-Ordner: ohne den zweiten Test zeigt der
        # Bericht auf pandas statt auf die Zeile, die es angeht.
        eigen = [f for f in tb
                 if str(REPO) in f.filename and "/.venv/" not in f.filename]
        wo = (f"{Path(eigen[-1].filename).name}:{eigen[-1].lineno}"
              if eigen else "?")
        q.put(("umgefallen", f"{type(exc).__name__} in {wo}: {str(exc)[:90]}"))
    except SystemExit:
        q.put(("ok", "SystemExit"))
    except BaseException as exc:                        # noqa: BLE001
        q.put(("sauber", f"{type(exc).__name__}: {str(exc)[:90]}"))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("szenario", choices=sorted(SZENARIEN))
    p.add_argument("skripte", nargs="*",
                   help="Standard: alle 51 Analyseskripte")
    a = p.parse_args()

    skripte = a.skripte or sorted(
        str(q.relative_to(REPO)) for q in REPO.glob("*/p*.py"))

    umgefallen, sauber, ok = [], [], []
    for pfad in skripte:
        q: mp.Queue = mp.Queue()
        prozess = mp.Process(target=_lauf, args=(pfad, a.szenario, q))
        prozess.start()
        prozess.join(ZEITLIMIT_S)
        if prozess.is_alive():
            prozess.terminate()
            prozess.join()
            art, info = "zeitlimit", f"> {ZEITLIMIT_S} s"
        else:
            art, info = q.get() if not q.empty() else ("zeitlimit",
                                                       "kein Ergebnis")

        kurz = Path(pfad).name.split("_")[0]
        if art == "umgefallen":
            umgefallen.append((kurz, pfad, info))
            print(f"  UMGEFALLEN  {kurz}  {info}", flush=True)
        elif art == "sauber":
            sauber.append(kurz)
            print(f"  sauber      {kurz}  {info}", flush=True)
        elif art == "ok":
            ok.append(kurz)
            print(f"  ok          {kurz}", flush=True)
        else:
            print(f"  {art}   {kurz}  {info}", flush=True)

    print(f"\n=== {a.szenario}: {len(ok)} durchgelaufen, {len(sauber)} "
          f"sauber abgebrochen, {len(umgefallen)} umgefallen ===")
    for kurz, pfad, info in umgefallen:
        print(f"  {kurz:<5} {info}\n        {pfad}")
    return 1 if umgefallen else 0


if __name__ == "__main__":
    raise SystemExit(main())
