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

``seiten``
    Kein Szenario, sondern eine andere Oberflaeche: alle Dashboard-Seiten
    gegen den *warmen* Cache. Der Rauchtest in tests/test_app_seiten.py
    deckt nur den leeren Cache ab - und leer heisst: keine Session, also
    auch keine kaputte. Genau dort lag der Fehler, an dem zwei Seiten beim
    Saison-Scan umfielen, und der sich erst mit echten Daten zeigt.

Bewertet wird nach Absturzart, nicht nach Erfolg: ein Skript darf
abbrechen, wenn es nichts auszuwerten gibt - es muss dann nur sagen,
*dass* nichts da ist. Umfaellt heisst KeyError, IndexError, ValueError
und Verwandte; das sind die Faelle, in denen der Nutzer einen Traceback
sieht statt einer Aussage.

Braucht einen warmen FastF1-Cache mit den Zielsessions und laeuft rund
20 Minuten - deshalb kein Test und nicht Teil der CI.

``seiten`` dauert nochmal deutlich laenger, und das liegt an der
Pruefumgebung, nicht an der App: unter AppTest gibt es keine
Streamlit-Laufzeit, ``st.cache_data(persist="disk")`` bekommt ein
DummyCacheStorage untergeschoben und speichert nichts. Jeder Lauf
rechnet also alles neu, waehrend die App im Betrieb nur beim ersten
Aufruf rechnet. Die ML-Seite trainiert dabei mehrere Modelle und laeuft
in ihr Zeitlimit - das ist kein Befund ueber die Seite.

Aufruf::

    python robustheit.py duenn
    python robustheit.py alt 05_reifen_strategie/p13_*.py
    python robustheit.py seiten
"""
from __future__ import annotations

import argparse
import contextlib
import io
import logging
import multiprocessing as mp
import queue
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
    "seiten": None,
}

# Die Seiten rechnen deutlich laenger als die Skripte: sie trainieren
# Modelle und koennen den Diskcache hier nicht nutzen (siehe Modul-
# Docstring). Die ML-Seite laeuft auch in 900 Sekunden nicht immer durch -
# das ist ein Befund ueber die Pruefumgebung, nicht ueber die Seite.
ZEITLIMIT_S = {"seiten": 900}
ZEITLIMIT_STANDARD_S = 240
# AppTest bricht selbst ab und wirft dabei einen RuntimeError. Sein Limit
# muss unter dem Prozesslimit liegen, sonst laufen beide gegeneinander und
# man sieht nicht, welches zuerst gegriffen hat - aber nur knapp darunter:
# mit 600 Sekunden liefen drei Seiten ins Limit, die mit 900 durchlaufen
# (die ML-Seite, der Startplatz-Scan mit seinen Wartezeiten und die
# Renndynamik). Die Differenz reicht fuers Aufraeumen und Melden.
APPTEST_LIMIT_S = 840


def _seite_laufen(pfad: str, q) -> None:
    """eine Dashboard-Seite ueber Streamlits AppTest durchfahren.

    AppTest faengt jede Ausnahme der *Seite* ab und legt sie in
    ``app.exception``, statt sie durchzureichen - deshalb hier ein eigener
    Weg statt runpy. Seine eigenen Fehler wirft es dagegen normal; der
    Aufrufer faengt sie ab und trennt Zeitlimit von Absturz.
    """
    sys.path.insert(0, str(REPO / "app"))       # die Seiten importieren "common"
    from streamlit.testing.v1 import AppTest

    # AppTest laeuft ohne Streamlit-Laufzeit und warnt darueber je Seite
    # einmal - in einem Bericht ueber 27 Seiten sind das 27 Zeilen ohne
    # Aussage. Der Logger muss namentlich getroffen werden: er traegt ein
    # eigenes Level, ein ERROR auf dem Eltern-Logger "streamlit" laesst
    # die Warnung durch. Das Level erst nach dem Import setzen, Streamlit
    # richtet sein Logging beim Import selbst ein.
    logging.getLogger(
        "streamlit.runtime.scriptrunner_utils.script_run_context"
    ).setLevel(logging.ERROR)

    with contextlib.redirect_stdout(io.StringIO()), \
         contextlib.redirect_stderr(io.StringIO()):
        seite = AppTest.from_file(str(REPO / pfad),
                                  default_timeout=APPTEST_LIMIT_S)
        seite.run()

    if not seite.exception:
        q.put(("ok", ""))
        return
    fehler = seite.exception[0]
    spur = [z for z in (fehler.stack_trace or [])
            if "/app/" in z or "/f1lab/" in z]
    wo = spur[-1].strip().splitlines()[0][-80:] if spur else "?"
    q.put(("umgefallen", f"{fehler.message[:70]} | {wo}"))


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

    if szenario == "seiten":
        # eigenes try: _seite_laufen laeuft ausserhalb des Blocks weiter
        # unten, und eine Ausnahme hier (AppTest selbst, ein Import) haette
        # den Kindprozess ohne Eintrag in der Queue sterben lassen. Der
        # Bericht sagte dann "Prozess ohne Ergebnis beendet" und
        # verschwieg, was tatsaechlich passiert ist.
        try:
            _seite_laufen(pfad, q)
        except RuntimeError as exc:
            # AppTests eigener Abbruch. Kein Befund ueber die Seite: unter
            # AppTest gibt es keine Streamlit-Laufzeit, persist="disk"
            # speichert nichts, und jede Seite rechnet alles neu.
            art = ("zeitlimit" if "timed out" in str(exc) else "umgefallen")
            q.put((art, f"AppTest: {str(exc)[:90]}"))
        except BaseException as exc:                # noqa: BLE001
            q.put(("umgefallen",
                   f"{type(exc).__name__} ausserhalb AppTest: "
                   f"{str(exc)[:90]}"))
        return

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
    elif szenario == "ergast":
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
                   help="Standard: alle 51 Analyseskripte, bei 'seiten' "
                        "alle Dashboard-Seiten")
    a = p.parse_args()

    if a.szenario == "seiten":
        ziele = a.skripte or [str(q.relative_to(REPO)) for q in
                              [REPO / "app" / "Start.py",
                               *sorted((REPO / "app" / "pages").glob("*.py"))]]
    else:
        ziele = a.skripte or sorted(
            str(q.relative_to(REPO)) for q in REPO.glob("*/p*.py"))
    skripte = ziele

    grenze = ZEITLIMIT_S.get(a.szenario, ZEITLIMIT_STANDARD_S)
    umgefallen, sauber, ok, zeitlimit = [], [], [], []
    for pfad in skripte:
        q: mp.Queue = mp.Queue()
        prozess = mp.Process(target=_lauf, args=(pfad, a.szenario, q))
        prozess.start()
        prozess.join(grenze)
        if prozess.is_alive():
            prozess.terminate()
            prozess.join()
            art, info = "zeitlimit", f"> {grenze} s"
        else:
            # kurzer Nachlauf statt q.empty(): die Queue schreibt in einem
            # eigenen Thread, nach join() ist ein Eintrag noch nicht
            # zwingend sichtbar. Ein erfolgreicher Lauf wurde dadurch als
            # "kein Ergebnis" gemeldet. Nicht gleich mit q.get(grenze)
            # warten - dann haette ein hart abgestuerzter Kindprozess das
            # volle Limit blockiert, statt sofort aufzufallen.
            try:
                art, info = q.get(timeout=10)
            except queue.Empty:
                # der Exit-Code sagt, ob der Prozess sauber endete (0),
                # sich selbst beendete oder von aussen abgeschossen wurde
                # (negativ = Signal, -9 heisst in der Regel Speichermangel).
                art = "zeitlimit"
                info = ("Prozess ohne Ergebnis beendet, Exit-Code "
                        f"{prozess.exitcode}")

        # Skripte heissen p04_..., Seiten 18_Ueberholschwierigkeit.py -
        # bei denen ist die Nummer allein keine Auskunft.
        kurz = (Path(pfad).stem if a.szenario == "seiten"
                else Path(pfad).name.split("_")[0])
        if art == "umgefallen":
            umgefallen.append((kurz, pfad, info))
            print(f"  UMGEFALLEN  {kurz:<26} {info}", flush=True)
        elif art == "sauber":
            sauber.append(kurz)
            print(f"  sauber      {kurz:<26} {info}", flush=True)
        elif art == "ok":
            ok.append(kurz)
            print(f"  ok          {kurz}", flush=True)
        else:
            # ein abgewuergter Lauf hat nichts bewiesen und darf nicht
            # unter den Tisch fallen - frueher zaehlte er in keiner der
            # drei Listen und fehlte damit in der Zusammenfassung ganz.
            zeitlimit.append((kurz, pfad, info))
            print(f"  ZEITLIMIT   {kurz:<26} {info}", flush=True)

    print(f"\n=== {a.szenario}: {len(ok)} durchgelaufen, {len(sauber)} "
          f"sauber abgebrochen, {len(umgefallen)} umgefallen, "
          f"{len(zeitlimit)} am Zeitlimit ===")
    for kurz, pfad, info in umgefallen + zeitlimit:
        print(f"  {kurz:<5} {info}\n        {pfad}")
    return 1 if (umgefallen or zeitlimit) else 0


if __name__ == "__main__":
    raise SystemExit(main())
