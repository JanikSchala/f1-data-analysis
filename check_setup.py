#!/usr/bin/env python3
"""Prueft, ob alles bereit ist. Starte mich als erstes:  python check_setup.py"""
import importlib
import platform
import sys
from pathlib import Path

OK, WARN, BAD = "  [ok] ", "  [!!] ", "  [XX] "
problems = []

print("\n=== Umgebung ===")
print(f"  Python  {platform.python_version()}  ({sys.executable})")
if sys.version_info < (3, 9):  # noqa: UP036 - prueft die Nutzer-Installation, kein totes Gate
    problems.append("Python 3.9 oder neuer wird benoetigt.")
    print(BAD + "Version zu alt")
else:
    print(OK + "Version passt")

in_venv = sys.prefix != getattr(sys, "base_prefix", sys.prefix)
print(OK + "virtuelle Umgebung aktiv" if in_venv
      else WARN + "keine virtuelle Umgebung aktiv (.venv auswaehlen!)")

print("\n=== Pakete ===")
CORE = ["fastf1", "pandas", "numpy", "matplotlib"]
OPTIONAL = {
    "sklearn": "Projekte 23-25 (Machine Learning)",
    "duckdb": "Projekt 26 (Data Warehouse)",
    "pyarrow": "Projekt 02 und 26 (Parquet)",
    "fastapi": "Projekt 27 (REST-API)",
    "streamlit": "Projekt 28 (Dashboard)",
    "plotly": "Projekt 28 (Dashboard)",
    "typer": "Projekt 34 (CLI-Tool)",
}

for mod in CORE:
    try:
        m = importlib.import_module(mod)
        print(OK + f"{mod:<12} {getattr(m, '__version__', '?')}")
    except ImportError:
        print(BAD + f"{mod:<12} fehlt")
        problems.append(f"{mod} installieren:  pip install {mod}")

for mod, why in OPTIONAL.items():
    try:
        importlib.import_module(mod)
        print(OK + f"{mod:<12} vorhanden")
    except ImportError:
        print(WARN + f"{mod:<12} fehlt - gebraucht fuer {why}")

print("\n=== Cache ===")
cache = Path.home() / "f1_cache"
cache.mkdir(exist_ok=True)
n = len(list(cache.rglob("*"))) if cache.exists() else 0
print(OK + f"{cache}  ({n} Eintraege)")

print("\n=== Verbindung zur F1-API ===")
try:
    import fastf1
    fastf1.Cache.enable_cache(str(cache))
    sched = fastf1.get_event_schedule(2024, include_testing=False)
    print(OK + f"Kalender 2024 geladen: {len(sched)} Rennen, "
               f"erstes = {sched.iloc[0]['EventName']}")
except Exception as exc:
    print(BAD + f"Kein Zugriff: {exc}")
    problems.append("Internetverbindung oder Firewall pruefen.")

print("\n" + "=" * 58)
if problems:
    print("Noch zu erledigen:")
    for p in problems:
        print("  - " + p)
else:
    print("Alles bereit. Starte mit 01_grundlagen/p01_*.py")
print("=" * 58 + "\n")
