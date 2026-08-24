# f1analyze

Ein CLI-Tool, das die Analysen aus `f1lab` (dem gemeinsamen Kernpaket dieses
Repositories) zu einem installierbaren Wochenend-Analyzer buendelt:

```bash
cd 10_data_engineering/f1analyze
pip install -e ".[dev]"

f1analyze weekend 2024 Monza
f1analyze pace 2024 Monza --top 10
f1analyze strategy 2024 Monza
f1analyze telemetry 2024 Monza VER LEC
f1analyze report 2024 Monza --out monza.pdf
f1analyze optimize 2024 Bahrain          # P35: exakter Boxenstopp-Plan
f1analyze lap-sim 2024 Bahrain           # P37: Punktmassen-Rundenzeitsimulation
f1analyze overtakes 2024 Monza           # P39: Ueberholungen gegen DRS-Zonen
f1analyze traffic 2024 Bahrain 3         # P41: Optimum gegen 3-Stopp im Verkehr
```

`optimize`/`lap-sim`/`overtakes`/`traffic` kamen mit P35/P37/P39/P41 dazu.
P38 (Ueberholschwierigkeit je Strecke) und P40 (Startplatz-Paritaet) sind
bewusst KEIN Subcommand - beide brauchen einen Season-Scan ueber Dutzende
bis hunderte Sessions (Minuten, nicht Sekunden), das passt nicht zur
Kostenstruktur eines Einzelaufrufs. Beide leben nur als Dashboard-Seiten.

## Warum ein eigener Ordner statt eines Skripts

Die anderen 43 Projekte in diesem Repository sind bewusst einzelne,
eigenstaendig lauffaehige Skripte. Dieses hier baut ausdruecklich ein
*installierbares Paket* (`pip install -e .`, ein `f1analyze`-Kommando nach
der Installation, eigene `pyproject.toml`) - das war die Aufgabe, nicht ein
weiteres Skript mit `if __name__ == "__main__":`.

## Architektur

- `f1analyze/data.py` - Session laden (Wrapper um `f1lab.load()`)
- `f1analyze/analysis.py` - Race Pace, Degradation, Stints (Wrapper um
  `f1lab`, keine Neuimplementierung - siehe Modul-Docstring)
- `f1analyze/viz.py` - Grafiken im Hausstil (`f1lab.design`), PDF-Report
- `f1analyze/cli.py` - Typer-Subcommands: `weekend`, `pace`, `strategy`,
  `telemetry`, `report`, `optimize`, `lap-sim`, `overtakes`, `traffic`

`f1lab` selbst ist kein PyPI-Paket, sondern ein Ordner im Repository-Root -
`f1analyze/__init__.py` haengt den Repository-Root einmal an `sys.path`,
genau wie es jedes `p*.py`-Skript im Projekt einzeln tut.

## Tests

```bash
pytest
```

Laeuft komplett gegen einen mitgelieferten FastF1-Cache-Ausschnitt
(`tests/fixtures/f1_cache_bahrain2024/`, ~147 MB, per
`f1lab.enable_cache(path=..., offline=True)` in `tests/conftest.py`) - kein
Netzzugriff, keine flakiness durch API-Limits, und deterministisch fuer
jeden, der das Repo klont (nicht abhaengig davon, was zufaellig im eigenen
`~/f1_cache` liegt).

## CI

`.github/workflows/f1analyze-ci.yml` (im Repo-Wurzelverzeichnis - GitHub
Actions erkennt nur dort liegende Workflows, ein verschachteltes
`.github/` innerhalb dieses Ordners haette nie ausgeloest) fuehrt bei
jedem Push, der diesen Pfad aendert, Tests,
`ruff check` und `mypy` aus - das GitHub-Actions-Aequivalent der
AUSBAUSTUFE. Der zweite Teil
der AUSBAUSTUFE (Veroeffentlichung auf PyPI) ist bewusst nicht ausgefuehrt:
das ist eine echte, oeffentliche, kaum rueckgaengig zu machende Aktion, die
ausserhalb dieses Projekts entschieden werden sollte, nicht automatisiert
beim Bauen dieses Skripts.
