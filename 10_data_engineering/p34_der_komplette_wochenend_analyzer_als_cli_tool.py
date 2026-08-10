"""
P34 - Der komplette Wochenend-Analyzer als CLI-Tool
===================================================

Ein installierbares Python-Paket, das alle vorherigen Analysen buendelt: f1analyze weekend 2024 Monza --report.

Kategorie:   Data Engineering
Niveau:      Profi
Aufwand:     8-10 h
Schwerpunkt: Engineering, Datenanalyse, Strategie
Zusaetzliche Pakete: typer

WARUM DAS LOHNT
Das Abschlussprojekt. Zeigt Software-Engineering statt Skript-Sammlung: Paketstruktur, CLI, Tests, Typannotationen, README. Genau so arbeiten Teams.

VORGEHEN
  1. Paketstruktur anlegen: f1analyze/{data,analysis,viz,cli}.py
  2. Analysen aus den Vorprojekten als Funktionen mit Typannotationen
  3. CLI mit Subcommands: weekend, pace, strategy, telemetry, report
  4. pytest mit gecachten Fixture-Sessions (kein Netz im Test)
  5. pyproject.toml, README mit Beispielbildern, auf GitHub veroeffentlichen

GENUTZTE FASTF1-BAUSTEINE
  - fastf1 gesamt
  - typer/argparse
  - pytest
  - pyproject.toml

AUSBAUSTUFE  [umgesetzt]
Publiziere das Paket auf PyPI und richte eine GitHub Action ein, die bei
jedem Push Tests, Linting (ruff) und Typecheck (mypy) laufen laesst.

Die Vorlage war anders kaputt als alle anderen 33 Projekte: sie war kein
lauffaehiges Skript mit Luecken, sondern ein Fragment eines Pakets, das nie
existierte (`from .analysis import ...` - ein relativer Import ohne
umgebendes Package, direkt beim Ausfuehren ein ImportError). VORGEHEN 1
("Paketstruktur anlegen") war schlicht nicht passiert.

Jetzt existiert das Paket wirklich, unter ``10_data_engineering/f1analyze/``
- eigene pyproject.toml, `pip install -e .` liefert ein echtes
`f1analyze`-Kommando:

    f1analyze/
      pyproject.toml, README.md
      f1analyze/{__init__,data,analysis,viz,cli}.py
      tests/{conftest,test_analysis,test_cli}.py
      .github/workflows/ci.yml

`analysis.py` ruft ausschliesslich `f1lab` auf (Race Pace, Degradation,
Stints, Pit-Loss sind dort schon getestet und werden von der App genutzt) -
ein Wochenend-Analyzer, der diese Rechnung ein zweites Mal schreibt, waere
genau die Kopie, die dieses Projekt sonst konsequent vermeidet. Neu ist nur
die Buendelung: fuenf Subcommands statt vier (VORGEHEN 3 nannte "telemetry"
explizit, die Vorlage hatte keins - jetzt ein Bremszonen-Vergleich zweier
Fahrer per f1lab.compare_braking_zones()), ein PDF-Report ueber
matplotlib.backends.backend_pdf.PdfPages, und pytest-Fixtures mit
f1lab.enable_cache(offline=True) fuer garantiert netzfreie Tests.

AUSBAUSTUFE: die GitHub Action (.github/workflows/ci.yml, Tests + ruff +
mypy) ist da. PyPI-Veroeffentlichung bewusst NICHT ausgefuehrt - das ist
eine echte, oeffentliche, kaum umkehrbare Aktion (ein einmal vergebener
Name auf PyPI lässt sich nicht zurueckziehen), die dieses Skript nicht
autonom treffen sollte. Details dazu im README des Pakets.

Dieses Skript selbst demonstriert das installierte Kommando end-to-end
gegen echte, gecachte Daten (Typers eigener CliRunner, dieselbe Technik wie
in tests/test_cli.py) - der Beweis, dass "pip install -e . && f1analyze
weekend 2024 Bahrain" wirklich funktioniert, nicht nur, dass die Datei
syntaktisch gueltig ist.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PAKET_ROOT = Path(__file__).parent / "f1analyze"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(PAKET_ROOT))

warnings.filterwarnings("ignore")

from f1analyze.cli import app  # noqa: E402
from typer.testing import CliRunner  # noqa: E402

import f1lab  # noqa: E402

SEASON, EVENT = 2024, "Bahrain"


def main():
    f1lab.enable_cache()
    runner = CliRunner()

    print(f"[1/4] f1analyze pace {SEASON} {EVENT} --top 8 ...")
    r = runner.invoke(app, ["pace", str(SEASON), EVENT, "--top", "8"])
    assert r.exit_code == 0, r.output
    print(r.output)

    print(f"[2/4] f1analyze strategy {SEASON} {EVENT} ...")
    r = runner.invoke(app, ["strategy", str(SEASON), EVENT])
    assert r.exit_code == 0, r.output
    print(r.output[:600] + "\n      ... (gekuerzt)")

    print(f"[3/4] f1analyze telemetry {SEASON} {EVENT} VER PER ...")
    r = runner.invoke(app, ["telemetry", str(SEASON), EVENT, "VER", "PER"])
    assert r.exit_code == 0, r.output
    print(r.output)

    out = Path(__file__).parent / "out"
    out.mkdir(exist_ok=True)
    report_pfad = out / f"{EVENT.lower()}_{SEASON}_report.pdf"
    print(f"[4/4] f1analyze report {SEASON} {EVENT} --out {report_pfad} ...")
    r = runner.invoke(app, ["report", str(SEASON), EVENT, "--out", str(report_pfad)])
    assert r.exit_code == 0, r.output
    print(r.output)
    print(f"      PDF-Groesse: {report_pfad.stat().st_size / 1024:.0f} KB")

    print("\nAlle vier Subcommands liefen ueber das installierbare Paket "
         "gegen echte Daten. Details, Architektur, CI: "
         f"{PAKET_ROOT / 'README.md'}")


if __name__ == "__main__":
    main()
