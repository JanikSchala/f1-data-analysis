"""Verweise auf f1lab in Docstrings und Kommentaren muessen ins Ziel zeigen.

Das Repository ist bewusst so geschrieben, dass die Prosa die
Arbeitsteilung erklaert: die Dashboard-Seiten sagen "gerechnet wird
nirgends hier, jede Kennzahl kommt aus f1lab.weather_join() /
temperature_effect()", die Skripte nennen die Bausteine, die sie
verwenden. Vierzig solcher Nennungen stehen im Code.

Wird eine f1lab-Funktion umbenannt, faellt der Aufruf sofort auf - die
Prosa daneben nicht. Sie zeigt dann auf einen Namen, den es nicht mehr
gibt, und erklaert eine Arbeitsteilung, die so nicht mehr stimmt. Genau
diese Erklaerungen sind hier aber der Grund, warum jemand den Code
ueberhaupt lesen kann.

Geprueft wird nur die Form "f1lab." plus Funktionsname plus Klammern. Ein
blosser Funktionsname ohne Praefix waere mehrdeutig - die Seiten haben
eigene Hilfsfunktionen mit denselben Namensmustern.

Das Muster ist oben bewusst umschrieben statt hingeschrieben: dieser
Docstring wird selbst mit eingesammelt, ein Beispiel darin wuerde also
als echter Verweis zaehlen und einen Fehlschlag ausloesen. Beim ersten
Anlauf ist genau das passiert - zweimal.
"""
from __future__ import annotations

import ast
import pathlib
import re

import pytest

import f1lab

WURZEL = pathlib.Path(__file__).resolve().parents[1]
AUSGENOMMEN = {".venv", ".git", "__pycache__", "graphify-out", "build", "dist"}

# "f1lab.pace_table()" und "f1lab.session.reference_lap()"
VERWEIS = re.compile(r"f1lab(?:\.\w+)?\.(\w+)\(\)")


def _prosa(pfad: pathlib.Path) -> str:
    """Docstrings und Kommentare einer Datei, ohne den ausfuehrbaren Code."""
    quelle = pfad.read_text()
    stuecke = []
    for knoten in ast.walk(ast.parse(quelle)):
        if isinstance(knoten, (ast.Module, ast.ClassDef, ast.FunctionDef,
                               ast.AsyncFunctionDef)):
            doku = ast.get_docstring(knoten)
            if doku:
                stuecke.append(doku)
    stuecke += [z.split("#", 1)[1] for z in quelle.splitlines()
                if "#" in z and not z.strip().startswith(("'", '"'))]
    return "\n".join(stuecke)


def _dateien() -> list[pathlib.Path]:
    return [p for p in sorted(WURZEL.rglob("*.py"))
            if not any(t in AUSGENOMMEN for t in p.parts)]


def _verweise() -> list[tuple[str, str]]:
    """(datei, name) je Nennung in Prosa."""
    aus = []
    for pfad in _dateien():
        try:
            text = _prosa(pfad)
        except SyntaxError:                          # pragma: no cover
            continue
        for name in sorted(set(VERWEIS.findall(text))):
            aus.append((str(pfad.relative_to(WURZEL)), name))
    return aus


def test_es_gibt_ueberhaupt_verweise():
    """Schutz gegen einen Test, der gruen ist, weil er nichts findet -
    zum Beispiel weil sich die Schreibweise geaendert hat."""
    assert len(_verweise()) >= 20


@pytest.mark.parametrize("datei,name", _verweise(),
                         ids=lambda v: v if isinstance(v, str) else str(v))
def test_der_verweis_zeigt_auf_eine_echte_funktion(datei, name):
    assert hasattr(f1lab, name), (
        f"{datei} nennt f1lab.{name}(), das es in f1lab nicht gibt")
