"""haelt jeden Fremd-Import im Repo gegen pyproject.toml.

diese Luecke hat hier schon dreimal zugeschlagen: scipy, requests und
zuletzt torch/typer standen in requirements.txt oder gar nirgends, aber in
keiner Gruppe von pyproject.toml. Wer nach der dokumentierten Anleitung
installiert hat, bekam eine Umgebung, in der Skripte beim Import abbrechen -
und nichts hat es gemeldet, weil requirements.txt in der Entwicklungs-venv
ja laengst installiert war.

der Test liest die Importe aus dem Syntaxbaum statt per Textsuche, damit
weder auskommentierte Zeilen noch Strings mitzaehlen.
"""
from __future__ import annotations

import ast
import pathlib
import sys
from importlib.metadata import packages_distributions

import pytest

WURZEL = pathlib.Path(__file__).resolve().parents[1]
AUSGENOMMEN = {".venv", ".git", "__pycache__", "graphify-out", "build", "dist"}

# Module, die zum Repository selbst gehoeren und nirgends deklariert werden.
# "common" ist das gemeinsame Modul der Dashboard-Seiten, das Streamlit
# ueber den Seitenordner findet.
EIGEN = {"f1lab", "f1analyze", "common", "conftest"}


def _python_dateien() -> list[pathlib.Path]:
    return [p for p in WURZEL.rglob("*.py")
            if not any(teil in AUSGENOMMEN for teil in p.parts)]


def _fremd_importe(dateien: list[pathlib.Path]) -> dict[str, set[str]]:
    """top-level-Modulname -> Dateien, die ihn importieren."""
    alle = _python_dateien()
    lokal = {p.stem for p in alle}
    lokal |= {p.name for p in WURZEL.iterdir() if p.is_dir()}

    treffer: dict[str, set[str]] = {}
    for p in dateien:
        try:
            baum = ast.parse(p.read_text())
        except SyntaxError:                      # nicht unser Problem hier
            continue
        for knoten in ast.walk(baum):
            if isinstance(knoten, ast.Import):
                namen = [a.name for a in knoten.names]
            elif isinstance(knoten, ast.ImportFrom) and knoten.level == 0:
                namen = [knoten.module] if knoten.module else []
            else:
                continue
            for name in namen:
                wurzel_modul = name.split(".")[0]
                treffer.setdefault(wurzel_modul, set()).add(
                    str(p.relative_to(WURZEL)))

    return {m: d for m, d in treffer.items()
            if m not in sys.stdlib_module_names
            and m not in EIGEN
            and m not in lokal
            and not m.startswith("_")}


def _deklarierte_distributionen(pyproject: pathlib.Path) -> set[str]:
    """Basis plus jede Extra-Gruppe *einer* pyproject.toml."""
    if sys.version_info >= (3, 11):
        import tomllib
    else:                                        # pragma: no cover
        tomllib = pytest.importorskip("tomli")

    projekt = tomllib.loads(pyproject.read_text())["project"]
    specs = list(projekt.get("dependencies", []))
    for gruppe in projekt.get("optional-dependencies", {}).values():
        specs.extend(gruppe)

    namen = set()
    for spec in specs:
        # "scikit-learn>=1.3" -> "scikit_learn"
        roh = spec.split(";")[0].split("[")[0]
        for trenner in (">", "<", "=", "!", "~"):
            roh = roh.split(trenner)[0]
        namen.add(roh.strip().lower().replace("-", "_"))
    return namen


# jedes Paket wird gegen seine *eigene* Deklaration geprueft. beide Mengen zu
# vereinen waere zu nachsichtig: f1analyze deklariert scipy und typer fuer
# sich, das wuerde dieselben Luecken im Wurzelpaket zudecken - und genau die
# haben hier zugeschlagen (p34 importiert typer und liegt ausserhalb von
# f1analyze).
F1ANALYZE = WURZEL / "10_data_engineering" / "f1analyze"
PAKETE = {
    "Wurzelpaket f1lab": (
        WURZEL / "pyproject.toml",
        lambda p: F1ANALYZE not in p.parents),
    "Teilpaket f1analyze": (
        F1ANALYZE / "pyproject.toml",
        lambda p: F1ANALYZE in p.parents),
}


class TestJederImportIstDeklariert:
    @pytest.mark.parametrize("paket", list(PAKETE))
    def test_es_gibt_ueberhaupt_importe(self, paket):
        """Schutz gegen einen Test, der nur deshalb gruen ist, weil das
        Einsammeln nichts gefunden hat."""
        _, gehoert_dazu = PAKETE[paket]
        dateien = [p for p in _python_dateien() if gehoert_dazu(p)]
        assert dateien, paket
        assert len(_fremd_importe(dateien)) >= 4, paket

    @pytest.mark.parametrize("paket", list(PAKETE))
    def test_keine_undeklarierte_abhaengigkeit(self, paket):
        pyproject, gehoert_dazu = PAKETE[paket]
        verteilung = packages_distributions()
        deklariert = _deklarierte_distributionen(pyproject)
        dateien = [p for p in _python_dateien() if gehoert_dazu(p)]

        luecken = {}
        for modul, quellen in sorted(_fremd_importe(dateien).items()):
            dists = {d.lower().replace("-", "_")
                     for d in verteilung.get(modul, [])}
            if not dists:
                # nicht installiert - dann kann dieser Test nichts aussagen,
                # das faellt beim Import des Skripts selbst auf.
                continue
            if not dists & deklariert:
                luecken[modul] = sorted(quellen)[:3]

        assert not luecken, (
            f"nicht in {pyproject.relative_to(WURZEL)} deklariert: "
            + "; ".join(f"{m} (z.B. {', '.join(f)})" for m, f in luecken.items()))
