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


def _deklarierte_distributionen(pyproject: pathlib.Path, *,
                                nur_basis: bool = False) -> set[str]:
    """Basis plus jede Extra-Gruppe *einer* pyproject.toml.

    Args:
        nur_basis: nur die Pflicht-Abhaengigkeiten, ohne Extra-Gruppen.
    """
    if sys.version_info >= (3, 11):
        import tomllib
    else:                                        # pragma: no cover
        tomllib = pytest.importorskip("tomli")

    projekt = tomllib.loads(pyproject.read_text())["project"]
    specs = list(projekt.get("dependencies", []))
    if not nur_basis:
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


def _modul_zu_dist() -> dict[str, set[str]]:
    """Modulname -> Distributionen, die ihn liefern (scipy -> {"scipy"},
    sklearn -> {"scikit-learn"})."""
    aus = {}
    for modul, dists in packages_distributions().items():
        aus[modul] = {d.lower().replace("-", "_") for d in dists}
    return aus


def _check_setup_listen() -> tuple[list[str], list[str]]:
    """CORE und OPTIONAL aus check_setup.py, ohne es zu importieren.

    Das Skript arbeitet schon beim Import: es druckt, legt den Cache-Ordner
    an und fragt die F1-API ab. Ein Test darf das nicht ausloesen, deshalb
    ueber den Syntaxbaum.
    """
    baum = ast.parse((WURZEL / "check_setup.py").read_text())
    werte = {}
    for knoten in baum.body:
        if isinstance(knoten, ast.Assign) and len(knoten.targets) == 1:
            ziel = knoten.targets[0]
            if isinstance(ziel, ast.Name) and ziel.id in ("CORE", "OPTIONAL"):
                werte[ziel.id] = ast.literal_eval(knoten.value)
    return list(werte["CORE"]), list(werte["OPTIONAL"])


class TestCheckSetupPasstZurDeklaration:
    """check_setup.py ist das erste, was ein neuer Nutzer aufruft - es soll
    genau dann "Alles bereit" sagen, wenn es das auch ist.

    Vor diesem Durchgang fehlte scipy in CORE, obwohl es eine
    Basis-Abhaengigkeit ist und 22 Dateien es importieren: das Skript
    konnte gruen melden und die Analyseskripte danach reihenweise beim
    Import abbrechen. Ein Pruefskript, das seine eigene Pruefliste von
    Hand pflegt, driftet - dieser Test bindet sie an pyproject.toml.
    """

    def test_jede_basis_abhaengigkeit_steht_in_core(self):
        core, _ = _check_setup_listen()
        modul_zu_dist = _modul_zu_dist()
        abgedeckt = {d for modul in core
                     for d in modul_zu_dist.get(modul, {modul})}

        fehlend = _deklarierte_distributionen(
            WURZEL / "pyproject.toml", nur_basis=True) - abgedeckt
        assert not fehlend, (
            f"Basis-Abhaengigkeit(en) {sorted(fehlend)} fehlen in "
            f"check_setup.CORE - das Skript kann 'Alles bereit' melden, "
            f"waehrend Skripte beim Import abbrechen")

    def test_core_verlangt_nichts_optionales(self):
        """umgekehrt genauso falsch: was nur in einer Extra-Gruppe steht,
        darf nicht als fehlend gemeldet werden."""
        core, _ = _check_setup_listen()
        modul_zu_dist = _modul_zu_dist()
        basis = _deklarierte_distributionen(WURZEL / "pyproject.toml",
                                            nur_basis=True)
        zuviel = [m for m in core
                  if not (modul_zu_dist.get(m, {m}) & basis)]
        assert not zuviel, (
            f"{zuviel} steht in CORE, ist aber keine Basis-Abhaengigkeit")

    def test_jede_extra_gruppe_ist_in_optional_vertreten(self):
        """Geprueft wird je *Gruppe*, nicht je Paket.

        check_setup.py ist ein Wegweiser, kein vollstaendiges Manifest -
        "sklearn fehlt, gebraucht fuer Projekte 23/24/36" ist die
        nuetzliche Auskunft, nicht das Aufzaehlen jeder transitiven
        Abhaengigkeit (pydantic kommt ohnehin mit fastapi). Eine Gruppe
        ganz zu verschweigen ist der Fehler: genau so war torch nirgends
        erwaehnt, obwohl P25 und die ML-Seite es brauchen.

        dev bleibt aussen vor - Entwicklungswerkzeuge sind nicht das, was
        ein Nutzer beim Setup vermisst.
        """
        import tomllib
        gruppen = tomllib.loads(
            (WURZEL / "pyproject.toml").read_text()
        )["project"]["optional-dependencies"]

        _, optional = _check_setup_listen()
        modul_zu_dist = _modul_zu_dist()
        genannt = {d for modul in optional
                   for d in modul_zu_dist.get(modul, {modul})}

        stumm = []
        for name, specs in gruppen.items():
            if name == "dev":
                continue
            dists = {s.split(">")[0].split("=")[0].split("[")[0]
                     .strip().lower().replace("-", "_") for s in specs}
            if not dists & genannt:
                stumm.append(f"{name} ({', '.join(sorted(dists))})")

        assert not stumm, (
            f"Extra-Gruppe(n) ohne einen einzigen Vertreter in "
            f"check_setup.OPTIONAL: {stumm}")


class TestOptionalIstWirklichOptional:
    """was in einer Extra-Gruppe steht, darf das Dashboard nicht mitreissen.

    torch liegt in der Gruppe "deeplearning" (rund 500 MB) und wird nur vom
    Autoencoder auf einer einzigen Reiter-Haelfte gebraucht. Ein
    ungeschuetztes ``import torch`` auf Modulebene liess die ganze Seite
    abstuerzen, sobald jemand nur ".[dashboard]" installiert hatte - genau
    so ist es in der CI aufgeschlagen, nachdem torch ueberhaupt erst
    deklariert wurde. Eine Abhaengigkeit als optional zu deklarieren und
    sie dann hart zu importieren ist schlimmer als beides nicht zu tun.

    statisch geprueft: der Rauchtest in test_app_seiten.py faengt es zwar
    auch, aber nur in einer Umgebung ohne torch. Hier faellt es ueberall
    auf.
    """

    GRUPPE = "deeplearning"

    def _optionale_module(self) -> set[str]:
        import tomllib
        gruppe = tomllib.loads(
            (WURZEL / "pyproject.toml").read_text()
        )["project"]["optional-dependencies"][self.GRUPPE]
        return {spec.split(">")[0].split("=")[0].strip().lower()
                for spec in gruppe}

    def test_gruppe_existiert_und_ist_nicht_leer(self):
        assert self._optionale_module()

    def test_app_importiert_sie_nur_geschuetzt(self):
        optional = self._optionale_module()
        ungeschuetzt = []
        for pfad in sorted((WURZEL / "app").rglob("*.py")):
            baum = ast.parse(pfad.read_text())
            # alles, was in einem try steht, gilt als abgesichert
            in_try = set()
            for knoten in ast.walk(baum):
                if isinstance(knoten, ast.Try):
                    for kind in ast.walk(knoten):
                        in_try.add(id(kind))
            for knoten in ast.walk(baum):
                if isinstance(knoten, ast.Import):
                    namen = [a.name for a in knoten.names]
                elif isinstance(knoten, ast.ImportFrom) and knoten.module:
                    namen = [knoten.module]
                else:
                    continue
                if id(knoten) in in_try:
                    continue
                for name in namen:
                    if name.split(".")[0].lower() in optional:
                        ungeschuetzt.append(
                            f"{pfad.relative_to(WURZEL)}:{knoten.lineno} {name}")

        assert not ungeschuetzt, (
            f"Gruppe '{self.GRUPPE}' ist optional, wird aber ungeschuetzt "
            f"importiert: {ungeschuetzt}")


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
