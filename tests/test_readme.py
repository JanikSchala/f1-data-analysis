"""das README muss zum Repository passen.

Es ist die Seite, die Fremde zuerst sehen, und es verweist auf jedes
einzelne der 51 Skripte. Ein umbenanntes oder verschobenes Skript bricht
die Links, ohne dass irgendwo etwas rot wird - auf GitHub sieht man es
erst beim Klicken.

Bewusst nur Aussagen, die *nicht* bei jedem Commit rotten: tote Links,
fehlende Projekte, falsche Anzahlen. Die Zahl der Tests stand hier
frueher auch drin und war nach zwei Sitzungen zweimal falsch - solche
Zahlen gehoeren nicht in eine Datei, die niemand mitpflegt.
"""
from __future__ import annotations

import pathlib
import re

import pytest

WURZEL = pathlib.Path(__file__).resolve().parents[1]
README = WURZEL / "README.md"


@pytest.fixture(scope="module")
def text() -> str:
    return README.read_text()


def _skripte() -> list[pathlib.Path]:
    return sorted(WURZEL.glob("*/p*.py"))


class TestLinks:
    def test_es_gibt_ueberhaupt_links(self, text):
        assert len(re.findall(r"\[[^\]]*\]\(([^)\s]+)\)", text)) > 50

    def test_kein_relativer_link_ins_leere(self, text):
        tot = []
        for ziel in re.findall(r"\[[^\]]*\]\(([^)\s]+)\)", text):
            if ziel.startswith(("http", "#", "mailto:")):
                continue
            # "datei.py#L12" -> "datei.py"
            if not (WURZEL / ziel.split("#")[0]).exists():
                tot.append(ziel)
        assert not tot, f"tote relative Links im README: {tot}"


class TestVollstaendigkeit:
    def test_jedes_skript_ist_erwaehnt(self, text):
        """P01-P51 - der Projektindex ist das Inhaltsverzeichnis des
        Repositories. Ein neues Skript, das dort fehlt, findet niemand."""
        genannt = set(re.findall(r"\bP(\d{2})\b", text))
        vorhanden = {p.name[1:3] for p in _skripte()}
        assert not vorhanden - genannt, (
            f"Skripte ohne Erwaehnung im README: "
            f"{sorted(vorhanden - genannt)}")

    def test_kein_erfundenes_projekt(self, text):
        genannt = set(re.findall(r"\bP(\d{2})\b", text))
        vorhanden = {p.name[1:3] for p in _skripte()}
        assert not genannt - vorhanden, (
            f"im README erwaehnt, aber nicht vorhanden: "
            f"{sorted(genannt - vorhanden)}")

    def test_die_genannte_anzahl_stimmt(self, text):
        """Die Zahl steht an mehreren prominenten Stellen ("51
        eigenstaendige Analysen", "Alle 51 Analysen"). Sie ist die einzige
        Zahl im README, die sich nur aendert, wenn wirklich ein Projekt
        dazukommt - deshalb darf sie drinbleiben, muss dann aber stimmen.
        """
        n = len(_skripte())
        genannte = re.findall(r"\b(\d+)\s+(?:eigenständige\s+)?Analysen", text)
        assert genannte, "keine Angabe der Analysenzahl im README gefunden"
        falsch = [g for g in genannte if int(g) != n]
        # jede Nennung, nicht irgendeine: die Zahl steht an mehreren
        # Stellen, und beim Gegenpruefen ist genau das aufgefallen - eine
        # falsche Stelle blieb unentdeckt, solange eine richtige existierte.
        assert not falsch, (
            f"{n} Skripte vorhanden, README nennt aber {sorted(set(falsch))}")

    def test_dashboard_seitenzahl(self, text):
        n = len(list((WURZEL / "app" / "pages").glob("*.py")))
        assert f"{n} Seiten" in text, (
            f"{n} Dashboard-Seiten vorhanden, README nennt eine andere Zahl")
