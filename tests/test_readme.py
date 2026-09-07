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


class TestZahlenStimmenMitDenKennzahlen:
    """die Zahlen im Fliesstext muessen aus assets/kennzahlen.json kommen.

    make_assets.py erzeugt die Grafiken *und* schreibt die Kennzahlen
    daneben. Der README-Text zitiert sie im Fliesstext ("1124 von 1310
    Runden", "0.002 s dahinter"). Wer die Assets neu erzeugt - anderes
    Rennen, andere FastF1-Version, geaenderte Filterregel -, bekommt neue
    Grafiken und neue Kennzahlen, aber der Text daneben bleibt stehen.
    Dann behauptet das README etwas, das seine eigene Grafik widerlegt.

    Geprueft werden die Zahlen, die im Text tatsaechlich vorkommen, nicht
    alle: eine Kennzahl darf in kennzahlen.json stehen, ohne im README
    zitiert zu werden.
    """

    @pytest.fixture(scope="class")
    def kennzahlen(self) -> dict:
        import json
        pfad = WURZEL / "assets" / "kennzahlen.json"
        if not pfad.exists():                        # pragma: no cover
            pytest.skip("assets/kennzahlen.json fehlt")
        return json.loads(pfad.read_text())

    @staticmethod
    def _kommt_vor(text: str, wert) -> bool:
        """steht die Zahl als eigenstaendige Zahl im Text?

        ohne Wortgrenze wuerde "1124" auch in "11245" treffen und der Test
        waere wertlos.
        """
        return re.search(rf"(?<![\d.,]){re.escape(str(wert))}(?![\d])",
                         text) is not None

    def test_race_pace_zahlen(self, text, kennzahlen):
        rp = kennzahlen["racepace"]
        fehlend = [f"{name}={wert}" for name, wert in (
            ("runden_gesamt", rp["runden_gesamt"]),
            ("runden_nach_filter", rp["runden_nach_filter"]),
        ) if not self._kommt_vor(text, wert)]
        assert not fehlend, f"nicht im README-Text: {fehlend}"

    def test_race_pace_deltas(self, text, kennzahlen):
        """die vier Rueckstaende hinter dem Schnellsten - der Absatz
        argumentiert damit, dass sieben Fahrer ununterscheidbar sind."""
        deltas = [e["delta_s"] for e in kennzahlen["racepace"]["top5"][1:]]
        fehlend = [d for d in deltas if not self._kommt_vor(text, f"{d:.3f}")]
        assert not fehlend, f"Rueckstaende nicht im README: {fehlend}"

    def test_gangkarte_zahlen(self, text, kennzahlen):
        g = kennzahlen["gangkarte"]
        assert self._kommt_vor(text, g["streckenlaenge_m"])
        assert self._kommt_vor(text, g["punkte"])

    def test_overlay_zahlen(self, text, kennzahlen):
        o = kennzahlen["overlay"]
        assert self._kommt_vor(text, f"{o['delta_s']:.3f}")
        for v in o["vmax_kmh"].values():
            assert self._kommt_vor(text, int(v)), v


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

    def test_werkzeuge_auf_oberster_ebene_sind_gelistet(self, text):
        """der Aufbau-Abschnitt nennt die Ordner und die Skripte daneben.

        Ein neues Werkzeug, das dort fehlt, existiert fuer jeden Leser
        nicht - genau so ist robustheit.py beim Hinzufuegen durchgerutscht.
        conftest.py bleibt aussen vor: pytest-Innereien, kein Werkzeug.
        """
        werkzeuge = {p.name for p in WURZEL.glob("*.py")} - {"conftest.py"}
        fehlend = sorted(w for w in werkzeuge if w not in text)
        assert not fehlend, f"nicht im README erwaehnt: {fehlend}"

    def test_dashboard_seitenzahl(self, text):
        n = len(list((WURZEL / "app" / "pages").glob("*.py")))
        assert f"{n} Seiten" in text, (
            f"{n} Dashboard-Seiten vorhanden, README nennt eine andere Zahl")
