"""tests fuer f1lab.design.

die Farbregeln stehen ausfuehrlich im Modul-Docstring, waren aber bis jetzt
nirgends festgenagelt. genau solche Regeln driften still: make_assets.py
hatte einmal eine eigene Farbkopie, die von matplotlib_stil() abgewichen
ist, ohne dass es jemandem auffiel.
"""
from __future__ import annotations

import pytest

from f1lab.design import (
    BG,
    COMPOUND,
    FG,
    GRID,
    MAX_SERIEN,
    MUTED,
    PHASE,
    RAMPE,
    RAMPE_SCALE,
    SERIEN,
    matplotlib_stil,
    plotly_layout,
    rampe_farben,
)


def ist_hex(farbe: str) -> bool:
    return (isinstance(farbe, str) and farbe.startswith("#")
            and len(farbe) == 7
            and all(c in "0123456789abcdefABCDEF" for c in farbe[1:]))


class TestKategoriepalette:
    """die Grenze von drei Serien ist eine gepruefte Aussage, keine Marotte:
    ab der vierten faellt die Unterscheidbarkeit unter die Schwelle."""

    def test_max_serien_passt_zur_palette(self):
        assert MAX_SERIEN == len(SERIEN) == 3

    def test_farben_sind_verschieden(self):
        assert len(set(SERIEN)) == len(SERIEN)

    def test_alles_hex(self):
        for f in [*SERIEN, *RAMPE, BG, FG, MUTED, GRID]:
            assert ist_hex(f), f

    def test_keine_serienfarbe_ist_die_flaeche(self):
        """eine Serie in Hintergrundfarbe waere unsichtbar."""
        assert BG not in SERIEN and FG not in SERIEN


class TestFesteBedeutungen:
    """Mischungen und Flaggenphasen tragen im Sport gesetzte Farben. sie
    duerfen nicht gegen die Kategoriepalette getauscht werden."""

    def test_mischungen_vollstaendig(self):
        for m in ("SOFT", "MEDIUM", "HARD", "INTERMEDIATE", "WET", "UNKNOWN"):
            assert m in COMPOUND

    def test_mischungen_sind_unterscheidbar(self):
        ohne_unknown = {k: v for k, v in COMPOUND.items() if k != "UNKNOWN"}
        assert len(set(ohne_unknown.values())) == len(ohne_unknown)

    def test_phasen_vollstaendig(self):
        for p in ("gruen", "gelb", "safety car", "vsc", "rot"):
            assert p in PHASE

    def test_gruen_ist_nicht_der_akzent(self):
        """gruen ist der Normalfall und soll ein Diagramm nicht zustellen."""
        assert PHASE["gruen"] != PHASE["rot"]


class TestRampe:
    def test_scale_deckt_null_bis_eins_ab(self):
        assert RAMPE_SCALE[0][0] == 0.0
        assert RAMPE_SCALE[-1][0] == 1.0

    def test_farben_von_dunkel_nach_hell(self):
        def helligkeit(f):
            h = f.lstrip("#")
            return sum(int(h[i:i + 2], 16) for i in (0, 2, 4))
        werte = [helligkeit(f) for f in rampe_farben(6)]
        assert werte == sorted(werte)

    def test_anzahl_stimmt(self):
        for n in (1, 2, 3, 4, 9, 20):
            assert len(rampe_farben(n)) == n

    def test_alle_verschieden(self):
        """der eigentliche Zweck gegenueber der zyklischen Kategoriepalette:
        neun Saisons bekommen neun Farben, nicht dreimal dieselben drei."""
        assert len(set(rampe_farben(9))) == 9

    def test_endpunkte_sind_die_rampenenden(self):
        got = rampe_farben(5)
        assert got[0].lower() == RAMPE[0].lower()
        assert got[-1].lower() == RAMPE[-1].lower()

    def test_drei_farben_geben_die_rampe_selbst(self):
        assert [f.lower() for f in rampe_farben(3)] == [f.lower() for f in RAMPE]

    def test_alles_hex(self):
        assert all(ist_hex(f) for f in rampe_farben(7))

    def test_null_farben_sind_ein_fehler(self):
        with pytest.raises(ValueError, match="mindestens eine"):
            rampe_farben(0)


class TestStilvorlagen:
    def test_matplotlib_stil_setzt_flaeche_und_schrift(self):
        stil = matplotlib_stil()
        assert stil["figure.facecolor"] == BG
        assert stil["savefig.facecolor"] == BG    # sonst weisser Rand beim Export
        assert stil["text.color"] == FG

    def test_plotly_layout_nimmt_dieselben_farben(self):
        """App und Skripte sollen gleich aussehen - beide Vorlagen muessen
        sich aus denselben Konstanten bedienen."""
        layout = plotly_layout()
        assert layout["paper_bgcolor"] == BG == layout["plot_bgcolor"]
        assert layout["font"]["color"] == FG
        assert layout["xaxis"]["gridcolor"] == GRID

    def test_plotly_layout_hoehe_und_ueberschreiben(self):
        assert plotly_layout(hoehe=555)["height"] == 555
        assert plotly_layout(showlegend=False)["showlegend"] is False
