"""tests fuer f1analyze.viz.

viz.py war mit 22% das am schlechtesten abgedeckte Modul des Pakets.
Grafikcode wird gern ungetestet gelassen, weil "sieht man ja" - genau
deshalb brechen die Zusagen darin still: eine falsch gestapelte
Stint-Leiste oder eine vertauschte Sortierung faellt beim Drueberschauen
nicht auf, ein fehlender Balken schon gar nicht.

Geprueft wird deshalb nicht, dass eine Figure herauskommt, sondern was
in ihr steht: Reihenfolge, Hervorhebung, Balkenlaengen und -positionen.
Alles ohne Session, damit die Tests auch ohne Fixture-Cache laufen.
"""
from __future__ import annotations

import matplotlib
import pandas as pd
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from f1analyze import viz

PACE_SPALTEN = ["driver", "team", "laps", "median_s", "delta_s",
                "ci_lo", "ci_hi", "ci_width"]
STINT_SPALTEN = ["Driver", "stint", "Compound", "laps"]


@pytest.fixture
def pace():
    """vier Fahrer, absichtlich unsortiert uebergeben."""
    return pd.DataFrame({
        "driver": ["LANGSAM", "SCHNELL", "MITTE", "VIERTER"],
        "team": ["A", "B", "C", "D"], "laps": [30, 30, 30, 30],
        "median_s": [93.0, 90.0, 91.0, 92.0],
        "delta_s": [3.0, 0.0, 1.0, 2.0],
        "ci_lo": [2.8, -0.1, 0.9, 1.8],
        "ci_hi": [3.2, 0.1, 1.1, 2.2],
        "ci_width": [0.4, 0.2, 0.2, 0.4],
    })


@pytest.fixture
def stints():
    return pd.DataFrame({
        "Driver": ["VER", "VER", "VER", "HAM", "HAM"],
        "stint": [1, 2, 3, 1, 2],
        "Compound": ["SOFT", "HARD", "MEDIUM", "MEDIUM", "HARD"],
        "laps": [15, 25, 17, 30, 27],
    })


@pytest.fixture(autouse=True)
def _figuren_schliessen():
    yield
    plt.close("all")


class TestPlotPace:
    def test_schnellster_steht_oben(self, pace):
        """die Tabelle kommt sortiert, muss es aber nicht - plot_pace
        sortiert selbst und dreht die Achse um.

        beides gehoert geprueft: barh() zeichnet von unten nach oben, die
        Sortierung allein wuerde den Schnellsten also nach *unten* setzen.
        Erst invert_yaxis() dreht das um. Ein Test nur auf die
        Beschriftungsreihenfolge uebersieht ein fehlendes invert_yaxis()
        vollstaendig - genau das ist beim Gegenpruefen aufgefallen.
        """
        ax = viz.plot_pace(pace, "t").axes[0]
        beschriftungen = [t.get_text() for t in ax.get_yticklabels()]
        assert beschriftungen[0] == "SCHNELL"
        assert beschriftungen[-1] == "LANGSAM"
        unten, oben = ax.get_ylim()
        assert unten > oben, "y-Achse nicht umgedreht: erste Zeile liegt unten"

    def test_top_drei_sind_hervorgehoben(self, pace):
        """drei Akzentfarben, der Rest gedaempft - sonst hebt sich nichts
        ab."""
        from f1lab.design import MUTED, SERIEN
        ax = viz.plot_pace(pace, "t").axes[0]
        farben = [b.get_facecolor() for b in ax.patches]
        akzent = matplotlib.colors.to_rgba(SERIEN[0])
        gedaempft = matplotlib.colors.to_rgba(MUTED)
        assert farben[:3] == [akzent] * 3
        assert farben[3] == gedaempft

    def test_balkenlaengen_sind_die_deltas(self, pace):
        ax = viz.plot_pace(pace, "t").axes[0]
        assert sorted(b.get_width() for b in ax.patches) == [0.0, 1.0, 2.0, 3.0]

    def test_leerer_rahmen_gibt_eine_leere_grafik(self):
        """ein abgebrochenes Rennen liefert keine Pace-Tabelle. Das ist
        eine leere Aussage, kein Fehler."""
        fig = viz.plot_pace(pd.DataFrame(columns=PACE_SPALTEN), "t")
        assert not fig.axes[0].patches

    def test_verkehrte_ci_grenzen_werfen_nicht(self, pace):
        """matplotlib lehnt negative xerr ab. pace_table() rundet
        delta_s/ci_lo/ci_hi unabhaengig, wodurch die Untergrenze rechnerisch
        um 0.001 ueber den Punktwert rutschen kann.
        """
        kaputt = pace.copy()
        kaputt.loc[0, "ci_lo"] = kaputt.loc[0, "delta_s"] + 0.001
        viz.plot_pace(kaputt, "t")          # darf nicht werfen


class TestPlotStrategy:
    def test_stints_stapeln_sich_luecklos(self, stints):
        """jeder Stint beginnt, wo der vorige endete - sonst behauptet die
        Grafik Runden, die es nicht gab, oder verschluckt welche."""
        ax = viz.plot_strategy(stints, ["VER"], "t").axes[0]
        balken = sorted(((b.get_x(), b.get_width()) for b in ax.patches),
                        key=lambda p: p[0])
        assert [b[1] for b in balken] == [15, 25, 17]
        assert [b[0] for b in balken] == [0, 15, 40]

    def test_gesamtlaenge_ist_die_rundenzahl(self, stints):
        ax = viz.plot_strategy(stints, ["VER", "HAM"], "t").axes[0]
        je_fahrer = {}
        for b in ax.patches:
            je_fahrer.setdefault(round(b.get_y()), 0)
            je_fahrer[round(b.get_y())] += b.get_width()
        assert sorted(je_fahrer.values()) == [57, 57]

    def test_mischungsfarben(self, stints):
        from f1lab.design import COMPOUND
        ax = viz.plot_strategy(stints, ["VER"], "t").axes[0]
        balken = sorted(ax.patches, key=lambda b: b.get_x())
        erwartet = [matplotlib.colors.to_rgba(COMPOUND[c])
                    for c in ("SOFT", "HARD", "MEDIUM")]
        assert [b.get_facecolor() for b in balken] == erwartet

    def test_fahrer_ohne_stint_bleibt_leer(self, stints):
        """order kommt aus den Rennergebnissen, stints_df aus den Runden -
        ein Fahrer kann in einem stehen und im anderen fehlen (Ausfall in
        Runde 1)."""
        ax = viz.plot_strategy(stints, ["VER", "NIEGEFAHREN"], "t").axes[0]
        assert len(ax.patches) == 3

    def test_leerer_rahmen_gibt_eine_leere_grafik(self):
        fig = viz.plot_strategy(pd.DataFrame(columns=STINT_SPALTEN), [], "t")
        assert not fig.axes[0].patches


class TestBuildPdf:
    class _Event(dict):
        year = 2024

    class _Session:
        def __init__(self, fahrer):
            self.event = TestBuildPdf._Event(EventName="Testing Grand Prix")
            self.results = pd.DataFrame({
                "Position": range(1, len(fahrer) + 1),
                "Abbreviation": fahrer})

    def test_schreibt_ein_lesbares_pdf(self, tmp_path, pace, stints):
        ziel = tmp_path / "report.pdf"
        viz.build_pdf(self._Session(["VER", "HAM"]), None, pace, stints, ziel)
        kopf = ziel.read_bytes()[:5]
        assert kopf == b"%PDF-", kopf
        assert ziel.stat().st_size > 1000

    def test_reihenfolge_kommt_aus_dem_ergebnis(self, tmp_path, pace, stints):
        """build_pdf() ordnet die Strategie nach Zielposition und laesst
        Fahrer weg, fuer die es keine Stints gibt. Faellt das um, steht im
        Report eine falsche Rangfolge - sichtbar wird das nirgends."""
        ses = self._Session(["HAM", "VER", "OHNE_STINTS"])
        gesehen = {}
        echt = viz.plot_strategy

        def merken(stints_df, order, titel):
            gesehen["order"] = order
            return echt(stints_df, order, titel)

        viz.plot_strategy = merken
        try:
            viz.build_pdf(ses, None, pace, stints, tmp_path / "r.pdf")
        finally:
            viz.plot_strategy = echt
        assert gesehen["order"] == ["HAM", "VER"]
