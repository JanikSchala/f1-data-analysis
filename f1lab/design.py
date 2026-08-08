"""Farben und Beschriftungen, einmal festgelegt.

Skripte und Dashboard sollen gleich aussehen und dieselben Farben fuer
dieselbe Bedeutung verwenden. Deshalb stehen sie hier und nicht je Datei neu.

Die Kategoriefarben sind gegen die Flaeche :data:`BG` geprueft worden, nicht
nach Gefuehl gewaehlt - Pruefkriterien waren Helligkeitsband, Buntheit,
Unterscheidbarkeit bei Farbsehschwaeche (CVD), Unterscheidbarkeit bei
normalem Sehen und Kontrast zur Flaeche.

Ergebnis der Pruefung auf allen Paaren (Linien, Punkte, Streudiagramme):

    2 Serien   CVD dE 26.8   normal 31.8   bestanden
    3 Serien   CVD dE  9.4   normal 20.9   bestanden
    4 Serien   CVD dE  4.8   normal 10.6   DURCHGEFALLEN

Deshalb :data:`MAX_SERIEN` = 3. Wer mehr Reihen zeigen will, braucht eine
andere Form - ein sortiertes Balkendiagramm mit einer einzigen Farbe etwa
traegt beliebig viele Fahrer, weil die Identitaet dann an der Achse haengt
und nicht an der Farbe.
"""
from __future__ import annotations

# --- Flaechen und Schrift -------------------------------------------------
BG = "#15151e"          # Diagrammflaeche
BG_HELL = "#1e1e2a"     # Karten, Panels
FG = "#f0f0f0"          # Haupttext
MUTED = "#8a8a99"       # Achsen, Nebentext
GRID = "#2a2a38"        # Gitter, Rahmen

# --- Kategoriefarben, feste Reihenfolge ----------------------------------
# Nie zyklisch weiterdrehen: Farbe haengt am Ding, nicht an seinem Rang.
SERIEN = ["#3987e5", "#d95926", "#199e70"]
MAX_SERIEN = len(SERIEN)

# --- Feste Bedeutungen ----------------------------------------------------
KONVENTIONELL, SPRINT = SERIEN[0], SERIEN[1]

# Reifenmischungen tragen ihre eigene, im Sport etablierte Farbcodierung.
# Die ist gesetzt und wird nicht gegen die Kategoriepalette getauscht.
COMPOUND = {
    "SOFT": "#e34948", "MEDIUM": "#eda100", "HARD": "#e8e8e8",
    "INTERMEDIATE": "#1baf7a", "WET": "#2a78d6", "UNKNOWN": MUTED,
}

# Flaggenphasen tragen wie die Mischungen eine feste Bedeutung: die Farbe
# steht fuer den Zustand der Strecke und nicht fuer eine Kategorie unter
# anderen. Gruen bleibt bewusst blass - es ist der Normalfall und soll ein
# Diagramm nicht zustellen.
PHASE = {
    "gruen": "#1baf7a", "gelb": "#eda100", "safety car": "#e2761b",
    "vsc": "#a679d6", "vsc endet": "#6f6f8a", "rot": "#e34948",
}

# Wertung einer einzelnen Zahl, keine Kategorie: hebt sich ab / hebt sich
# nicht ab, lohnt sich / lohnt sich nicht.
POSITIV, NEGATIV = SERIEN[2], SERIEN[1]

# Sequenzielle Rampe fuer Groessen (Hoehenmeter, Degradation): ein Farbton,
# dunkel nach hell. Kein Regenbogen - der erfindet Kategoriegrenzen, wo
# keine sind.
RAMPE = ["#14413a", "#199e70", "#6fe3bb"]

# Dieselbe Rampe als Plotly-Colorscale.
RAMPE_SCALE = [[i / (len(RAMPE) - 1), f] for i, f in enumerate(RAMPE)]

IDENT_NAME = {
    "FP1": "1. Freies Training", "FP2": "2. Freies Training",
    "FP3": "3. Freies Training", "Q": "Qualifying",
    "SQ": "Sprint-Qualifying", "S": "Sprint", "R": "Rennen",
}


def matplotlib_stil() -> dict:
    """rcParams fuer den dunklen Stil, wie ihn make_assets.py nutzt."""
    return {
        "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
        "text.color": FG, "axes.labelcolor": FG,
        "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.edgecolor": GRID, "grid.color": GRID, "font.size": 11,
    }


def plotly_layout(hoehe: int = 420, **kw) -> dict:
    """Grundlayout fuer Plotly im selben Stil."""
    layout = {
        "height": hoehe,
        "paper_bgcolor": BG, "plot_bgcolor": BG,
        "font": {"color": FG, "size": 13},
        "xaxis": {"gridcolor": GRID, "zeroline": False,
                  "linecolor": GRID, "tickfont": {"color": MUTED}},
        "yaxis": {"gridcolor": GRID, "zeroline": False,
                  "linecolor": GRID, "tickfont": {"color": MUTED}},
        "legend": {"orientation": "h", "y": -0.18, "x": 0,
                   "bgcolor": "rgba(0,0,0,0)"},
        "margin": {"l": 60, "r": 24, "t": 32, "b": 60},
        "hoverlabel": {"bgcolor": BG_HELL, "bordercolor": GRID,
                       "font": {"color": FG}},
    }
    layout.update(kw)
    return layout
