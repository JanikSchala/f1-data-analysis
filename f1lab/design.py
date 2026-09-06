"""farben und beschriftungen, einmal festgelegt.

skripte und dashboard sollen gleich aussehen und dieselben farben fuer
dieselbe bedeutung verwenden. deshalb stehen sie hier statt je datei neu.

die kategoriefarben sind gegen die flaeche :data:`BG` geprueft, nicht nach
gefuehl gewaehlt. geprueft wurden helligkeitsband, buntheit,
unterscheidbarkeit bei farbsehschwaeche (CVD), unterscheidbarkeit bei
normalem sehen und kontrast zur flaeche. ab vier serien faellt die
unterscheidbarkeit unter die schwelle. deshalb ist :data:`MAX_SERIEN` = 3.

wer mehr reihen zeigen will, braucht eine andere form. ein sortiertes
balkendiagramm mit einer einzigen farbe traegt beliebig viele fahrer. die
identitaet haengt dann an der achse und nicht an der farbe.
"""
from __future__ import annotations

# --- flaechen und schrift ---
BG = "#15151e"          # diagrammflaeche
BG_HELL = "#1e1e2a"     # karten, panels
FG = "#f0f0f0"          # haupttext
MUTED = "#8a8a99"       # achsen, nebentext
GRID = "#2a2a38"        # gitter, rahmen

# --- chrome-akzente (oberflaeche, keine daten) ---
# nur fuer rahmen/glow/ueberschriften in app/common.py, nicht fuer diagramme.
# anders als SERIEN/COMPOUND/PHASE unten sind das keine kategoriefarben.
# sie muessen nie zwei dinge voneinander unterscheiden, deshalb sind sie
# nicht gegen farbsehschwaeche geprueft: reine dekoration der oberflaeche,
# kein traeger einer bedeutung.
ACCENT = "#ff2e44"       # racing-rot: primaryColor, glow-rahmen, akzentlinien
ACCENT_GLOW = "rgba(255, 46, 68, 0.35)"
TELEMETRY = "#2dd4ff"    # elektrisches cyan: HUD-zahlen, zweiter akzent

# --- kategoriefarben, feste reihenfolge ---
# nie zyklisch weiterdrehen: farbe haengt am ding, nicht an seinem rang.
SERIEN = ["#3987e5", "#d95926", "#199e70"]
MAX_SERIEN = len(SERIEN)

# --- feste bedeutungen ---
KONVENTIONELL, SPRINT = SERIEN[0], SERIEN[1]

# reifenmischungen tragen ihre eigene, im sport etablierte farbcodierung.
# sie ist gesetzt und wird nicht gegen die kategoriepalette getauscht.
COMPOUND = {
    "SOFT": "#e34948", "MEDIUM": "#eda100", "HARD": "#e8e8e8",
    "INTERMEDIATE": "#1baf7a", "WET": "#2a78d6", "UNKNOWN": MUTED,
}

# flaggenphasen tragen wie die mischungen eine feste bedeutung: die farbe
# steht fuer den zustand der strecke, nicht fuer eine kategorie unter
# anderen. gruen bleibt bewusst blass: es ist der normalfall und soll ein
# diagramm nicht zustellen.
PHASE = {
    "gruen": "#1baf7a", "gelb": "#eda100", "safety car": "#e2761b",
    "vsc": "#a679d6", "vsc endet": "#6f6f8a", "rot": "#e34948",
}

# wertung einer einzelnen zahl, keine kategorie: hebt sich ab / hebt sich
# nicht ab, lohnt sich / lohnt sich nicht.
POSITIV, NEGATIV = SERIEN[2], SERIEN[1]

# sequenzielle rampe fuer groessen (hoehenmeter, degradation): ein farbton,
# dunkel nach hell. kein regenbogen, der erfindet kategoriegrenzen, wo
# keine sind.
RAMPE = ["#14413a", "#199e70", "#6fe3bb"]

# dieselbe rampe als Plotly-colorscale.
RAMPE_SCALE = [[i / (len(RAMPE) - 1), f] for i, f in enumerate(RAMPE)]


def rampe_farben(n: int) -> list[str]:
    """n farben aus :data:`RAMPE`, dunkel nach hell.

    fuer *geordnete* reihen mit mehr als :data:`MAX_SERIEN` eintraegen -
    saisons, jahrgaenge, stufen. die kategoriepalette darf dafuer nicht
    zyklisch weitergedreht werden: bei vier reihen bekaeme die vierte
    dieselbe farbe wie die erste, und zwei verschiedene dinge waeren im
    diagramm nicht mehr auseinanderzuhalten.

    die rampe hat dieses problem nicht, weil sie die reihenfolge selbst
    traegt statt identitaeten zu vergeben. fuer *ungeordnete* kategorien ist
    sie deshalb falsch - dort braucht es eine andere darstellungsform (siehe
    modul-docstring).

    Args:
        n: anzahl gewuenschter farben, mindestens 1.

    Returns:
        hex-farben von dunkel nach hell. n=1 gibt den mittleren ton.
    """
    if n < 1:
        raise ValueError(f"mindestens eine farbe noetig, {n} angefragt")

    stufen = [_hex_zu_rgb(f) for f in RAMPE]
    if n == 1:
        return [_rgb_zu_hex(stufen[len(stufen) // 2])]

    out = []
    for i in range(n):
        pos = i / (n - 1) * (len(stufen) - 1)
        unten = min(int(pos), len(stufen) - 2)
        t = pos - unten
        a, b = stufen[unten], stufen[unten + 1]
        out.append(_rgb_zu_hex(tuple(round(a[k] + t * (b[k] - a[k]))
                                     for k in range(3))))
    return out


def _hex_zu_rgb(farbe: str) -> tuple[int, int, int]:
    h = farbe.lstrip("#")
    return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))


def _rgb_zu_hex(rgb) -> str:
    return "#{:02x}{:02x}{:02x}".format(*rgb)

IDENT_NAME = {
    "FP1": "1. Freies Training", "FP2": "2. Freies Training",
    "FP3": "3. Freies Training", "Q": "Qualifying",
    "SQ": "Sprint-Qualifying", "S": "Sprint", "R": "Rennen",
}


def matplotlib_stil() -> dict:
    """rcParams fuer den dunklen stil, wie ihn make_assets.py nutzt."""
    return {
        "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
        "text.color": FG, "axes.labelcolor": FG,
        "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.edgecolor": GRID, "grid.color": GRID, "font.size": 11,
    }


def plotly_layout(hoehe: int = 420, **kw) -> dict:
    """grundlayout fuer Plotly im selben stil."""
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
