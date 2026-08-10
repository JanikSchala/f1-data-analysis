"""Session-Zugriff: ein duenner Wrapper um f1lab.load().

Der sys.path-Trick fuer f1lab (kein eigenstaendiges Package, ein Ordner im
selben Repository) steht einmal zentral in f1analyze/__init__.py - das
Package-__init__ laeuft garantiert vor jedem Untermodul, deshalb reicht hier
ein normaler Import.
"""
from __future__ import annotations

import f1lab


def load_session(year: int, gp: str | int, ident: str = "R",
                 telemetry: bool = False):
    """Session laden, mit demselben Cache wie der Rest des Projekts."""
    f1lab.enable_cache()
    return f1lab.load(year, gp, ident, telemetry=telemetry)
