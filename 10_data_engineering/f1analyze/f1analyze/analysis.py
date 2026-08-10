"""Analyse-Funktionen mit Typannotationen (VORGEHEN 2).

Bewusst duenne Wrapper um f1lab, nicht neu geschriebene Rechnung: f1lab ist
in diesem Repository schon die getestete, von der App und den P-Skripten
gemeinsam genutzte Quelle fuer Race Pace, Degradation und Stints. Ein
"Wochenend-Analyzer", der diese drei Dinge stattdessen neu implementiert,
waere genau die Art Kopie, die das Projekt an anderer Stelle wiederholt
vermieden hat (siehe CLAUDE.md, DRY-Regel) - hier zaehlt, dass die
Bausteine zu einem Werkzeug zusammenkommen, nicht dass sie ein zweites Mal
geschrieben werden.
"""
from __future__ import annotations

import pandas as pd

import f1lab


def race_pace(session, threshold: float = 1.07) -> pd.DataFrame:
    """Bereinigte, treibstoffkorrigierte Race Pace, schnellster zuerst."""
    return f1lab.pace_table(session, threshold=threshold)


def degradation(session, threshold: float = 1.10, min_laps: int = 6) -> pd.DataFrame:
    """Degradation je Stint, mit Treibstoffkorrektur."""
    return f1lab.degradation(session, threshold=threshold, min_laps=min_laps)


def degradation_by_compound(session, **kwargs) -> pd.DataFrame:
    """Mittlere Degradation je Reifenmischung, nur belastbare Fits."""
    return f1lab.degradation_by_compound(session, **kwargs)


def stint_summary(session) -> pd.DataFrame:
    """Ein Datensatz je Stint: Fahrer, Compound, Start, Ende, Laenge."""
    return f1lab.stints(session)


def pit_loss(session) -> float:
    """Zeitverlust eines Boxenstopps auf dieser Strecke, in Sekunden."""
    return f1lab.pit_loss(session)
