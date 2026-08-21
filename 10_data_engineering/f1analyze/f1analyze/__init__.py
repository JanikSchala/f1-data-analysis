"""f1analyze: Wochenend-Analyzer, gebaut auf f1lab.

f1lab ist kein eigenstaendiges Package. es ist ein Ordner im selben
Repository (siehe f1analyze/data.py). der Pfad wird deshalb hier einmal
ergaenzt, beim Import des Packages. jedes Untermodul kann sich danach auf
"import f1lab" verlassen, ohne den Trick selbst zu wiederholen.
"""
from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

__version__ = "0.1.0"
