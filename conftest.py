"""Macht das Repo-Wurzelverzeichnis fuer pytest importierbar.

Ohne diese Datei muesste das Paket erst installiert werden (pip install -e .),
damit `import f1lab` in den Tests funktioniert. Da pytest das Verzeichnis der
obersten conftest.py automatisch auf sys.path legt, reicht ihre blosse
Existenz - die Tests laufen damit direkt aus einem frischen Checkout.
"""
