"""macht das Repo-Wurzelverzeichnis fuer pytest importierbar.

ohne diese Datei muesste das Paket erst installiert werden (pip install -e .).
nur so funktioniert `import f1lab` in den Tests. pytest legt das Verzeichnis
der obersten conftest.py automatisch auf sys.path. ihre blosse Existenz
genuegt also. die Tests laufen damit direkt aus einem frischen Checkout.
"""
