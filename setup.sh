#!/bin/bash
# Doppelklick auf diese Datei richtet alles ein.
cd "$(dirname "$0")" || exit 1

echo ""
echo "==========================================================="
echo "   F1 Portfolio - Einrichtung"
echo "==========================================================="
echo ""

PY=""
for c in python3.12 python3.11 python3.10 python3; do
  if command -v "$c" >/dev/null 2>&1; then PY="$c"; break; fi
done

if [ -z "$PY" ]; then
  echo "  Python 3 wurde nicht gefunden."
  echo ""
  echo "  Bitte zuerst installieren:"
  echo "    Variante A  ->  https://www.python.org/downloads/  (Installer, am einfachsten)"
  echo "    Variante B  ->  brew install python@3.12"
  echo ""
  echo "  Danach diese Datei erneut doppelklicken."
  echo ""
  read -r -p "  Mit Enter schliessen ..." _
  exit 1
fi

echo "  Python gefunden: $($PY --version)  ->  $(command -v $PY)"
echo ""

if [ ! -d ".venv" ]; then
  echo "  Lege virtuelle Umgebung an (.venv) ..."
  "$PY" -m venv .venv || { echo "  Fehlgeschlagen."; read -r -p "  Enter ..." _; exit 1; }
else
  echo "  .venv existiert bereits."
fi

# shellcheck disable=SC1091
source .venv/bin/activate

echo "  Aktualisiere pip ..."
python -m pip install --upgrade pip --quiet

echo "  Installiere Pakete (das dauert beim ersten Mal ein paar Minuten) ..."
python -m pip install -r requirements.txt || {
  echo "  Installation fehlgeschlagen."; read -r -p "  Enter ..." _; exit 1; }

echo ""
echo "  Pruefe Installation ..."
echo ""
python check_setup.py

echo ""
echo "==========================================================="
echo "   Fertig. Naechster Schritt:"
echo "   VS Code oeffnen  ->  Ordner waehlen  ->  unten rechts"
echo "   den Interpreter '.venv' auswaehlen  ->  p01 starten."
echo "==========================================================="
echo ""
read -r -p "  Mit Enter schliessen ..." _
