#!/bin/bash
# Richtet die virtuelle Umgebung ein und installiert alle Pakete.
#
#   ./setup.sh
#
# Unter macOS geht auch ein Doppelklick auf 1_Setup_starten.command -
# das ruft nur dieses Skript hier auf.

cd "$(dirname "$0")" || exit 1

echo ""
echo "==========================================================="
echo "   F1 Data Analysis - Einrichtung"
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
  echo "    https://www.python.org/downloads/   (Installer, am einfachsten)"
  echo "    oder:  brew install python@3.12"
  echo ""
  exit 1
fi

echo "  Python gefunden: $($PY --version)  ->  $(command -v "$PY")"
echo ""

if [ ! -d ".venv" ]; then
  echo "  Lege virtuelle Umgebung an (.venv) ..."
  "$PY" -m venv .venv || { echo "  Fehlgeschlagen."; exit 1; }
else
  echo "  .venv existiert bereits."
fi

# shellcheck disable=SC1091
source .venv/bin/activate

echo "  Aktualisiere pip ..."
python -m pip install --upgrade pip --quiet

echo "  Installiere Pakete (beim ersten Mal ein paar Minuten) ..."
python -m pip install -r requirements.txt || {
  echo "  Installation fehlgeschlagen."; exit 1; }

echo ""
echo "  Pruefe Installation ..."
echo ""
python check_setup.py

echo ""
echo "==========================================================="
echo "   Fertig. Naechster Schritt:"
echo "     source .venv/bin/activate"
echo "     python 01_grundlagen/p01_session_explorer_*.py"
echo "==========================================================="
echo ""
