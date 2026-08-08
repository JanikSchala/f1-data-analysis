#!/bin/bash
# Doppelklick-Start fuer das Dashboard. Richtet sich an Leute, die kein
# Terminal benutzen: alles, was schiefgehen kann, wird hier abgefangen und
# in einem verstaendlichen Satz erklaert - nie als Python-Fehlermeldung.

cd "$(dirname "$0")" || exit 1

BLAU=$'\033[38;5;39m'; GRUEN=$'\033[38;5;42m'; ROT=$'\033[38;5;203m'
GRAU=$'\033[38;5;245m'; FETT=$'\033[1m'; AUS=$'\033[0m'

clear
printf '%s\n' "${BLAU}${FETT}"
printf '   ███████╗ ██╗\n'
printf '   ██╔════╝███║   Formel-1-Analyse\n'
printf '   █████╗  ╚██║   Dashboard\n'
printf '   ██╔══╝   ██║\n'
printf '   ██║      ██║\n'
printf '   ╚═╝      ╚═╝\n'
printf '%s\n' "${AUS}"

abbruch() {
  printf '\n  %s✗ %s%s\n\n' "${ROT}" "$1" "${AUS}"
  printf '  %s%s%s\n\n' "${GRAU}" "$2" "${AUS}"
  read -r -p "  Mit Enter schliessen ..." _
  exit 1
}

# --- 1. Umgebung ----------------------------------------------------------
printf '  %s›%s Pruefe Einrichtung ... ' "${GRAU}" "${AUS}"
if [ ! -x ".venv/bin/python" ]; then
  printf '\n'
  abbruch "Die Arbeitsumgebung fehlt." \
          "Starte zuerst einmal \"1_Setup_starten.command\" per Doppelklick.
  Das richtet alles Noetige ein und muss nur einmal gemacht werden."
fi

if ! .venv/bin/python -c "import streamlit" 2>/dev/null; then
  printf '\n'
  abbruch "Das Dashboard-Paket fehlt." \
          "Starte \"1_Setup_starten.command\" per Doppelklick, das ergaenzt es."
fi
printf '%s✓%s\n' "${GRUEN}" "${AUS}"

# --- 2. Daten -------------------------------------------------------------
printf '  %s›%s Suche gespeicherte Renndaten ... ' "${GRAU}" "${AUS}"
ANZAHL=$(.venv/bin/python -c "
import sys
sys.path.insert(0, '.')
try:
    import f1lab
    print(len(f1lab.cached_sessions()))
except Exception:
    print(0)
" 2>/dev/null)

if [ -z "$ANZAHL" ] || [ "$ANZAHL" -eq 0 ] 2>/dev/null; then
  printf '\n'
  abbruch "Es sind noch keine Renndaten gespeichert." \
          "Das Dashboard zeigt nur Rennen, die auf diesem Rechner liegen.
  Lade sie einmalig herunter - das laeuft mehrere Stunden von allein:

      .venv/bin/python 01_grundlagen/p01_*.py"
fi
printf '%s✓%s  %s Sessions\n' "${GRUEN}" "${AUS}" "${ANZAHL}"

# --- 3. Start -------------------------------------------------------------
printf '\n'
printf '  %sDas Dashboard oeffnet sich gleich im Browser.%s\n' "${FETT}" "${AUS}"
printf '  %sFalls nicht, rufe von Hand auf:  http://localhost:8501%s\n\n' \
       "${GRAU}" "${AUS}"
printf '  %sZum Beenden: dieses Fenster schliessen oder Strg+C druecken.%s\n\n' \
       "${GRAU}" "${AUS}"

.venv/bin/python -m streamlit run app/Start.py

printf '\n  %sDashboard beendet.%s\n\n' "${GRAU}" "${AUS}"
read -r -p "  Mit Enter schliessen ..." _
