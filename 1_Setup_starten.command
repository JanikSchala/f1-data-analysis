#!/bin/bash
# Doppelklick-Einstieg fuer macOS. Die eigentliche Arbeit macht setup.sh -
# so steht die Einrichtung nur an einer Stelle.

cd "$(dirname "$0")" || exit 1
./setup.sh

echo ""
read -r -p "  Mit Enter schliessen ..." _
