"""
P22 - 75 Jahre F1: Historische Trendanalyse
===========================================

Alle Saisons seit 1950 aus der Ergast-kompatiblen API: Dominanzphasen, Zuverlaessigkeit, Nationen, Streckenwandel.

Kategorie:   Historie & Ergast-API
Niveau:      Fortgeschritten
Aufwand:     4-5 h
Schwerpunkt: Datenanalyse

WARUM DAS LOHNT
Grosse Datenmengen, viele API-Calls, saubere Aggregation - und am Ende Zusammenhaenge, die man vorher nicht kannte. Wann war die F1 am ausgeglichensten, wann am langweiligsten?

VORGEHEN
  1. Alle Saisons iterieren und Constructor-Standings sammeln
  2. Dominanz-Index berechnen (Punkteanteil des Siegers)
  3. DNF-Rate ueber die Jahrzehnte aus finishing_status
  4. Multi-Panel-Grafik mit vier Trends

GENUTZTE FASTF1-BAUSTEINE
  - Ergast.get_seasons
  - Ergast.get_race_results
  - Ergast.get_constructor_standings
  - Ergast.get_finishing_status

AUSBAUSTUFE
Ergaenze eine Analyse der Rundenzeitenentwicklung auf Strecken, die seit Jahrzehnten unveraendert sind (Monza, Spa, Silverstone).
"""

import time
import pandas as pd
import matplotlib.pyplot as plt
from fastf1.ergast import Ergast

erg = Ergast(result_type="pandas", auto_cast=True)

rows = []
for year in range(1958, 2025):          # Konstrukteurs-WM ab 1958
    try:
        cs = erg.get_constructor_standings(season=year).content[0]
    except Exception:
        continue
    total = cs["points"].sum()
    if total == 0:
        continue
    rows.append({
        "Season": year,
        "Champion": cs.iloc[0]["constructorName"],
        "Anteil": cs.iloc[0]["points"] / total,
        "Teams": len(cs),
    })
    time.sleep(0.2)                      # API freundlich behandeln

dom = pd.DataFrame(rows)
print(dom.nlargest(10, "Anteil").to_string(index=False))

# DNF-Rate je Saison
dnf_rows = []
for year in range(1990, 2025):
    try:
        st = erg.get_finishing_status(season=year).content[0]
    except Exception:
        continue
    total = st["count"].sum()
    finished = st.loc[st["status"].str.contains("Finished|Lap", na=False), "count"].sum()
    dnf_rows.append({"Season": year, "DNF_Rate": 1 - finished / total})
    time.sleep(0.2)

dnf = pd.DataFrame(dnf_rows)

fig, ax = plt.subplots(2, 1, figsize=(12, 9))
ax[0].bar(dom["Season"], dom["Anteil"] * 100, color="#e10600")
ax[0].set_ylabel("Punkteanteil Champion [%]")
ax[0].set_title("Dominanz des Konstrukteurs-Weltmeisters")
ax[1].plot(dnf["Season"], dnf["DNF_Rate"] * 100, marker="o", color="#00d2be")
ax[1].set_ylabel("DNF-Rate [%]"); ax[1].set_xlabel("Saison")
ax[1].set_title("Zuverlaessigkeit ueber die Zeit")
plt.tight_layout()
plt.show()
