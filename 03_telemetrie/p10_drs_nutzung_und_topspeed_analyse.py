"""
P10 - DRS-Nutzung und Topspeed-Analyse
======================================

Wie lange ist DRS offen, wie viel bringt es, und wer nutzt es am effizientesten?

Kategorie:   Telemetrie
Niveau:      Fortgeschritten
Aufwand:     3 h
Schwerpunkt: Datenanalyse, Strategie

WARUM DAS LOHNT
DRS-Effektivitaet ist direkt ueberholrelevant. Der DRS-Kanal ist codiert (nicht 0/1) - dass du das weisst, zeigt Detailtiefe.

VORGEHEN
  1. DRS-Codes verstehen: 10/12/14 = aktiv, 0/1/8 = zu
  2. Aktivzonen als Distanzintervalle extrahieren
  3. Topspeed mit und ohne DRS je Fahrer vergleichen
  4. DRS-Zeitanteil pro Runde als Ranking

GENUTZTE FASTF1-BAUSTEINE
  - Telemetry DRS
  - Telemetry Speed/Distance
  - Laps.pick_fastest

AUSBAUSTUFE
Vergleiche die DRS-Zonen ueber mehrere Jahre derselben Strecke und zeige, wie die FIA sie verschoben hat.
"""

import pandas as pd
import fastf1

fastf1.Cache.enable_cache("~/f1_cache")
ses = fastf1.get_session(2024, "Italy", "Q")   # Monza: 2 DRS-Zonen
ses.load()

DRS_OPEN = {10, 12, 14}     # aktiv-Codes im F1-Feed

rows = []
for drv in ses.drivers:
    try:
        lap = ses.laps.pick_drivers(drv).pick_fastest()
        tel = lap.get_car_data().add_distance()
    except Exception:
        continue
    if tel is None or tel.empty:
        continue

    tel = tel.copy()
    tel["DRSopen"] = tel["DRS"].isin(DRS_OPEN)
    tel["dt"] = tel["Time"].diff().dt.total_seconds().fillna(0)

    open_t = tel.loc[tel["DRSopen"], "dt"].sum()
    total_t = tel["dt"].sum()

    rows.append({
        "Driver": ses.get_driver(drv)["Abbreviation"],
        "Team": ses.get_driver(drv)["TeamName"],
        "DRS_s": round(open_t, 2),
        "DRS_%": round(100 * open_t / total_t, 1),
        "vmax_DRS": round(tel.loc[tel["DRSopen"], "Speed"].max(), 1),
        "vmax_zu": round(tel.loc[~tel["DRSopen"], "Speed"].max(), 1),
    })

df = pd.DataFrame(rows)
df["Gewinn_kmh"] = (df["vmax_DRS"] - df["vmax_zu"]).round(1)
print(df.sort_values("vmax_DRS", ascending=False).to_string(index=False))

# DRS-Zonen der Strecke lokalisieren
lap = ses.laps.pick_fastest()
tel = lap.get_car_data().add_distance()
tel["open"] = tel["DRS"].isin(DRS_OPEN)
tel["grp"] = (tel["open"] != tel["open"].shift()).cumsum()
zones = (tel[tel["open"]].groupby("grp")["Distance"]
         .agg(["min", "max"]).round(0))
print("\n" + "DRS-Zonen [m]:")
print(zones.to_string(index=False))
