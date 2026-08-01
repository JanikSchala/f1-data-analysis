"""
P07 - Telemetrie-Overlay: Zwei schnellste Runden uebereinanderlegen
===================================================================

Speed, Throttle, Brake, Gang und DRS zweier Fahrer ueber die Distanz - plus kumuliertes Zeit-Delta.

Kategorie:   Telemetrie
Niveau:      Fortgeschritten
Aufwand:     3 h
Rollen:      DS, STRAT

WARUM DAS ZAEHLT
Das ist die Standard-Grafik jedes Performance-Engineers. Wer sie sauber inkl. delta_time() bauen kann, hat den Kern der Telemetrie-Arbeit verstanden.

VORGEHEN
  1. Schnellste Runde beider Fahrer holen
  2. Car-Data mit add_distance() anreichern
  3. delta_time() fuer die Zeitdifferenz ueber die Distanz
  4. Fuenf gestapelte Subplots mit gemeinsamer x-Achse

GENUTZTE FASTF1-BAUSTEINE
  - Lap.get_car_data
  - Telemetry.add_distance
  - fastf1.utils.delta_time
  - fastf1.plotting.get_driver_style

AUSBAUSTUFE
Markiere automatisch die Bremspunkte (Brake-Flanken) und annotiere, wer wo spaeter bremst.
"""

import fastf1
import fastf1.plotting as f1plt
from fastf1.utils import delta_time
import matplotlib.pyplot as plt

fastf1.Cache.enable_cache("~/f1_cache")
f1plt.setup_mpl(mpl_timedelta_support=False, color_scheme="fastf1")

ses = fastf1.get_session(2024, "Japan", "Q")
ses.load()

D1, D2 = "VER", "NOR"
lap1 = ses.laps.pick_drivers(D1).pick_fastest()
lap2 = ses.laps.pick_drivers(D2).pick_fastest()

t1 = lap1.get_car_data().add_distance()
t2 = lap2.get_car_data().add_distance()

delta, ref, cmp_ = delta_time(lap1, lap2)

s1 = f1plt.get_driver_style(D1, ["color", "linestyle"], session=ses)
s2 = f1plt.get_driver_style(D2, ["color", "linestyle"], session=ses)

fig, ax = plt.subplots(5, 1, figsize=(13, 11), sharex=True,
                       gridspec_kw={"height_ratios": [3, 2, 2, 2, 2]})

ax[0].plot(t1["Distance"], t1["Speed"], label=D1, **s1)
ax[0].plot(t2["Distance"], t2["Speed"], label=D2, **s2)
ax[0].set_ylabel("Speed [km/h]"); ax[0].legend()

ax[1].plot(ref["Distance"], delta, color="white")
ax[1].axhline(0, color="grey", lw=0.8)
ax[1].set_ylabel(f"Delta {D2} zu {D1} [s]")

ax[2].plot(t1["Distance"], t1["Throttle"], **s1)
ax[2].plot(t2["Distance"], t2["Throttle"], **s2)
ax[2].set_ylabel("Throttle [%]")

ax[3].plot(t1["Distance"], t1["Brake"].astype(int), **s1)
ax[3].plot(t2["Distance"], t2["Brake"].astype(int), **s2)
ax[3].set_ylabel("Brake")

ax[4].plot(t1["Distance"], t1["nGear"], **s1)
ax[4].plot(t2["Distance"], t2["nGear"], **s2)
ax[4].set_ylabel("Gang"); ax[4].set_xlabel("Distanz [m]")

fig.suptitle(f"{ses.event['EventName']} Quali - {D1} vs {D2}")
plt.tight_layout()
plt.show()
