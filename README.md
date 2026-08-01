# F1 Data Analysis

Analyse von Formel-1-Renndaten mit [FastF1](https://github.com/theOehrly/Fast-F1) —
Timing, Telemetrie, Reifenstrategie, Machine Learning und Data Engineering.

Die Bibliothek liefert Rohdaten aus dem offiziellen Live-Timing-Feed: Rundenzeiten,
Sektorzeiten, Positionsdaten im Zehntelsekundentakt, Telemetriekanäle für Speed,
Gas, Bremse, Gang und DRS. Was daraus wird, hängt davon ab, wie man sie behandelt.
Dieses Repository ist meine Antwort darauf: **34 eigenständige Analysen über zwölf
Themenfelder**, jede mit lauffähigem Code und dokumentiertem Vorgehen.

```bash
git clone https://github.com/JanikSchala/F1.git && cd F1
./setup.sh
source .venv/bin/activate
python 01_grundlagen/p01_session_explorer_jede_session_der_f1_historie_la.py
```

---

## Was drinsteckt

### Telemetrie auf Streckenebene

![Gangwechsel-Karte Spa 2024](assets/gangwechsel.png)

Positionsdaten (X/Y) und Telemetrie werden über den Zeitstempel zusammengeführt und
als `LineCollection` eingefärbt. Verstappens schnellste Runde in Spa 2024:
**1:53.159** über **6963 m**, aufgelöst in 844 Messpunkte. Man sieht die
Kemmel-Gerade im achten Gang ebenso wie die Bus-Stop-Schikane im zweiten.

*Code: [`03_telemetrie/p09_gangwechsel_karte_der_strecke.py`](03_telemetrie/p09_gangwechsel_karte_der_strecke.py)*

---

### Fahrervergleich mit Zeitdelta

![Telemetrie-Overlay Suzuka 2024](assets/telemetrie_overlay.png)

Verstappen gegen Norris im Qualifying von Suzuka 2024, **0.292 s** Unterschied auf
der Runde. Das kumulierte Delta zeigt, *wo* die Zeit entsteht: Norris liegt zwischen
Meter 800 und 1600 leicht vorn, verliert dann ab Meter 1700 kontinuierlich. Bei
Höchstgeschwindigkeit trennen die beiden 5 km/h (324 zu 319).

Berechnet mit `fastf1.utils.delta_time()`, das beide Runden über die zurückgelegte
Distanz interpoliert — nötig, weil die Telemetrie nicht synchron abgetastet wird.

*Code: [`03_telemetrie/p07_telemetrie_overlay_zwei_schnellste_runden_uebere.py`](03_telemetrie/p07_telemetrie_overlay_zwei_schnellste_runden_uebere.py)*

---

### Race Pace, sauber gerechnet

![Race Pace Barcelona 2024](assets/race_pace.png)

Die naive Auswertung aller Rundenzeiten ist wertlos: Out-Laps, In-Laps,
Safety-Car-Phasen und gestrichene Runden verzerren jede Verteilung. Nach dem Filter
auf grüne Flagge, ohne Boxenrunden, ohne Ausreißer über 107 % bleiben in Barcelona
2024 noch **1113 von 1310 Runden** übrig — 15 % fliegen raus.

Erst danach lässt sich rechnen. Norris war der Schnellste, Leclerc lag **0.059 s**
dahinter, Verstappen **0.177 s**. Die Balken tragen ein Bootstrap-Konfidenzintervall
über 1000 Resamples: Wo sich Intervalle überlappen, ist der Unterschied nicht
belegbar — bei den ersten drei ist genau das der Fall.

*Code: [`02_timing/p04_race_pace_ranking_wer_war_wirklich_am_schnellste.py`](02_timing/p04_race_pace_ranking_wer_war_wirklich_am_schnellste.py)*

---

### Reifenstrategie

![Strategieübersicht Ungarn 2024](assets/strategie.png)

Ungarn 2024, sortiert nach Endposition. Das Spitzenfeld fuhr überwiegend
Medium–Hard–Medium, das Mittelfeld setzte auf Hard–Hard. Im Schnitt 3 Stints pro
Fahrer, mittlere Stintlänge 29 Runden auf Hard gegenüber 7 Runden auf Soft.

*Code: [`05_reifen_strategie/p14_stint_und_strategie_uebersicht_als_gantt_chart.py`](05_reifen_strategie/p14_stint_und_strategie_uebersicht_als_gantt_chart.py)*

---

### Degradation quantifiziert

![Reifendegradation Bahrain 2024](assets/degradation.png)

Rundenzeiten steigen im Stint aus zwei Gründen: der Reifen baut ab, aber das Auto
wird gleichzeitig leichter. Wer beides nicht trennt, unterschätzt die Degradation
systematisch. Hier ist der Treibstoffeffekt herausgerechnet (1.8 kg pro Runde,
0.03 s pro kg), danach wird je Stint eine Regression über das Reifenalter gelegt.

Bahrain 2024, 62 auswertbare Stints: **Soft 0.127 s/Runde**, **Hard 0.092 s/Runde**.
Der Soft baut also rund 40 % schneller ab — bei einer Streuung, die zeigt, wie stark
Fahrstil und Auto die Zahl beeinflussen.

*Code: [`05_reifen_strategie/p13_reifendegradation_modellieren.py`](05_reifen_strategie/p13_reifendegradation_modellieren.py)*

---

## `f1lab` — die wiederverwendbaren Teile

Die 34 Skripte zeigen jeweils eine Analyse. Was mehrfach gebraucht wird, liegt als
installierbares Paket daneben — mit einer bewussten Trennung:

```
f1lab/core.py       reine Rechnung auf numpy-Arrays, ohne Netzzugriff testbar
f1lab/session.py    FastF1-Anbindung: laden, filtern, aggregieren
```

Diese Trennung ist der Grund, warum sich das Ganze überhaupt testen lässt. Wäre die
Degradationsschätzung fest mit `session.laps` verdrahtet, bräuchte jeder Test eine
Internetverbindung und eine geladene Session. So bekommt `fit_degradation()` zwei
Arrays und liefert eine Steigung — prüfbar gegen synthetische Daten mit bekannter
Wahrheit.

```python
import f1lab

ses = f1lab.load(2024, "Spain", "R")
print(f1lab.pace_table(ses).head())
print(f1lab.degradation_by_compound(ses))
print(f"Pitloss: {f1lab.pit_loss(ses):.2f} s")
```

**43 Tests, alle ohne Netzzugriff:**

```bash
pip install -e ".[dev]"
pytest -q
```

Die Tests prüfen nicht, ob der Code läuft, sondern ob er *richtig rechnet*. Ein paar
Beispiele für die Art von Aussage, die dahintersteht:

- Bei konstantem Reifen und reinem Spritverbrauch muss `fuel_correct()` die Zeiten
  vollständig flach ziehen — Standardabweichung null.
- `bootstrap_median()` muss bei einem einzelnen Extremausreißer stabil bleiben.
  Genau deswegen steht dort der Median und nicht der Mittelwert.
- `find_cliff()` darf bei einem linearen Stint *keinen* Knick melden, und einen
  flacher werdenden Verlauf nicht als Cliff durchgehen lassen.
- `braking_zones()` muss eine Bremszone erkennen, die direkt am Anfang des Arrays
  beginnt — die Flankenerkennung über `np.diff` verliert die sonst.

Der letzte Punkt ist kein hypothetisches Beispiel: der Test hat beim ersten Lauf
einen Off-by-one in der Kantenerkennung gefunden. Die Zone endete eine Probe zu
spät, wodurch die Ausgangsgeschwindigkeit gleich der Eingangsgeschwindigkeit war.

---

## Aufbau

```
f1lab/                installierbares Paket, core (rein) + session (FastF1)
tests/                43 Tests, laufen offline

01_grundlagen/        Datenzugriff, Caching, Kalender als Dimensionstabelle
02_timing/            Rundenzeiten, Pace-Ranking, Sektoren, Positionsverlauf
03_telemetrie/        Speed, Bremspunkte, Gänge, DRS, Starts, Dirty Air
04_strecke/           Streckenkarten, Kurvenprofile, Layout-Vergleich über Jahre
05_reifen_strategie/  Degradation, Stints, Undercut-Simulation, Boxenstopps
06_wetter/            Temperatur- und Regeneffekte auf die Pace
07_race_control/      Safety Car, Strafen, Track Limits
08_historie/          Ergast-API, WM-Simulation, 75 Jahre Trendanalyse
09_machine_learning/  Qualifying-Vorhersage, Fahrstil-Clustering, Anomalien
10_data_engineering/  DuckDB-Warehouse, REST-API, CLI-Paket
11_visualisierung/    Streamlit-Dashboard, automatischer PDF-Rennbericht
12_live_timing/       Echtzeit-Aufzeichnung des Timing-Streams

common/               geteilte Helfer: Cache, Rundenfilter, Fuel-Korrektur
make_assets.py        erzeugt die Grafiken oben
check_setup.py        prüft Umgebung, Pakete, Cache und API-Zugriff
```

Jedes Skript läuft eigenständig. Der Kopf-Docstring nennt Ziel, Vorgehen, die
genutzten FastF1-Bausteine und eine Ausbaustufe.

---

## Methodische Entscheidungen

Ein paar Dinge, die den Unterschied zwischen einer hübschen Grafik und einer
belastbaren Aussage ausmachen:

**Rundenfilterung.** `pick_track_status("1")` behält nur Runden unter grüner Flagge.
Ohne das mischen sich Safety-Car-Runden in die Verteilung und verschieben den Median
um mehrere Zehntel. Dazu `pick_wo_box()`, `pick_accurate()` und `pick_not_deleted()`,
zusammengefasst in `common.clean_laps()`.

**Treibstoffkorrektur.** Über eine Renndistanz summiert sich der Effekt auf mehrere
Sekunden pro Runde. Die verwendeten 0.03 s/kg sind ein Literaturwert, kein gemessener
— die Größenordnung stimmt, die dritte Nachkommastelle sollte man nicht
überinterpretieren.

**Unsicherheit ausweisen.** Ein Median ohne Streuungsmaß suggeriert Präzision, die
nicht da ist. Deswegen Bootstrap-Intervalle statt nackter Zahlen.

**Zeitliche Validierung beim ML.** Rennen kommen chronologisch. Ein zufälliger
Train-Test-Split würde mit Zukunftswissen trainieren, deshalb `TimeSeriesSplit`.

---

## Grenzen

Die Analysen kontrollieren nicht für Verkehr, Motormodus oder Tankstrategie. Ein
Fahrer, der 20 Runden im Windschatten festhing, sieht langsamer aus, als er war —
Projekt 32 nähert sich dem über `DistanceToDriverAhead`, löst es aber nicht.
Setup-Unterschiede sind aus der öffentlichen Telemetrie ohnehin nicht rekonstruierbar.

Telemetriedaten existieren erst ab Saison **2018**. Ergebnisse und Rundenzeiten
reichen über die Ergast-kompatible API zurück bis 1950.

---

## Setup

```bash
./setup.sh                    # venv anlegen, Pakete installieren
source .venv/bin/activate
python check_setup.py         # prüft Python, Pakete, Cache, API-Zugriff
```

Der erste Ladevorgang einer Session dauert 30–120 Sekunden, danach kommt alles aus
`~/f1_cache`. Für reine Rundenzeitanalysen ist
`session.load(telemetry=False, weather=False, messages=False)` deutlich schneller.

Session-Kürzel: `FP1` `FP2` `FP3` `Q` `S` (Sprint) `SQ` `R`.
Events per Name (`"Monza"`), Land (`"Italy"`) oder Rundennummer (`14`).

---

## Projektindex

| ID | Projekt | Kategorie | Niveau |
|----|---------|-----------|--------|
| `P01` | [Session-Explorer: Jede Session der F1-Historie laden](01_grundlagen/p01_session_explorer_jede_session_der_f1_historie_la.py) | Grundlagen | Einsteiger |
| `P02` | [Saison-Kalender und Event-Metadaten als Datenbank](01_grundlagen/p02_saison_kalender_event_metadaten_als_datenbank.py) | Grundlagen | Einsteiger |
| `P03` | [Rundenzeit-Qualitätsfilter: Was ist eine saubere Runde?](02_timing/p03_rundenzeit_qualitaetsfilter_was_ist_eine_saubere.py) | Timing | Einsteiger |
| `P04` | [Race Pace Ranking: Wer war wirklich am schnellsten?](02_timing/p04_race_pace_ranking_wer_war_wirklich_am_schnellste.py) | Timing | Fortgeschritten |
| `P05` | [Teamkollegen-Duell über eine ganze Saison](02_timing/p05_teamkollegen_duell_ueber_eine_ganze_saison.py) | Timing | Fortgeschritten |
| `P06` | [Sektor-Analyse: Wo genau geht die Zeit verloren?](02_timing/p06_sektor_analyse_wo_genau_geht_die_zeit_verloren.py) | Timing | Einsteiger |
| `P07` | [Telemetrie-Overlay: Zwei schnellste Runden übereinanderlegen](03_telemetrie/p07_telemetrie_overlay_zwei_schnellste_runden_uebere.py) | Telemetrie | Fortgeschritten |
| `P08` | [Bremspunkt-Detektor und Bremsphasen-Report](03_telemetrie/p08_bremspunkt_detektor_und_bremsphasen_report.py) | Telemetrie | Profi |
| `P09` | [Gangwechsel-Karte der Strecke](03_telemetrie/p09_gangwechsel_karte_der_strecke.py) | Telemetrie | Fortgeschritten |
| `P10` | [DRS-Nutzung und Topspeed-Analyse](03_telemetrie/p10_drs_nutzung_und_topspeed_analyse.py) | Telemetrie | Fortgeschritten |
| `P11` | [Streckenkarte mit nummerierten Kurven](04_strecke/p11_streckenkarte_mit_nummerierten_kurven.py) | Strecke | Einsteiger |
| `P12` | [Kurvengeschwindigkeits-Profil je Fahrer](04_strecke/p12_kurvengeschwindigkeits_profil_je_fahrer.py) | Strecke | Profi |
| `P13` | [Reifendegradation modellieren](05_reifen_strategie/p13_reifendegradation_modellieren.py) | Reifen | Fortgeschritten |
| `P14` | [Stint- und Strategie-Übersicht als Gantt-Chart](05_reifen_strategie/p14_stint_und_strategie_uebersicht_als_gantt_chart.py) | Reifen | Einsteiger |
| `P15` | [Undercut-Simulator: Wann lohnt sich der frühere Stopp?](05_reifen_strategie/p15_undercut_simulator_wann_lohnt_sich_der_frueher_s.py) | Reifen | Profi |
| `P16` | [Boxenstopp-Performance-Ranking der Teams](05_reifen_strategie/p16_boxenstopp_performance_ranking_der_teams.py) | Reifen | Fortgeschritten |
| `P17` | [Wetter-Impact: Regen und Streckentemperatur](06_wetter/p17_wetter_impact_wie_regen_und_streckentemperatur_d.py) | Wetter | Fortgeschritten |
| `P18` | [Safety-Car- und Track-Status-Chronik](07_race_control/p18_safety_car_und_track_status_chronik.py) | Race Control | Fortgeschritten |
| `P19` | [Race Control Messages: Strafen automatisch auswerten](07_race_control/p19_race_control_messages_strafen_und_untersuchungen.py) | Race Control | Fortgeschritten |
| `P20` | [Positionsverlauf und Überholmatrix](02_timing/p20_positionsverlauf_und_ueberholmatrix.py) | Timing | Fortgeschritten |
| `P21` | [WM-Stand-Simulator: Wer kann noch Weltmeister werden?](08_historie/p21_wm_stand_simulator_wer_kann_noch_weltmeister_wer.py) | Historie | Fortgeschritten |
| `P22` | [75 Jahre F1: Historische Trendanalyse](08_historie/p22_75_jahre_f1_historische_trendanalyse.py) | Historie | Fortgeschritten |
| `P23` | [Qualifying-Ergebnis vorhersagen (Machine Learning)](09_machine_learning/p23_qualifying_ergebnis_vorhersagen_machine_learning.py) | ML | Profi |
| `P24` | [Fahrstil-Clustering: Wer fährt wie?](09_machine_learning/p24_fahrstil_clustering_wer_faehrt_wie.py) | ML | Profi |
| `P25` | [Anomalie-Erkennung: Technische Probleme aus Telemetrie](09_machine_learning/p25_anomalie_erkennung_technische_probleme_aus_telem.py) | ML | Profi |
| `P26` | [F1-Data-Warehouse: Sternschema in DuckDB](10_data_engineering/p26_f1_data_warehouse_sternschema_in_duckdb.py) | Engineering | Profi |
| `P27` | [Telemetrie-API mit FastAPI](10_data_engineering/p27_telemetrie_api_mit_fastapi.py) | Engineering | Profi |
| `P28` | [Interaktives Streamlit-Dashboard](11_visualisierung/p28_interaktives_streamlit_dashboard.py) | Visualisierung | Fortgeschritten |
| `P29` | [Automatischer Rennbericht als PDF](11_visualisierung/p29_automatischer_rennbericht_als_pdf.py) | Visualisierung | Fortgeschritten |
| `P30` | [Live-Timing aufzeichnen und auswerten](12_live_timing/p30_live_timing_aufzeichnen_und_in_echtzeit_auswerte.py) | Live Timing | Profi |
| `P31` | [Startphasen-Analyse: Wer gewinnt die ersten 500 Meter?](03_telemetrie/p31_startphasen_analyse_wer_gewinnt_die_ersten_500_m.py) | Telemetrie | Fortgeschritten |
| `P32` | [Verfolgungsjagd: Abstand zum Vordermann und Dirty Air](03_telemetrie/p32_verfolgungsjagd_abstand_zum_vordermann_und_dirty.py) | Telemetrie | Profi |
| `P33` | [Streckenvergleich über Jahre: Hat sich das Layout geändert?](04_strecke/p33_streckenvergleich_ueber_jahre_hat_sich_das_layou.py) | Strecke | Fortgeschritten |
| `P34` | [Der komplette Wochenend-Analyzer als CLI-Tool](10_data_engineering/p34_der_komplette_wochenend_analyzer_als_cli_tool.py) | Engineering | Profi |

---

## Reproduzieren

Alle Grafiken in diesem README entstehen aus echten Daten:

```bash
python make_assets.py
```

Schreibt die PNGs nach `assets/` und die berechneten Kennzahlen nach
`assets/kennzahlen.json` — die Zahlen im Text oben stammen genau daher.

---

## Hinweis

FastF1 ist ein inoffizielles Open-Source-Projekt und steht in keiner Verbindung zu
den Formel-1-Gesellschaften. F1, FORMULA ONE, FORMULA 1, FIA FORMULA ONE WORLD
CHAMPIONSHIP, GRAND PRIX und verwandte Marken sind Marken der Formula One
Licensing B.V.
