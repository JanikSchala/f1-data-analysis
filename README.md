# F1 Data Portfolio

34 eigenstaendige Analyseprojekte auf Basis von
[FastF1](https://github.com/theOehrly/Fast-F1) - von Timing und Telemetrie
ueber Reifenstrategie bis zu Machine Learning, Data Engineering und Live Timing.

Jedes Skript laeuft fuer sich allein. Kopf-Docstring erklaert Ziel, Vorgehen,
genutzte FastF1-Bausteine und eine Ausbaustufe.

---

## Schnellstart

```bash
# 1) Einrichtung (einmalig)
./setup.sh                # oder: Doppelklick auf "1_Setup_starten.command"

# 2) Umgebung aktivieren
source .venv/bin/activate

# 3) Pruefen, ob alles passt
python check_setup.py

# 4) Erstes Projekt
python 01_grundlagen/p01_session_explorer.py
```

In VS Code: Ordner oeffnen, unten rechts den Interpreter `.venv` waehlen,
Datei oeffnen, `F5` druecken.

---

## Aufbau

```
f1-portfolio/
  common/            gemeinsame Helfer (Cache, Rundenfilter, Fuel-Korrektur)
  01_grundlagen/     Datenzugriff, Caching, Kalender
  02_timing/         Rundenzeiten, Pace, Sektoren, Positionen
  03_telemetrie/     Speed, Bremspunkte, Gaenge, DRS, Starts, Dirty Air
  04_strecke/        Streckenkarten, Kurvenprofile, Layout-Vergleich
  05_reifen_strategie/  Degradation, Stints, Undercut, Boxenstopps
  06_wetter/         Temperatur- und Regeneffekte
  07_race_control/   Safety Car, Strafen, Track Limits
  08_historie/       Ergast/jolpica, WM-Simulation, 75 Jahre F1
  09_machine_learning/  Vorhersage, Clustering, Anomalien
  10_data_engineering/  Warehouse, REST-API, CLI-Paket
  11_visualisierung/    Streamlit-Dashboard, PDF-Report
  12_live_timing/       Echtzeit-Aufzeichnung
  check_setup.py     prueft Python, Pakete, Cache und API-Zugriff
  requirements.txt
```

---

## Wichtig zu wissen

- **Telemetrie gibt es erst ab Saison 2018.** Ergebnisse und Rundenzeiten
  reichen zurueck bis 1950.
- **Der erste Ladevorgang** einer Session dauert 30-120 Sekunden. Danach kommt
  alles aus `~/f1_cache` und laeuft in Sekunden.
- **Session-Kuerzel:** `FP1` `FP2` `FP3` `Q` (Qualifying) `S` (Sprint)
  `SQ` (Sprint-Quali) `R` (Rennen).
- **Events** ansprechen per Name (`"Monza"`), Land (`"Italy"`) oder
  Rundennummer (`14`).
- `session.load(telemetry=False, weather=False, messages=False)` ist deutlich
  schneller, wenn nur Rundenzeiten gebraucht werden.

---

## Roadmap

### Woche 1 - Fundament legen

_Umgebung aufsetzen, Datenmodell verstehen, erste saubere Analyse._

- `P01` Session-Explorer: Jede Session der F1-Historie laden
- `P02` Saison-Kalender & Event-Metadaten als Datenbank
- `P03` Rundenzeit-Qualitaetsfilter: Was ist eine 'saubere' Runde?
- `P14` Stint- und Strategie-Uebersicht als Gantt-Chart

### Woche 2 - Analytische Tiefe

_Von der Beschreibung zur Aussage: Pace, Sektoren, Positionen._

- `P06` Sektor-Analyse: Wo genau geht die Zeit verloren?
- `P04` Race Pace Ranking: Wer war wirklich am schnellsten?
- `P20` Positionsverlauf und Ueberholmatrix
- `P05` Teamkollegen-Duell ueber eine ganze Saison

### Woche 3 - Telemetrie beherrschen

_Das Herzstueck. Ab hier klingst du wie jemand aus dem Fach._

- `P07` Telemetrie-Overlay: Zwei schnellste Runden uebereinanderlegen
- `P09` Gangwechsel-Karte der Strecke
- `P10` DRS-Nutzung und Topspeed-Analyse
- `P08` Bremspunkt-Detektor und Bremsphasen-Report
- `P31` Startphasen-Analyse: Wer gewinnt die ersten 500 Meter?

### Woche 4 - Strategie & Domaenenmodelle

_Reifen, Wetter, Safety Cars - die Sprache der Boxenmauer._

- `P13` Reifendegradation modellieren
- `P15` Undercut-Simulator: Wann lohnt sich der frueher Stopp?
- `P17` Wetter-Impact: Wie Regen und Streckentemperatur die Pace veraendern
- `P18` Safety-Car- und Track-Status-Chronik
- `P19` Race Control Messages: Strafen und Untersuchungen automatisch auswerten
- `P16` Boxenstopp-Performance-Ranking der Teams

### Woche 5 - Spezialisierung

_Waehle nach Zielrolle: ML, Engineering oder Strategie._

- `P11` Streckenkarte mit nummerierten Kurven
- `P12` Kurvengeschwindigkeits-Profil je Fahrer
- `P23` Qualifying-Ergebnis vorhersagen (Machine Learning)
- `P24` Fahrstil-Clustering: Wer faehrt wie?
- `P25` Anomalie-Erkennung: Technische Probleme aus Telemetrie erkennen
- `P26` F1-Data-Warehouse: Sternschema in DuckDB
- `P27` Telemetrie-API mit FastAPI
- `P32` Verfolgungsjagd: Abstand zum Vordermann und Dirty Air

### Woche 6 - Vorzeigbar machen

_Alles buendeln in Artefakte, die man in 5 Minuten vorfuehren kann._

- `P21` WM-Stand-Simulator: Wer kann noch Weltmeister werden?
- `P22` 75 Jahre F1: Historische Trendanalyse
- `P33` Streckenvergleich ueber Jahre: Hat sich das Layout geaendert?
- `P28` Interaktives Streamlit-Dashboard
- `P29` Automatischer Rennbericht als PDF
- `P30` Live-Timing aufzeichnen und in Echtzeit auswerten
- `P34` Der komplette Wochenend-Analyzer als CLI-Tool

---

## Projektindex

| ID | Projekt | Kategorie | Niveau | Aufwand |
|----|---------|-----------|--------|---------|
| `P01` | [Session-Explorer: Jede Session der F1-Historie laden](01_grundlagen/p01_session_explorer_jede_session_der_f1_historie_la.py) | Grundlagen & Datenzugriff | Einsteiger | 1-2 h |
| `P02` | [Saison-Kalender & Event-Metadaten als Datenbank](01_grundlagen/p02_saison_kalender_event_metadaten_als_datenbank.py) | Grundlagen & Datenzugriff | Einsteiger | 2 h |
| `P03` | [Rundenzeit-Qualitaetsfilter: Was ist eine 'saubere' Runde?](02_timing/p03_rundenzeit_qualitaetsfilter_was_ist_eine_saubere.py) | Timing & Rundenanalyse | Einsteiger | 2-3 h |
| `P04` | [Race Pace Ranking: Wer war wirklich am schnellsten?](02_timing/p04_race_pace_ranking_wer_war_wirklich_am_schnellste.py) | Timing & Rundenanalyse | Fortgeschritten | 3-4 h |
| `P05` | [Teamkollegen-Duell ueber eine ganze Saison](02_timing/p05_teamkollegen_duell_ueber_eine_ganze_saison.py) | Timing & Rundenanalyse | Fortgeschritten | 4-5 h |
| `P06` | [Sektor-Analyse: Wo genau geht die Zeit verloren?](02_timing/p06_sektor_analyse_wo_genau_geht_die_zeit_verloren.py) | Timing & Rundenanalyse | Einsteiger | 2 h |
| `P07` | [Telemetrie-Overlay: Zwei schnellste Runden uebereinanderlegen](03_telemetrie/p07_telemetrie_overlay_zwei_schnellste_runden_uebere.py) | Telemetrie | Fortgeschritten | 3 h |
| `P08` | [Bremspunkt-Detektor und Bremsphasen-Report](03_telemetrie/p08_bremspunkt_detektor_und_bremsphasen_report.py) | Telemetrie | Profi | 4-5 h |
| `P09` | [Gangwechsel-Karte der Strecke](03_telemetrie/p09_gangwechsel_karte_der_strecke.py) | Telemetrie | Fortgeschritten | 2-3 h |
| `P10` | [DRS-Nutzung und Topspeed-Analyse](03_telemetrie/p10_drs_nutzung_und_topspeed_analyse.py) | Telemetrie | Fortgeschritten | 3 h |
| `P11` | [Streckenkarte mit nummerierten Kurven](04_strecke/p11_streckenkarte_mit_nummerierten_kurven.py) | Strecke & Position | Einsteiger | 2 h |
| `P12` | [Kurvengeschwindigkeits-Profil je Fahrer](04_strecke/p12_kurvengeschwindigkeits_profil_je_fahrer.py) | Strecke & Position | Profi | 4 h |
| `P13` | [Reifendegradation modellieren](05_reifen_strategie/p13_reifendegradation_modellieren.py) | Reifen & Strategie | Fortgeschritten | 4-5 h |
| `P14` | [Stint- und Strategie-Uebersicht als Gantt-Chart](05_reifen_strategie/p14_stint_und_strategie_uebersicht_als_gantt_chart.py) | Reifen & Strategie | Einsteiger | 2-3 h |
| `P15` | [Undercut-Simulator: Wann lohnt sich der frueher Stopp?](05_reifen_strategie/p15_undercut_simulator_wann_lohnt_sich_der_frueher_s.py) | Reifen & Strategie | Profi | 5-6 h |
| `P16` | [Boxenstopp-Performance-Ranking der Teams](05_reifen_strategie/p16_boxenstopp_performance_ranking_der_teams.py) | Reifen & Strategie | Fortgeschritten | 3 h |
| `P17` | [Wetter-Impact: Wie Regen und Streckentemperatur die Pace veraendern](06_wetter/p17_wetter_impact_wie_regen_und_streckentemperatur_d.py) | Wetter & Bedingungen | Fortgeschritten | 3-4 h |
| `P18` | [Safety-Car- und Track-Status-Chronik](07_race_control/p18_safety_car_und_track_status_chronik.py) | Race Control & Regeln | Fortgeschritten | 3 h |
| `P19` | [Race Control Messages: Strafen und Untersuchungen automatisch auswerten](07_race_control/p19_race_control_messages_strafen_und_untersuchungen.py) | Race Control & Regeln | Fortgeschritten | 3-4 h |
| `P20` | [Positionsverlauf und Ueberholmatrix](02_timing/p20_positionsverlauf_und_ueberholmatrix.py) | Timing & Rundenanalyse | Fortgeschritten | 3 h |
| `P21` | [WM-Stand-Simulator: Wer kann noch Weltmeister werden?](08_historie/p21_wm_stand_simulator_wer_kann_noch_weltmeister_wer.py) | Historie & Ergast-API | Fortgeschritten | 3-4 h |
| `P22` | [75 Jahre F1: Historische Trendanalyse](08_historie/p22_75_jahre_f1_historische_trendanalyse.py) | Historie & Ergast-API | Fortgeschritten | 4-5 h |
| `P23` | [Qualifying-Ergebnis vorhersagen (Machine Learning)](09_machine_learning/p23_qualifying_ergebnis_vorhersagen_machine_learning.py) | Machine Learning | Profi | 6-8 h |
| `P24` | [Fahrstil-Clustering: Wer faehrt wie?](09_machine_learning/p24_fahrstil_clustering_wer_faehrt_wie.py) | Machine Learning | Profi | 5-6 h |
| `P25` | [Anomalie-Erkennung: Technische Probleme aus Telemetrie erkennen](09_machine_learning/p25_anomalie_erkennung_technische_probleme_aus_telem.py) | Machine Learning | Profi | 5 h |
| `P26` | [F1-Data-Warehouse: Sternschema in DuckDB](10_data_engineering/p26_f1_data_warehouse_sternschema_in_duckdb.py) | Data Engineering | Profi | 8-10 h |
| `P27` | [Telemetrie-API mit FastAPI](10_data_engineering/p27_telemetrie_api_mit_fastapi.py) | Data Engineering | Profi | 6-8 h |
| `P28` | [Interaktives Streamlit-Dashboard](11_visualisierung/p28_interaktives_streamlit_dashboard.py) | Visualisierung & Apps | Fortgeschritten | 5-6 h |
| `P29` | [Automatischer Rennbericht als PDF](11_visualisierung/p29_automatischer_rennbericht_als_pdf.py) | Visualisierung & Apps | Fortgeschritten | 4-5 h |
| `P30` | [Live-Timing aufzeichnen und in Echtzeit auswerten](12_live_timing/p30_live_timing_aufzeichnen_und_in_echtzeit_auswerte.py) | Live Timing | Profi | 6-8 h |
| `P31` | [Startphasen-Analyse: Wer gewinnt die ersten 500 Meter?](03_telemetrie/p31_startphasen_analyse_wer_gewinnt_die_ersten_500_m.py) | Telemetrie | Fortgeschritten | 3-4 h |
| `P32` | [Verfolgungsjagd: Abstand zum Vordermann und Dirty Air](03_telemetrie/p32_verfolgungsjagd_abstand_zum_vordermann_und_dirty.py) | Telemetrie | Profi | 4-5 h |
| `P33` | [Streckenvergleich ueber Jahre: Hat sich das Layout geaendert?](04_strecke/p33_streckenvergleich_ueber_jahre_hat_sich_das_layou.py) | Strecke & Position | Fortgeschritten | 3-4 h |
| `P34` | [Der komplette Wochenend-Analyzer als CLI-Tool](10_data_engineering/p34_der_komplette_wochenend_analyzer_als_cli_tool.py) | Data Engineering | Profi | 8-10 h |

---

## Hinweis

FastF1 ist ein inoffizielles Open-Source-Projekt und steht in keiner Verbindung
zu den Formel-1-Gesellschaften. F1, FORMULA ONE, FORMULA 1, FIA FORMULA ONE
WORLD CHAMPIONSHIP, GRAND PRIX und verwandte Marken sind Marken der
Formula One Licensing B.V.
