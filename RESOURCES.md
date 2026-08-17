# Ressourcen

Kuratierte Liste externer F1- und Motorsport-Ressourcen — APIs, Datensätze,
Telemetrie-Tools, Dashboards, Sim Racing, Lernmaterial. Übernommen aus
[subinium/awesome-f1](https://github.com/subinium/awesome-f1) (Stand: 2026-08-14),
zur eigenen Referenz neben den 34 Analysen in diesem Repo.

---

## Formel 1

### APIs und Libraries

- [FastF1](https://github.com/theOehrly/Fast-F1) - Python-Library für Timing, Telemetrie, Wetter, Sessions und Strategie-Analyse. Bereits Basis dieses Projekts.
- [FastF1 Docs](https://docs.fastf1.dev/) - Primäre Dokumentation für F1-Analyse-Workflows in Python.
- [OpenF1](https://openf1.org/) - Offene API für Rundendaten, Positionen, Team-Funk, Stints, Wetter, Sessions.
- [OpenF1 GitHub](https://github.com/br-g/openf1) - Quellcode und Docs zu OpenF1.
- [Jolpica-F1](https://github.com/jolpica/jolpica-f1) - Ergast-kompatible F1-API, Community-Standard seit der Ergast-Abschaltung. Bereits in P16/P21/P22 genutzt.
- [Jolpica API](https://api.jolpi.ca/ergast/f1/) - Gehosteter Endpunkt für historische F1-Daten.
- [F1DB](https://github.com/f1db/f1db) - Open-Source-F1-Datenbank (CSV/JSON/SQL/SQLite) mit Fahrern, Konstrukteuren, Strecken/Layouts, Saisons, Ständen, Ergebnissen, Startaufstellungen, schnellsten Runden, Boxenstopps.
- [fia-doc](https://github.com/harningle/fia-doc) - Python-Parser für strukturierte Renndaten aus offiziellen FIA-F1-PDFs (Nennlisten, Reifenmischungen, Quali-/Renn-Klassifikationen, Rundenzeiten, Boxenstopps).
- [LiveF1](https://github.com/GoktugOcal/LiveF1) - Python-Toolkit für Echtzeit- und historische F1-Daten mit Medallion-Architektur-ETL.
- [f1dataR](https://github.com/SCasanova/f1dataR) - R-Paket, wrappt FastF1 und Jolpica. Standard-R-Schnittstelle für F1-Daten.
- [f1api.dev](https://f1api.dev/) - Entwicklerorientierte F1-API und SDK-Oberfläche. [GitHub](https://github.com/rafacv23/f1-api).
- [F1PyStats](https://github.com/alec-kr/F1PyStats) - Python-Paket für Stände, Rennergebnisse und historische Zusammenfassungen.
- [API-Sports Formula 1](https://api-sports.io/documentation/formula-1/v1) - Kommerzielle API mit Free-Tier (100 Requests/Tag), Saisons, Stände, Ergebnisse.

### Datensätze und Telemetrie-Archive

- [Tracing Insights Archive](https://github.com/TracingInsights-Archive) - Saison-für-Saison-Telemetrie-CSV-Archive.
- [Tracing Insights Data](https://tracinginsights.com/data/) - Download-Hub für Telemetrie- und Session-Daten.
- [TracingInsights on HuggingFace](https://huggingface.co/datasets/tracinginsights/RaceData) - Vollständiger Ergast-Schema-Mirror, 20+ CSV-Dateien, aktualisiert innerhalb 3h nach jedem Rennen.
- [renumics/f1_dataset](https://huggingface.co/datasets/renumics/f1_dataset) - Montreal-2023-Telemetrie mit 40+ Spalten inkl. 882-dimensionalen Embeddings für ML.
- [formula1-datasets](https://github.com/toUpperCase78/formula1-datasets) - Saison-CSV-Datensätze und Jupyter-Notebooks für jede F1-Saison 2019-2026, aktiv gepflegt.
- [f1-circuits](https://github.com/bacinger/f1-circuits) - F1-Strecken als GeoJSON mit Layout und Koordinaten. 35+ von 77 historischen Strecken.
- [TUMFTM/racetrack-database](https://github.com/TUMFTM/racetrack-database) - Mittellinien, Streckenbreiten und berechnete Ideallinien für 20+ Strecken (TU München).
- [RelBench rel-f1](https://relbench.stanford.edu/datasets/rel-f1/) - Stanford Relational-ML-Benchmark, 9 Tabellen, 97K+ Zeilen, 6 ML-Tasks, 1950-heute. NeurIPS 2024.

### Kaggle-Datensätze

- [F1 World Championship 1950-2024](https://www.kaggle.com/datasets/rohanrao/formula-1-world-championship-1950-2020) - Kanonischer Kaggle-F1-Datensatz. 14 CSV-Tabellen aus Ergast. 153K+ Downloads.
- [Formula 1 Race Data](https://www.kaggle.com/datasets/jtrotman/formula-1-race-data) - Aktiv gepflegt nach der Ergast-Abschaltung via Jolpica, 1950 bis aktuell. 7.7K+ Downloads.
- [F1 Drivers Dataset](https://www.kaggle.com/datasets/dubradave/formula-1-drivers-dataset) - Alle F1-Fahrer aus Wikipedia mit Karrierestatistiken. 4.6K+ Downloads.

### Live-Timing

- [undercut-f1](https://github.com/JustAman62/undercut-f1) - TUI-Live-Timing mit variabler Verzögerung (TV-Sync), Session-Replay, Team-Funk-Transkription.
- [MultiViewer for F1](https://multiviewer.app) - Desktop-App für synchronisiertes Live-Timing, Mini-Sektoren, Speeds. Benötigt F1-TV-Abo.
- [F1 Pitwall](https://f1pitwall.fun/) - Live-Rennreplay und Telemetrie-Dashboard im Browser.

### Tools und Apps

- [F1 Replay Timing](https://github.com/adn8naiagent/F1ReplayTiming) - Replay-Viewer mit Timing-Overlays, Telemetrie-Charts, Streckenkarte.
- [f1_sensor](https://github.com/Nicxe/f1_sensor) - Home-Assistant-Integration: nächstes Rennen, Kalender, Live-Flaggen, Race-Control, Wetter, Stände, Historie.
- [f1-sensor-live-data-card](https://github.com/Nicxe/f1-sensor-live-data-card) - Home-Assistant-Cards für Live-Telemetrie und Meisterschaftsstände.
- [formulaone-card](https://github.com/marcokreeft87/formulaone-card) - Home-Assistant-Dashboard-Card für Stände, Termine, Ergebnisse (Jolpica oder f1_sensor).
- [Apify F1 Data Extractor](https://apify.com/richard.biros/f1-data-extractor) - Scraper für formula1.com, Renn-/Quali-/Trainings-/Boxenstopp-Daten seit 1950, JSON-Output.
- [F1AppleTV](https://github.com/NoahFetz/F1AppleTV) - F1TV-Client für Apple TV, Multi-Feed, mehrere Audiospuren.
- [BoxBox](https://github.com/BrightDV/BoxBox) - Open-Source F1/Formula-E-App (Flutter). News, Stände, Termine, Offline-Modus, F-Droid.

### Dashboards und Analytics

**Web-Plattformen:**

- [F1 Cosmos](https://f1cosmos.com/) - Umfangreiches F1-Dashboard mit Visualisierungen und Analytics.
- [F1 The Data](https://f1thedata.com/) - Fahrervergleiche, Telemetrie-Analyse, historische Datenexploration.
- [Pitwall](https://pitwall.app/) - F1-Datenbank mit Rundendaten, Boxenstopps, Rennverlauf seit 1950. Web + iOS.
- [F1 DataStop](https://f1datastop.com/) - Rundenzeit-Deltas, Stint-Strategie, Teamkollegen-Vergleich.
- [F1pace](https://f1pace.com/) - Vertiefte F1-Datenanalyse und Visualisierung.
- [BoardF1](https://boardf1.com/) - Echtzeit-Analytics: Ergebnisse, Rundenzeiten, Reifenstrategie, Team-Pace.
- [Formula-Timer](https://formula-timer.com/) - Live-Timing/Analytics mit Rundenzeit-Aufschlüsselung, Streckenguide.
- [F1 Tempo](https://www.f1-tempo.com/) - Rundenzeiten und Telemetrie im Vergleich.
- [Formula Live Pulse](https://www.f1livepulse.com/) - Live-Timing mit Telemetrie-Charts, 3D-Streckenkarte, KI-Assistent, Team-Funk. F1/F2/F3/F1-Academy.

**Open Source:**

- [F1 Race Replay](https://github.com/IAmTomShaw/f1-race-replay) - Replay-/Telemetrie-App mit Streckenrendering.
- [f1-dash](https://github.com/slowlydev/f1-dash) - Echtzeit-Dashboard (Next.js): Leaderboard, Reifen, Abstände, Mini-Sektoren. Live: [f1-dash.vercel.app](https://f1-dash.vercel.app/).
- [F1ReplayTiming](https://github.com/adn8naiagent/F1ReplayTiming) - Full-Stack Replay/Live-Timing mit Streckenkarte und synchronisiertem Playback.
- [Armchair Strategist](https://github.com/Casper-Guo/Armchair-Strategist) - Strategie-Dashboard für Boxenfenster, Pace-Vergleich, Rennverlauf.
- [f1-live-data](https://github.com/f1stuff/f1-live-data) - Echtzeit-Visualisierung via FastF1-Live-Timing, InfluxDB, Grafana.

### Kalender

- [F1 Calendar](https://github.com/sportstimes/f1) - Open-Source Next.js F1-Kalender mit E-Mail-Erinnerungen und Web-Push. Live: [f1calendar.com](https://f1calendar.com).

### Fantasy F1

- [F1 Fantasy API (Postman docs)](https://documenter.getpostman.com/view/11462073/TzY68Dsi) - REST-API-Doku für die inoffizielle F1-Fantasy-API.
- [Fantasy F1 API Endpoints Cheat Sheet](https://cheatography.com/sertalpbilal/cheat-sheets/fantasy-f1-api-endpoints/) - Endpunkt-Referenz.
- [F1 Fantasy Tools](https://f1fantasytools.com/) - Monte-Carlo-Team-Rechner, Budget-Builder, Live-Scoring.
- [F1 Pitwall.dev](https://f1pitwall.dev/) - Fantasy-Optimierer auf FP-/Quali-Daten mit Monte-Carlo-Konfidenzintervallen, 80.6% Backtest-Trefferquote.

### Simulation und Fahrzeugdynamik

- [fastest-lap](https://github.com/juanmanzanero/fastest-lap) - Fahrzeugdynamik-Simulator für optimale Rundenzeit, 3DOF/6DOF-Modelle (C++/Python).
- [TUMFTM/global_racetrajectory_optimization](https://github.com/TUMFTM/global_racetrajectory_optimization) - Minimalkrümmung/Minimalzeit-Ideallinien-Generierung (TU München), publizierte Forschung.
- [TUMFTM/race-simulation](https://github.com/TUMFTM/race-simulation) - Rundenweise Rennsimulation für Boxenstopp-Strategie mit Reinforcement Learning und Monte-Carlo.

### Historie und Statistik

- [F1 BigData](https://www.bigdataf1.com/) - Historische Statistikdatenbank seit 1950.
- [Formula 1 Archive](https://www.formula1archive.com/) - Komplette F1-Historie: 1.100+ Rennen, 900+ Fahrer, 75+ Saisons.
- [f1metrics](https://f1metrics.wordpress.com/) - Mathematische/statistische F1-Modellierung, Fahrerbewertungen.
- [F1 Analysis](https://f1-analysis.com/) - Mathematische/statistische F1-Analytics mit Vorhersagemodellen.
- [OldRacingCars.com](https://www.oldracingcars.com/f1/) - Chassis-genaue Historien für 3-Liter-F1-Autos (1966-1985).
- [x1z.net](https://x1z.net/) - Durchsuchbarer Index des F1-TV-Archivs mit Rennbewertungen.
- [Formula1Points.com](https://www.formula1points.com/) - Fahrervergleich über verschiedene Punktesysteme.
- [4mula1stats.com](https://www.4mula1stats.com/) - Ergebnisse, Statistiken, Charts seit 1950.

### Offiziell und Referenz

- [F1 Live Timing](https://www.formula1.com/en/timing/f1-live) - Offizieller Live-Timing-Hub.
- [F1 Schedule](https://www.formula1.com/en/racing/2026.html) - Offizieller Kalender.
- [F1 Results](https://www.formula1.com/en/results.html) - Offizielle Ergebnisse und Stände.
- [F1 Teams](https://www.formula1.com/en/teams.html) - Offizielles Team-Verzeichnis.
- [F1 Drivers](https://www.formula1.com/en/drivers.html) - Offizielles Fahrer-Verzeichnis.
- [FIA Formula 1 Regulations](https://www.fia.com/regulation/category/29) - Offizielles Sport- und Technisches Reglement.

### News und Medien

- [Formula 1 Latest](https://www.formula1.com/en/latest) - Offizielle News, Videos, Stände, Ergebnisse.
- [Autosport Formula 1](https://www.autosport.com/f1/news/) - Motorsport-Journalismus seit 1950.
- [Motorsport.com Formula 1](https://www.motorsport.com/f1/news/) - Globale F1-Berichterstattung.
- [The Race Formula 1](https://www.the-race.com/category/formula-1/) - Unabhängige F1-News, Analyse, Podcasts.
- [PlanetF1](https://www.planetf1.com/) - News, Live-Coverage, Stände, Daten, Technik, Forum.
- [RaceFans](https://www.racefans.net/) - Unabhängige Berichterstattung mit Renndaten, Statistiken.
- [RacingNews365](https://racingnews365.com/) - Tägliche F1-News, Kalender, Ergebnisse.
- [Motor Sport Magazine F1](https://www.motorsportmagazine.com/articles/category/single-seaters/f1/) - Langform-Features, Historie, Archive.

### Technische Medien

- [Auto Motor und Sport Formel 1](https://www.auto-motor-und-sport.de/formel-1/aktuell/) - Deutschsprachige F1-Berichterstattung mit technischem Fokus.
- [Racecar Engineering Formula One](https://www.racecar-engineering.com/formulaone/) - Technik-/Engineering-Artikel.
- [Giorgio Piola's F1 Technical Analysis](https://www.motorsport.com/topic/giorgio-piola-s-f1-technical-analysis/23/) - Technische Illustrationen und Analyse.
- [SomersF1](https://somersf1.substack.com/) - Matthew Somerfields technische F1-Analyse.

### Junior- und Nachwuchsserien-Medien

- [Formula Scout](https://formulascout.com/) - Nachwuchs-Formelserien: F2, F3, F4, Road to Indy.
- [Feeder Series](https://feederseries.net/) - Previews, Reviews, Ergebnisse, Podcasts zu Nachwuchsserien.

### Offizielle Session-Quellen

- [FIA F1 Decision Documents](https://www.fia.com/documents/championships/fia-formula-one-world-championship-14/season/season-2026-2072) - Offizielle Rennunterlagen, Steward-Entscheidungen, Zeitpläne, Klassifikationen.
- [FIA F1 Archives](https://api.fia.com/f1-archives) - Offizielles FIA-Saison-/Event-Archiv: Klassifikationen, Timing, Rundenanalyse, Sektoranalyse, Boxenstopps, Wetter.
- [FIA F1 Press Conference Transcripts](https://www.fia.com/news?category=Formula%201) - Offizielle Pressekonferenz-Transkripte.
- [Pirelli F1 Press Area](https://press.pirelli.com/en/?h=1&t=formula+1) - Offizielle Reifenmischungs-Nominierungen, Vorschauen, Infografiken.

### Archiviert / Historische Projekte

- [RaceControl](https://github.com/robvdpol/RaceControl) - Archivierter F1TV-Desktop-Client (Windows, .NET).
- [f1viewer](https://github.com/SoMuchForSubtlety/f1viewer) - Archiviertes TUI für F1TV in Go, VOD und Live via MPV.
- [f1ml](https://github.com/Jared-Chan/f1ml) - Archiviertes Runde-für-Runde F1-Vorhersageprojekt.

## Formel 2, Formel 3 und F1 Academy

- [FIA Formula 2 Event API](https://api.fia.com/events/formula-2-championship/) - FIA-Eventdokumente und Meisterschaftsmetadaten.
- [FIA Formula 2](https://www.fiaformula2.com/) - Offizielle Seite mit [Kalender](https://www.fiaformula2.com/Calendar), [Ständen](https://www.fiaformula2.com/Standings/Driver), [Reglement](https://www.fiaformula2.com/About/DyImndAsBNFcqYOOm4yWS/the-regulations-f2).
- [FIA Formula 3](https://www.fiaformula3.com/) - Offizielle Seite mit [Kalender](https://www.fiaformula3.com/Calendar) und [Reglement](https://www.fiaformula3.com/About/6Iosy860VzDs0INyfKw37E/the-rules-and-regulations-f3).
- [F1 Academy](https://www.f1academy.com/) - Offizielle Seite der reinen Frauen-Nachwuchsserie.
- [F1 Academy YouTube](https://www.youtube.com/@f1academy) - Offizieller Kanal, kostenlose Livestreams von Quali/Rennen.

## Formula E

- [FIA Formula E Event API](https://api.fia.com/events/abb-fia-formula-e-world-championship/) - FIA-Eventdaten.
- [Sportradar Formula E API](https://developer.sportradar.com/racing/reference/formula-e-seasons) - Kommerzielle Formula-E-Coverage.
- [TheSportsDB Formula E](https://www.thesportsdb.com/league/4371-formula-e) - Freier allgemeiner Sport-API-Einstieg.
- [Formula E](https://fiaformulae.com/) - Offizielle Seite. [Reglement](https://fiaformulae.com/en/championship/rules-and-regulations). [Teams](https://fiaformulae.com/en/teams).

## Endurance, WEC, Le Mans, IMSA

### Offizielle Serien und Ergebnisse

- [FIA WEC](https://www.fiawec.com/) - Offizielle Seite. [Reglement](https://www.fiawec.com/en/page/regulations-1). [Event API](https://api.fia.com/events/world-endurance-championship/).
- [24 Hours of Le Mans](https://www.24h-lemans.com/) - Offizielle Le-Mans-Seite.
- [IMSA](https://www.imsa.com/) - Offizielle Seite. [Zeitplan](https://www.imsa.com/weathertech/weathertech-2026-schedule/). [Reglement](https://www.imsa.com/competitors/2025-imsa-rules-regulations/).
- [European Le Mans Series](https://www.europeanlemansseries.com/) - Offizielle ELMS-Seite.
- [GT World Challenge Europe](https://www.gt-world-challenge-europe.com/) - Europäische GT3-Endurance-/Sprintserie.
- [Michelin Le Mans Cup](https://www.lemanscup.com/) - Offizielle Seite für LMP3-/GT3-Supportserie.
- [Asian Le Mans Series](https://www.asianlemansseries.com/) - Offizielle Asian-Le-Mans-Seite.
- [24h Nürburgring Results](https://www.24h-rennen.de/en/results/) - Offizielles Ergebnisarchiv Nürburgring 24h.

### Timing, Telemetrie und Daten

- [Al Kamel Systems](https://alkamelsystems.com/) - Branchenstandard-Timing-Anbieter im Langstreckensport.
- [WEC Timing Results](https://fiawec.alkamelsystems.com/) - Offizielles Al-Kamel-Timing-Archiv für FIA WEC.
- [IMSA GTP Telemetry](https://www.imsa.com/gtp-telemetry/) - Offizielle Telemetrie-Referenz für IMSAs Topklasse.
- [ELMS Timing Results](https://elms.alkamelsystems.com/) - Al-Kamel-Timing-Archiv für ELMS.
- [Le Mans Cup Timing Results](https://lemanscup.alkamelsystems.com/) - Al-Kamel-Timing-Archiv für Michelin Le Mans Cup.
- [Asian Le Mans Series Timing Results](https://alms.alkamelsystems.com/) - Al-Kamel-Timing-Archiv für Asian Le Mans Series.
- [Racing Sports Cars](https://www.racingsportscars.com/) - Historische Sportwagen-Datenbank (Le Mans Series, ALMS, FIA GT, IMSA, Can-Am).
- [World Sports Racing Prototypes](http://www.wsrp.cz/) - Sportwagen-Statistiken, Ergebnisse, Chassis-Historien.

### Medien und Analyse

- [Sportscar365](https://sportscar365.com/) - Sportwagen-News (IMSA, WEC, Le Mans, GT).
- [dailysportscar](https://www.dailysportscar.com/) - Langjähriges Sportwagen-/GT-Magazin.
- [Endurance-Info](https://en.endurance-info.com/) - Endurance-News, Ergebnisse, Live-Coverage.

## WRC und Rally

- [WRC Results API](https://api.wrc.com/results-api) - Stage-/Splitzeiten, Roadbook, Klassifikationen via JSON (undokumentiert, reverse-engineered).
- [RallyDataJunkie](https://rallydatajunkie.com/visualising-wrc-rally-results/) - Online-Buch zu WRC-API, Datenaufbereitung, Visualisierung.
- [eWRC-results.com](https://ewrc-results.com/) - Rally-Datenbank seit 1911, bis hinunter zu nationalen Rallyes.
- [FIA WRC Event API](https://api.fia.com/events/world-rally-championship/) - FIA-Eventdokumente.
- [Sportradar Rally API](https://developer.sportradar.com/racing/reference/rally-overview) - Kommerzielle API mit WRC-Ergebnissen.
- [WRC](https://www.wrc.com/) - Offizielle Seite.
- [FIA World Rallycross](https://www.fiaworldrallycross.com/) - Offizielle WRX-Ergebnisse und -Stände.

## MotoGP und WorldSBK

- [TheSportsDB MotoGP](https://www.thesportsdb.com/league/4407-motogp) - Freie Community-API.
- [TheSportsDB WorldSBK](https://www.thesportsdb.com/league/4454-sbk) - Freie Community-API.
- [MotoGP](https://www.motogp.com/) - Offizielle Seite. [Kalender](https://www.motogp.com/en/calendar/2026).
- [FIM Grand Prix Regulations](https://www.fim-moto.com/en/documents/view/fim-2026-motogp-moto2-moto3-world-championship-regulations) - Offizielles Reglement (PDF).
- [WorldSBK](https://www.worldsbk.com/) - Offizielle Seite.

## NASCAR

- [NASCAR Feed API](https://feed.nascar.com/swagger/ui/index) - Offizielle öffentliche API für Zeitpläne, Nennungen, Stände, Renndaten.
- [nascaR.data](https://cran.r-project.org/web/packages/nascaR.data/index.html) - R-Paket für historische NASCAR-Datensätze (Cup 1949+, Xfinity 1982+, Trucks 1995+).
- [Racing-Reference](https://www.racing-reference.info/) - Umfangreiche NASCAR-/Motorsport-Historie mit Loop-Data.
- [SportsDataIO NASCAR](https://sportsdata.io/nascar-motorsports-api) - Kommerzielle NASCAR-API.
- [Sportradar Racing API](https://developer.sportradar.com/racing) - Kommerzielle Racing-API mit NASCAR-Coverage.
- [NASCAR](https://www.nascar.com/) - Offizielle Seite. [Cup Schedule](https://www.nascar.com/nascar-cup-series/2026/schedule/).

## IndyCar

- [Sportradar IndyCar API](https://developer.sportradar.com/racing/reference/indycar-statistics-summary) - Kommerzielle IndyCar-Coverage.
- [TheSportsDB IndyCar](https://www.thesportsdb.com/league/4373-indycar-series) - Freie allgemeine Sport-API mit IndyCar.
- [IndyCar](https://www.indycar.com/) - Offizielle Seite. [Rulebook](https://epaddock.indycar.com/docs/default-source/rules-regulations-and-policies/indycar-rulebook.pdf).

## Weitere Serien

- [DTM](https://www.dtm.com/en) - Offizielle Seite mit Ergebnissen, Ständen, Terminen.
- [Super Formula](https://superformula.net/sf2/en/) - Offizielle Ergebnisse, Stände, Termine.
- [Super GT Results](https://supergt.net/results?ln=en) - Offizielle GT500-/GT300-Ergebnisse.
- [Supercars Live Timing](https://www.supercars.com/live-timing) - Offizielles Live-Timing.
- [TouringCars.Net Database](https://www.touringcars.net/database/) - BTCC-Daten seit 1979, TCR-Ergebnisse.

## Sim Racing

**Plattformübergreifend**

- [SimHub](https://github.com/SHWotever/SimHub) - Branchenstandard-Multi-Sim-Dashboard, Bass-Shaker-Treiber, Plugin-Plattform.
- [MoTeC i2](https://www.motec.com.au/i2/i2downloads/) - Professionelle Telemetrie-Analyse aus dem realen Motorsport. i2 Standard ist kostenlos.
- [Grafana Simracing Telemetry](https://github.com/alexanderzobnin/grafana-simracing-telemetry) - Grafana-Datenquelle für ACC- und iRacing-Telemetrie.
- [RaceLab](https://racelab.app/) - Kommerzielle Overlay-App mit Free-Tier für iRacing, ACC, rF2, LMU, AMS2, F1.

**iRacing**

- [pyirsdk](https://github.com/kutu/pyirsdk) - Python-3-iRacing-SDK für Live-Telemetrie, Session-Daten, Broadcast-Kommandos.
- [iracingdataapi](https://github.com/jasondilworth56/iracingdataapi) - Python-Wrapper für die iRacing-/data-API.
- [pyracing](https://github.com/Esterni/pyracing) - Async-Python-Client, iRacing-JSON auf strukturierte Objekte gemappt.
- [ibt-telemetry](https://github.com/SkippyZA/ibt-telemetry) - Vollständiger IBT-Parser, plattformübergreifend, Node.js.
- [Cosworth Pi Toolbox](https://www.iracing.com/cosworth-pi-toolbox/) - Professionelle Datenanalyse, liest IBT nativ, kostenlose Lite-Version.
- [iRacing Developers](https://members-ng.iracing.com/data-api/about) - Offizielle iRacing-Data-API-Übersicht.

**ACC**

- [PyAccSharedMemory](https://github.com/rrennoir/PyAccSharedMemory) - Python-Shared-Memory-Reader für ACC (PyPI).
- [acctelemetry](https://github.com/gotzl/acctelemetry) - Anzeige/Analyse von MoTeC-Telemetrie aus ACC.

**F1-Spiel**

- [f1-telemetry-client](https://github.com/racehub-io/f1-telemetry-client) - TypeScript-UDP-Client/Parser, F1 2018 bis F1 23.
- [Pits n' Giggles](https://github.com/ashwin-nat/pits-n-giggles) - Live-Telemetrie-Tool für F1 23/24/25 mit Overlays/Dashboards (Python).

**rFactor 2 / LMU**

- [TinyPedal](https://github.com/TinyPedal/TinyPedal) - Freies Open-Source-Telemetrie-Overlay für rFactor 2 und Le Mans Ultimate.
- [rF2SharedMemoryMapPlugin](https://github.com/TheIronWolfModding/rF2SharedMemoryMapPlugin) - Shared-Memory-Kernplugin für rF2/LMU.
- [LMU Trace](https://lmutrace.com/) - Telemetrie-Analyse-Webtool für Le Mans Ultimate.

**GT7**

- [gt7dashboard](https://github.com/snipem/gt7dashboard) - Live-Dashboard für GT7 mit Rundenvergleich und Performance-Analyse.

## Serienübergreifend und allgemeiner Motorsport

- [TheSportsDB](https://www.thesportsdb.com/) - Breite Sport-API mit Motorsport-Abdeckung.
- [Motorsport Stats](https://motorsportstats.com/) - Größtes Motorsport-Datenrepository, ~3.000 Rennen/Saison, API im Abo.
- [SportsDataIO Motorsport](https://sportsdata.io/motorsports-api) - Kommerzielle Motorsportdaten über Serien hinweg.
- [Sportradar Racing](https://developer.sportradar.com/racing) - Kommerzielle Multi-Serien-Racing-API.
- [Race Monitor API](https://www.race-monitor.com/Home/API) - Generische Race-Timing-API, Amateur- bis Profi-Events.
- [Motor Sport Magazine Database](https://www.motorsportmagazine.com/database/) - Fast jedes Motorsport-Rennen seit 1894.
- [DriverDB](https://www.driverdb.com/) - Karrierestatistiken für 70.000+ Fahrer, Elo-artiges Ranking.
- [FIA Results and Statistics](https://fiaresultsandstatistics.motorsportstats.com/) - Offizielles FIA-Ergebnisportal (powered by Motorsport Stats).

### Seed-Metadaten

- [List of Formula One Circuits](https://en.wikipedia.org/wiki/List_of_Formula_One_circuits) - Ausgangspunkt für Streckenmetadaten.
- [List of Motor Racing Circuits by FIA Grade](https://en.wikipedia.org/wiki/List_of_motor_racing_circuits_by_FIA_grade) - Ausgangspunkt für Streckenklassifizierung.

## Lernen und Weiterbildung

### Bücher

- [Wrangling F1 Data With R](https://leanpub.com/wranglingf1datawithr) - Tony Hirst zu F1-Datenverarbeitung und Charting in R.
- [Wrangling F1 Data With Python](https://f1datajunkie.github.io/wranglingf1datawithpython/index.html) - Begleitbuch mit FastF1-Demos und Python-Workflows.

### Blogs und Tutorials

- [F1 Data Junkie](http://www.f1datajunkie.com/) - Tony Hirsts langjähriger F1-Datenanalyse-Blog.
- [RaceFans Race Data](https://www.racefans.net/category/race-data/) - Unabhängige Analyseartikel mit Renndaten.
- [TracingInsights](https://tracinginsights.com/) - Interaktive Charts für Rundenzeiten/Telemetrie, ab 2018.
- [F1technical.net](https://www.f1technical.net/) - Tiefe technische Artikel zu Aerodynamik, Fahrzeugdesign, Reglement.
- [ScarbsF1](https://scarbsf1.wordpress.com/) - Craig Scarboroughs technischer F1-Blog mit Illustrationen.
- [Towards Formula 1 Analysis](https://medium.com/towards-formula-1-analysis) - FastF1-Tutorials auf Medium.
- [Motorsport Engineer](https://motorsportengineer.net/) - Kurse von aktuellen/ehemaligen F1-Ingenieuren.

### YouTube-Kanäle

- [Driver61](https://www.youtube.com/@Driver61) - Fahrtechnik, Setup-Wissenschaft, F1-Engineering.
- [The Race](https://www.youtube.com/@TheRace) - Größter unabhängiger Motorsport-Kanal.
- [WTF1](https://www.youtube.com/@WTF1official) - Community-Kanal mit News, Erklärungen, Fan-Fokus.
- [Sky Sports F1](https://www.youtube.com/@SkySportsF1) - Interviews, Analysen, Renn-Clips.
- [Kyle Engineers](https://www.youtube.com/@KyleEngineers) - F1-Engineering-Analyse mit Aero-/Fahrdynamik-Tiefgang.
- [Peter Windsor](https://www.youtube.com/@peterwindsor) - F1 aus Sicht eines erfahrenen Journalisten.
- [CYMotorsport](https://www.youtube.com/@CYMotorsport) - Datengetriebene Fahrervergleiche, Historie.
- [Chain Bear](https://www.youtube.com/@chainbear) - Klare Visualisierungen von Regeln, Strategie, Technik.

### Podcasts

- [Beyond the Grid](https://www.youtube.com/@BeyondTheGridF1) - Offizieller F1-Podcast mit Fahrer-/Teamchef-Interviews.
- [Data Driven F1](https://podcasts.apple.com/us/podcast/data-driven-f1/id1527676761) - Technologie, Daten, Verhaltensanalyse in F1.
- [The Race F1 Podcast](https://the-race.com/podcasts/) - Analysen der The-Race-Journalisten.
- [The Race F1 Tech Show](https://podcasts.apple.com/us/podcast/the-race-f1-tech-show/id1502430647) - Gary Anderson und Edd Straw zu Fahrzeugtechnik.

## Filme, Dokumentationen und Spiele

### Dokumentationen

- [Schumacher '94](https://www.imdb.com/title/tt39453404/) (2026, Netflix) - Schumachers erste Meisterschaftssaison 1994.
- [Drive to Survive](https://www.netflix.com/title/80204890) (2019-2026, Netflix) - Behind-the-scenes-Doku-Serie, 8 Staffeln.
- [Brawn: The Impossible Formula 1 Story](https://www.imdb.com/title/tt22297946/) (2023, Disney+) - Brawn GPs WM-Titel 2009.
- [F1: The Academy](https://www.imdb.com/title/tt36711188/) (2025, Netflix) - F1-Academy-Nachwuchsserie.
- [The Seat](https://www.imdb.com/title/tt36741795/) (2025, Netflix) - Antonellis Aufstieg bei Mercedes.
- [Schumacher](https://www.netflix.com/title/81399308) (2021, Netflix) - Leben und Karriere Michael Schumachers.
- [A Life of Speed: The Juan Manuel Fangio Story](https://www.netflix.com/title/80208059) (2020, Netflix) - Juan Manuel Fangio.
- [Williams](https://www.imdb.com/title/tt5765218/) (2017) - Sir Frank Williams.
- [McLaren](https://www.imdb.com/title/tt6209326/) (2017) - Bruce McLaren.
- [1](https://www.imdb.com/title/tt2518788/) (2013) - Die goldene Ära der Formel 1.
- [Senna](https://www.imdb.com/title/tt1424432/) (2010) - Ayrton Senna.

### Filme

- [F1](https://www.imdb.com/title/tt16311594/) (2025) - Mit Brad Pitt, Regie Joseph Kosinski.
- [Ford v Ferrari](https://www.imdb.com/title/tt1950186/) (2019) - Le Mans 1966.
- [Rush](https://www.imdb.com/title/tt1979320/) (2013) - Hunt gegen Lauda 1976, Regie Ron Howard.
- [Grand Prix](https://www.imdb.com/title/tt0060472/) (1966) - Klassiker mit echten Strecken/Fahrern der Ära.

### Spiele

- [EA Sports F1 25](https://www.ea.com/games/f1/f1-25) (2025) - Offizielles F1-Rennspiel mit UDP-Telemetrie-Output.
- [F1 Manager 2024](https://store.steampowered.com/app/2287220/F1_Manager_2024/) (2024) - Offizielle F1-Management-Simulation.

## Offizielle Referenzlinks

**Serien:** [Formula 1](https://www.formula1.com/) · [Formula 2](https://www.fiaformula2.com/) · [Formula 3](https://www.fiaformula3.com/) · [Formula E](https://fiaformulae.com/) · [NASCAR](https://www.nascar.com/) · [IndyCar](https://www.indycar.com/) · [FIA WEC](https://www.fiawec.com/) · [Le Mans](https://www.24h-lemans.com/) · [IMSA](https://www.imsa.com/) · [MotoGP](https://www.motogp.com/) · [WorldSBK](https://www.worldsbk.com/) · [WRC](https://www.wrc.com/) · [DTM](https://www.dtm.com/en)

**Regelwerke:** [FIA Regulations](https://www.fia.com/regulations) · [FIA Formula 1 Regulations](https://www.fia.com/regulation/category/29) · [FIA WEC Regulations](https://www.fiawec.com/en/page/regulations-1) · [FIM Documents](https://www.fim-moto.com/en/documents)

---

Quelle: [subinium/awesome-f1](https://github.com/subinium/awesome-f1). Jede verlinkte Ressource
behält ihre eigene Lizenz/Nutzungsbedingungen.
