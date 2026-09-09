"""f1lab.session gegen echte Renndaten, offline.

Bis hierher war session.py zu rund 30 % abgedeckt: fast alles darin
braucht eine geladene Session, und die Tests kommen ohne Netz aus. Der
Rechenkern (core.py, 99 %) war geprueft, die Schicht darueber - die aus
FastF1-Rohdaten die Kennzahlen macht - praktisch nicht.

Der noetige Cache liegt laengst im Repository: das Teilpaket f1analyze
bringt Bahrain 2024 (Rennen und Qualifying, mit Telemetrie) als Fixture
mit, damit seine eigene CI offline laeuft. Dieselben Dateien tragen hier
die Tests der Kernbibliothek - ohne ein Byte zusaetzliches Gewicht.

Geprueft werden Aussagen, keine Ausfuehrbarkeit. "laeuft ohne Ausnahme
durch" haette keinen der Fehler gefunden, die in diesem Repository
aufgetreten sind - die Funktionen liefen ja, sie lieferten nur das
Falsche. Deshalb stehen hier konkrete Zahlen aus einem konkreten Rennen.

Aendert FastF1 seinen Parser, koennen sich einzelne Werte verschieben.
Das ist gewollt: dann soll es auffallen und nicht stillschweigend
durchgehen.
"""
from __future__ import annotations

import pathlib

import pandas as pd
import pytest

import f1lab

FIXTURE = (pathlib.Path(__file__).resolve().parents[1] / "10_data_engineering"
           / "f1analyze" / "tests" / "fixtures" / "f1_cache_bahrain2024")

pytestmark = pytest.mark.skipif(
    not FIXTURE.exists(),
    reason="Fixture-Cache von f1analyze nicht vorhanden")


def _laden(ident: str, telemetry: bool = False):
    # offline=True: eine fehlende Session scheitert sofort statt in einer
    # minutenlangen Netzanfrage zu haengen (siehe f1lab.enable_cache).
    f1lab.enable_cache(path=FIXTURE, offline=True)
    return f1lab.load(2024, "Bahrain", ident, telemetry=telemetry)


@pytest.fixture(scope="module")
def rennen():
    return _laden("R")


@pytest.fixture(scope="module")
def rennen_tel():
    return _laden("R", telemetry=True)


@pytest.fixture(scope="module")
def quali_tel():
    return _laden("Q", telemetry=True)


class TestRundenfilter:
    def test_clean_laps_wirft_etwas_weg_aber_nicht_alles(self, rennen):
        """der Filter ist der Ausgangspunkt jeder Pace-Rechnung.

        Beide Randfaelle waeren still falsch: filtert er nichts, stehen
        Boxen- und Safety-Car-Runden im Median; filtert er zu viel, bleibt
        nichts uebrig. In Bahrain 2024 fallen 155 von 1129 Runden raus.
        """
        alle, sauber = len(rennen.laps), len(f1lab.clean_laps(rennen))
        assert alle == 1129
        assert sauber == 974
        assert 0.5 < sauber / alle < 0.95


class TestPaceTabelle:
    @pytest.fixture(scope="class")
    def pace(self, rennen):
        return f1lab.pace_table(rennen)

    def test_alle_fahrer_und_feste_spalten(self, pace):
        assert len(pace) == 20
        assert list(pace.columns) == list(f1lab.session.PACE_SPALTEN)

    def test_nach_delta_sortiert_und_der_schnellste_hat_null(self, pace):
        assert pace["delta_s"].is_monotonic_increasing
        assert pace.iloc[0]["delta_s"] == 0.0
        assert pace.iloc[0]["driver"] == "VER"

    def test_das_konfidenzintervall_umschliesst_den_punktwert(self, pace):
        """ci_lo <= delta_s <= ci_hi. Sonst zeichnet jede Grafik darueber
        einen Fehlerbalken in die falsche Richtung - matplotlib lehnt
        negative xerr rundheraus ab."""
        assert (pace["ci_lo"] <= pace["delta_s"]).all()
        assert (pace["delta_s"] <= pace["ci_hi"]).all()

    def test_die_spitze_liegt_dicht_beieinander(self, pace):
        """die Aussage des Projekts ueber Bahrain 2024: hinter Verstappen
        trennt die naechsten drei weniger als eine Sekunde."""
        assert pace.iloc[1]["driver"] == "SAI"
        assert pace.iloc[1]["delta_s"] == pytest.approx(0.25, abs=0.01)
        assert pace.iloc[3]["delta_s"] < 1.0


class TestStintsUndDegradation:
    def test_stints_treffen_die_gefahrenen_mischungen(self, rennen):
        st = f1lab.stints(rennen)
        assert len(st) == 63
        # Bahrain 2024 wurde auf Soft und Hard gefahren, Medium kam nicht
        # zum Einsatz - ein Ergebnis, kein Filterfehler.
        assert sorted(st["Compound"].unique()) == ["HARD", "SOFT"]

    def test_die_meisten_fits_sind_belastbar(self, rennen):
        deg = f1lab.degradation(rennen)
        assert len(deg) == 62
        assert int(deg["reliable"].sum()) == 59
        assert list(deg.columns) == f1lab.session.DEGRADATION_SPALTEN

    def test_degradation_ist_ueberwiegend_positiv(self, rennen):
        """Reifen werden langsamer, nicht schneller. Einzelne negative
        Steigungen gibt es (kuehlende Strecke, freiwerdender Verkehr), die
        Mehrheit muss aber in die andere Richtung zeigen."""
        deg = f1lab.degradation(rennen)
        rel = deg[deg["reliable"]]
        assert (rel["deg_s_per_lap"] > 0).mean() > 0.7

    def test_pitloss_liegt_im_plausiblen_bereich(self, rennen):
        """volle Boxengassenzeit, nicht die Standzeit - fuer Bahrain rund
        25 Sekunden. Als Spanne, weil der Wert aus Medianen echter Stopps
        kommt und nicht auf die Hundertstel festgenagelt gehoert."""
        assert 20.0 < f1lab.pit_loss(rennen) < 30.0


class TestRennverlauf:
    def test_ueberholungen_und_fuehrungswechsel(self, rennen):
        """180 Ueberholungen, aber kein einziger Fuehrungswechsel:
        Verstappen fuehrte in Bahrain 2024 von Runde 1 bis ins Ziel. Genau
        diese Unterscheidung macht overtake_events() gegen lead_changes()
        aus - eine Verwechslung faellt sonst nirgends auf.
        """
        assert len(f1lab.overtake_events(rennen)) == 180
        assert len(f1lab.lead_changes(rennen)) == 0

    def test_ueberholmatrix_passt_zu_den_einzelereignissen(self, rennen):
        matrix = f1lab.overtakes_matrix(rennen)
        assert matrix.values.sum() == len(f1lab.overtake_events(rennen))
        # niemand ueberholt sich selbst
        assert all(matrix.loc[d, d] == 0 for d in matrix.index)

    def test_teamkollegen_duelle(self, rennen):
        """zehn Teams, also hoechstens zehn Duelle."""
        duelle = f1lab.teammate_duels(rennen)
        assert len(duelle) == 10
        assert all(set(f1lab.DUELL_SPALTEN) <= set(d) for d in duelle)
        assert all(d["a"] != d["b"] for d in duelle)

    def test_phasen_decken_die_session_ab(self, rennen):
        phasen = f1lab.track_status_phases(rennen)
        assert len(phasen) == 7
        assert (phasen["duration_s"] > 0).all()
        # lueckenlos aneinander: jede Phase beginnt, wo die vorige endet
        assert (phasen["start"].iloc[1:].to_numpy()
                == phasen["end"].iloc[:-1].to_numpy()).all()

    def test_sieg_attribution_nennt_den_sieger(self, rennen):
        erg = f1lab.sieg_attribution(rennen)
        assert erg["sieger"] == "VER"
        assert erg["fuehrungsrunden"] == erg["gesamtrunden"]
        assert erg["fuehrungsanteil"] == pytest.approx(1.0)


class TestTelemetrie:
    def test_referenzrunde_ist_die_schnellste_der_session(self, quali_tel):
        """nicht die Pole-Runde: Pole faellt in Q3, die schnellste Runde
        des ganzen Qualifyings kann in Q2 stehen. In Bahrain 2024 war
        genau das der Fall - Leclerc 1:29.165 in Q2, Verstappens Pole
        1:29.179."""
        lap = f1lab.reference_lap(quali_tel)
        assert lap["Driver"] == "LEC"
        assert lap["LapTime"].total_seconds() == pytest.approx(89.165,
                                                               abs=0.001)
        schnellste = quali_tel.laps["LapTime"].min()
        assert lap["LapTime"] == schnellste

    def test_bremszonen_haben_hand_und_fuss(self, quali_tel):
        zonen = f1lab.driver_braking_zones(quali_tel, "VER")
        assert len(zonen) == 7
        assert (zonen["end_m"] > zonen["start_m"]).all()
        assert (zonen["v_entry_kmh"] > zonen["v_min_kmh"]).all()
        assert (zonen["decel_g"] > 0).all()

    def test_drs_zonen(self, quali_tel):
        """Bahrain hat drei DRS-Zonen, die Funktion findet vier - und das
        ist richtig so: die Start/Ziel-Gerade wird von der Rundengrenze
        zerschnitten. Ihr Ende steht als 100-m-Stueck am Rundenende
        (5249-5349 m von 5380 m), ihr Anfang als eigene Zone ab Meter 7.
        Die drei "echten" Zonen sind die langen.
        """
        zonen = f1lab.drs_zones(quali_tel, "VER")
        assert len(zonen) == 4
        assert (zonen["length_m"] > 0).all()
        lang = zonen[zonen["length_m"] > 200]
        assert len(lang) == 3
        rest = zonen[zonen["length_m"] <= 200].iloc[0]
        assert rest["end_m"] > zonen["end_m"].max() - 50

    def test_geschwindigkeitsprofil(self, quali_tel):
        dist, kappa, speed = f1lab.lap_speed_profile(quali_tel)
        assert len(dist) == len(kappa) == len(speed)
        assert dist[0] < dist[-1]                    # monoton wachsend
        # die *gefahrene* Linie, nicht die offizielle Streckenlaenge:
        # die Ideallinie schneidet Kurven und faellt kuerzer aus (5380 m
        # gegen 5412 m offiziell).
        assert 5300 < dist[-1] < 5450
        assert 0 < speed.max() < 120                 # m/s, nicht km/h

    def test_startphase(self, rennen_tel):
        start = f1lab.start_performance(rennen_tel)
        assert not start.empty
        assert list(start.columns) == list(f1lab.START_PERF_DTYPEN)
        assert (start["m_nach_5s"].dropna() > 0).all()


class TestStrategiemodell:
    def test_race_config_kommt_aus_den_echten_daten(self, rennen):
        cfg = f1lab.race_config_from_session(rennen)
        assert cfg.n_laps == 57                      # Renndistanz Bahrain
        assert len(cfg.tyres) == 2                   # Soft und Hard
        assert 20.0 < cfg.pit_loss < 30.0

    def test_der_optimale_plan_ist_fahrbar(self, rennen):
        """die Bausteine aus core.py sind einzeln geprueft; hier zaehlt,
        dass sie auf echten Daten zusammen einen gueltigen Plan ergeben."""
        cfg = f1lab.race_config_from_session(rennen)
        plan = f1lab.optimal_strategy(cfg)
        assert plan.n_stops >= 1
        assert sum(s.end_lap - s.start_lap + 1 for s in plan.stints) == 57
        assert len({s.compound for s in plan.stints}) >= 2  # Pflicht-Regel

    def test_undercut_duelle(self, rennen):
        duelle = f1lab.undercut_duels(rennen)
        assert len(duelle) == 16
        assert duelle["erfolg"].dtype == bool
        assert (duelle["rival_lap"] > duelle["lap"]).all()


class TestQualifying:
    def test_streckenentwicklung_ueber_die_segmente(self, quali_tel):
        """die Strecke wird ueber das Qualifying schneller; die Deltas
        sind paarweise je Fahrer gebildet, nicht als Segment-Mittel."""
        d = f1lab.qualifying_track_evolution(quali_tel)
        assert set(d["segment"].unique()) <= {"Q1->Q2", "Q2->Q3"}
        assert list(d.columns) == ["driver", "segment", "delta_s"]
        assert d["delta_s"].median() > 0             # spaeter = schneller


class TestGegenprobeZuFastf1:
    """einzelne Werte gegen FastF1 selbst, nicht nur gegen sich selbst.

    Ein Test, der f1lab nur mit f1lab vergleicht, haelt auch einen
    gemeinsamen Denkfehler fuer richtig.
    """

    def test_sieger_stimmt_mit_der_ergebnistabelle(self, rennen):
        aus_ergebnis = rennen.results.sort_values("Position").iloc[0]
        assert f1lab.sieg_attribution(rennen)["sieger"] == \
            aus_ergebnis["Abbreviation"]

    def test_fahrerzahl_stimmt_mit_der_session(self, rennen):
        assert len(f1lab.pace_table(rennen)) == len(rennen.drivers)

    def test_clean_laps_ist_eine_teilmenge(self, rennen):
        sauber = f1lab.clean_laps(rennen)
        assert set(sauber.index) <= set(rennen.laps.index)
        assert not sauber["LapTime"].isna().any()
        assert isinstance(sauber, pd.DataFrame)


class TestUeberholorte:
    """overtake_locations() ist die aufwendigste Funktion in session.py:
    sie verknuepft Ueberholereignisse aus den Positionsdaten mit der
    Telemetrie beider Fahrer und schlaegt das Ergebnis gegen die DRS-Zonen
    einer zweiten Session. Vier Datenquellen, eine Aussage.
    """

    @pytest.fixture(scope="class")
    def orte(self, rennen_tel, quali_tel):
        return f1lab.overtake_locations(rennen_tel, drs_session=quali_tel)

    def test_jeder_ort_liegt_auf_der_strecke(self, orte, quali_tel):
        assert len(orte) == 140
        assert list(orte.columns) == ["gainer", "loser", "lap",
                                      "distance_m", "in_drs_zone"]
        laenge = f1lab.lap_speed_profile(quali_tel)[0][-1]
        assert orte["distance_m"].min() >= 0
        assert orte["distance_m"].max() <= laenge

    def test_nicht_jede_ueberholung_laesst_sich_lokalisieren(
            self, orte, rennen_tel):
        """140 von 180: fuer den Rest fehlt die Telemetrie eines der
        beiden Fahrer. Waeren es alle, wuerde die Funktion raten."""
        assert len(orte) < len(f1lab.overtake_events(rennen_tel))

    def test_die_mehrheit_faellt_in_eine_drs_zone(self, orte):
        """Bahrain 2024: 117 von 140 lokalisierten Ueberholungen liegen in
        einer DRS-Zone. Die Zahl ist die eigentliche Aussage von P39 - ein
        Fehler in der Zonenzuordnung wuerde sie kippen, ohne dass sonst
        etwas auffaellt."""
        assert orte["in_drs_zone"].dtype == bool
        assert int(orte["in_drs_zone"].sum()) == 117

    def test_niemand_ueberholt_sich_selbst(self, orte):
        assert (orte["gainer"] != orte["loser"]).all()


class TestStartaufstellung:
    def test_grid_gegen_runde_eins(self, rennen_tel):
        """der Gewinn ist die Differenz, nicht neu gezaehlt - ein
        Vorzeichenfehler faellt sonst niemandem auf."""
        g = f1lab.grid_lap1_positions(rennen_tel)
        assert len(g) == 20
        assert list(g.columns) == ["driver_number", "grid", "lap1", "gewinn"]
        assert (g["gewinn"] == g["grid"] - g["lap1"]).all()
        # was einer gewinnt, verliert ein anderer
        assert g["gewinn"].sum() == pytest.approx(0.0)

    def test_feldstreckung_je_runde(self, rennen):
        """Sekunden zwischen Erstem und Letztem, je Runde.

        Waechst ueber das Rennen, weil das Feld auseinanderfaehrt. Die
        beiden Randwerte stehen als Zahl da, nicht nur als ">0" und
        "waechst": beim Gegenpruefen hat eine Mutation (Betrag plus eins)
        genau diese beiden weichen Zusicherungen ueberlebt.
        """
        spread = f1lab.field_spread(rennen)
        assert len(spread) == 57
        assert (spread > 0).all()
        assert spread.iloc[-1] > spread.iloc[0]
        assert spread.min() == pytest.approx(16.3, abs=0.5)
        assert spread.max() == pytest.approx(210.4, abs=1.0)


class TestLeereAberGueltigeErgebnisse:
    """Faelle, in denen "nichts gefunden" die richtige Antwort ist.

    Solche Rueckgaben sind die haeufigste Fehlerquelle in diesem
    Repository gewesen: leer ist nicht dasselbe wie kaputt, aber ohne
    Spalten wird daraus beim Aufrufer ein KeyError.
    """

    def test_ohne_safety_car_bleibt_die_tabelle_leer_aber_vollstaendig(
            self, rennen):
        """Bahrain 2024 hatte keine Safety-Car-Phase."""
        sc = f1lab.sc_deployment_sectors(rennen)
        assert sc.empty
        assert list(sc.columns) == ["time", "driver", "sector"]

    def test_trockenes_rennen_hat_keinen_klassifikator(self, rennen):
        """wet_dry_classifier braucht beide Klassen. Frueher stand die
        Pruefung im Docstring ("das prueft der aufrufer") und nirgends im
        Code; jetzt sagt die Funktion selbst, was fehlt."""
        with pytest.raises(ValueError, match="trocken"):
            f1lab.wet_dry_classifier(rennen)


class TestKalender:
    def test_event_dimension_deckt_die_saison_ab(self):
        """laeuft ohne Session-Download, nur aus dem Kalender."""
        ed = f1lab.event_dimension([2024])
        assert len(ed) == 24                       # 24 Rennen 2024
        assert ed["season"].eq(2024).all()
        assert ed["round"].tolist() == list(range(1, 25))
        assert ed["is_sprint"].sum() == 6          # sechs Sprint-Wochenenden
