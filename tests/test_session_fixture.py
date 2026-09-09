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


@pytest.fixture(scope="module")
def rennen_voll():
    """Rennen mit Wetter und Race-Control-Meldungen.

    Beides liegt als eigene Pickle-Datei im Cache und wird von
    Session.load() nur auf Anforderung geladen - zusammen 17 KB neben den
    147 MB Telemetrie, aber sie schalten neun weitere f1lab-Funktionen
    fuer Offline-Tests frei.
    """
    return _laden_mit("R", weather=True, messages=True)


def _laden_mit(ident: str, **kw):
    f1lab.enable_cache(path=FIXTURE, offline=True)
    return f1lab.load(2024, "Bahrain", ident, **kw)


class TestWetter:
    def test_weather_join_haengt_die_messwerte_an(self, rennen_voll):
        """jede gewertete Runde bekommt den naechstgelegenen Messpunkt.
        Weniger Zeilen als clean_laps(), weil zusaetzlich Boxenrunden
        rausfallen."""
        m = f1lab.weather_join(rennen_voll)
        assert len(m) == 1006
        for spalte in ("AirTemp", "TrackTemp", "Humidity", "Rainfall",
                       "WindSpeed"):
            assert spalte in m.columns
        assert not m["TrackTemp"].isna().any()

    def test_bahrain_2024_war_durchgehend_trocken(self, rennen_voll):
        m = f1lab.weather_join(rennen_voll)
        assert m["Rainfall"].sum() == 0
        phasen = f1lab.weather_phases(rennen_voll)
        assert len(phasen) == 1
        assert phasen.iloc[0]["nass"] is False or not phasen.iloc[0]["nass"]

    def test_temperatureffekt_reproduziert_den_p17_befund(self, rennen_voll):
        """Der Kern von P17, an einem anderen Rennen: die naive gepoolte
        Regression findet praktisch nichts (R² 0.003), erst mit
        Fahrer-Median-Bereinigung und Reifenalter als zweiter Variable
        wird der Effekt sichtbar (R² 0.50 -> 0.61, Koeffizient +0.45 s/°C
        bei einem Standardfehler von 0.03).

        Genau diese Gegenueberstellung ist die Aussage der Funktion - ein
        Test nur auf "gibt ein dict zurueck" wuerde sie nicht schuetzen.
        """
        e = f1lab.temperature_effect(f1lab.weather_join(rennen_voll))
        assert e["n"] == 1002
        assert abs(e["naiv_r2"]) < 0.05                  # naiv: kein Signal
        assert e["r2_voll"] > e["r2_tyre_only"] > 0.4    # kontrolliert schon
        assert e["coef_temp"] == pytest.approx(0.448, abs=0.01)
        assert e["coef_temp"] / e["se_temp"] > 10        # klar von 0 getrennt


class TestRaceControl:
    def test_track_limits_werden_erkannt(self, rennen_voll):
        """20 Meldungen an vier Kurven. Die Regex hat bei ihrer
        Einfuehrung 0 von 6 Meldungen getroffen (siehe P19) - deshalb
        steht hier eine Zahl und nicht nur "nicht leer"."""
        tl = f1lab.parse_track_limits(rennen_voll.race_control_messages)
        assert len(tl) == 20
        assert list(tl.columns) == ["lap", "nr", "driver", "turn"]
        assert sorted(tl["turn"].unique()) == [4, 10, 13, 15]
        assert tl["driver"].nunique() == 10

    def test_keine_strafen_ist_ein_ergebnis_mit_spalten(self, rennen_voll):
        """Bahrain 2024 hatte keine Zeitstrafe. Der leere Rahmen muss
        trotzdem seine Spalten tragen - genau diese Klasse hat P19 fuer
        die ganze Saison 2018 zum Absturz gebracht."""
        pen = f1lab.parse_penalties(rennen_voll.race_control_messages)
        assert pen.empty
        assert list(pen.columns) == ["lap", "strafmass", "nr", "driver",
                                     "grund"]

    def test_blaue_flaggen_treffen_die_ueberrundeten(self, rennen_voll):
        """aus Flag/RacingNumber statt aus dem Freitext. Die Spitze der
        Liste ist kein Vergehen, sondern "wurde am oeftesten
        ueberrundet" - Sargeant im langsamsten Auto des Feldes."""
        bf = f1lab.blue_flags(rennen_voll, rennen_voll.race_control_messages)
        assert len(bf) == 26
        assert list(bf.columns) == ["time", "lap", "driver", "nr"]
        assert bf.groupby("driver").size().idxmax() == "SAR"

    def test_gegenpruefung_text_gegen_geloeschte_runden(self, rennen_voll):
        """20 Deleted-Runden in den Lap-Daten, 18 Meldungen mit
        eindeutiger Rundennummer, keine davon ohne Entsprechung. Die
        Differenz sind Meldungen ohne explizite Runde ("NEXT LAP"), kein
        Fehler - deshalb wird auf 0 fehlende geprueft, nicht auf
        Gleichheit der beiden Zahlen."""
        fehlend, deleted, n_mit_runde = f1lab.track_limit_crosscheck(
            rennen_voll, rennen_voll.race_control_messages)
        assert len(deleted) == 20
        assert n_mit_runde == 18
        assert fehlend.empty
        assert list(fehlend.columns) == ["driver", "runde"]

    def test_umgekehrte_gegenpruefung(self, rennen_voll):
        """DeletedReason -> gibt es eine Textmeldung dazu? Die andere
        Richtung derselben Pruefung."""
        fehlend = f1lab.deleted_reason_crosscheck(
            rennen_voll, rennen_voll.race_control_messages)
        assert fehlend.empty


class TestDrsNutzung:
    def test_drs_anteil_je_fahrer(self, quali_tel):
        """Zeitanteil mit offenem DRS und der Topspeed-Gewinn dadurch.

        Im Qualifying darf jeder DRS frei nutzen (keine
        Ein-Sekunden-Regel wie im Rennen), die Werte liegen deshalb eng
        beieinander - genau das macht sie als Test brauchbar: ein Fehler
        in der Flankenerkennung wuerde einzelne Fahrer ausreissen lassen.
        """
        d = f1lab.drs_usage(quali_tel)
        assert len(d) == 20
        assert list(d.columns) == list(f1lab.session.DRS_USAGE_DTYPEN)
        assert d["drs_pct"].is_monotonic_decreasing      # absteigend sortiert
        assert d["drs_pct"].between(20, 30).all()
        assert (d["vmax_offen"] > d["vmax_zu"]).all()    # offen ist schneller
        assert d["gewinn_kmh"].median() == pytest.approx(12.5, abs=1.0)


class TestDirtyAir:
    """Der Kern von P32, an zwei Fahrern desselben Rennens.

    Verstappen fuehrte in Bahrain 2024 von Runde 1 bis ins Ziel - vor ihm
    war niemand, er kann per Definition keine Dirty Air gehabt haben.
    Perez dahinter schon. Beide Faelle durch dieselbe Funktion zu
    schicken, ist schaerfer als jeder Einzeltest: ein Effekt, der auch
    beim Fuehrenden anschlaegt, waere ein Rechenfehler.
    """

    def test_der_fuehrende_hat_keine_dirty_air(self, rennen_tel):
        cf = f1lab.close_following(rennen_tel, "VER")
        assert len(cf) == 50
        assert cf["anteil_nah"].max() == 0.0
        slope, _inter, r2, d = f1lab.dirty_air_effect(cf)
        assert len(d) == 0
        assert slope != slope                            # NaN
        assert r2 != r2
        assert "sec_corr" in d.columns                   # Spalte trotzdem da

    def test_der_verfolger_zeigt_einen_messbaren_effekt(self, rennen_tel):
        cf = f1lab.close_following(rennen_tel, "PER")
        assert len(cf) == 51
        assert list(cf.columns) == list(f1lab.CLOSE_FOLLOW_DTYPEN)
        slope, _inter, r2, d = f1lab.dirty_air_effect(cf)
        assert len(d) == 46
        assert slope == pytest.approx(0.0082, abs=0.002)  # s je Prozent nah
        assert r2 == pytest.approx(0.39, abs=0.05)
        assert slope > 0                     # naeher dran heisst langsamer


class TestMiniSektoren:
    def test_wer_gewinnt_welchen_abschnitt(self, quali_tel):
        """25 Abschnitte, drei Fahrer, je Abschnitt genau ein Sieger.

        Die Zeit wird ueber die Distanz interpoliert, nicht die
        Geschwindigkeit gemittelt - nur Zeit ist additiv. Ein Fehler
        darin wuerde die Siegerverteilung kippen, ohne dass sonst etwas
        auffaellt.
        """
        r = f1lab.mini_sectors(quali_tel, ["VER", "PER", "LEC"], n=25)
        assert set(r) == {"telemetrie", "edges", "gewinner", "dauer"}
        assert len(r["gewinner"]) == 25
        assert r["dauer"].shape == (25, 3)
        assert set(r["gewinner"]) <= {"VER", "PER", "LEC"}
        # der schnellste Fahrer der Session gewinnt die meisten Abschnitte
        import collections
        haeufigkeit = collections.Counter(r["gewinner"])
        assert haeufigkeit["VER"] == 13
        assert sum(haeufigkeit.values()) == 25
        # der Sieger je Abschnitt ist der mit der kleinsten Zeit
        for i, sieger in enumerate(r["gewinner"]):
            assert r["dauer"].iloc[i].idxmin() == sieger


class TestDegradationJeMischung:
    def test_reproduziert_den_p13_befund(self, rennen):
        """P13 hat fuer Bahrain 2024 unabhaengig 0.095 s/Runde auf Hard
        und 0.124 auf Soft ermittelt. Der weiche Reifen baut schneller ab
        - faellt diese Ordnung um, stimmt die Fuel-Korrektur oder der Fit
        nicht mehr."""
        dc = f1lab.degradation_by_compound(rennen)
        assert list(dc.index) == ["HARD", "SOFT"]
        assert dc.loc["HARD", "mean"] == pytest.approx(0.097, abs=0.01)
        assert dc.loc["SOFT", "mean"] == pytest.approx(0.133, abs=0.01)
        assert dc.loc["SOFT", "mean"] > dc.loc["HARD", "mean"]
        assert dc["stints"].sum() == 59      # nur belastbare Fits


class TestVerkehrsSzenario:
    def test_die_kette_laeuft_auf_echten_daten_durch(self, rennen):
        """f1lab.traffic_scenario() bindet fuenf core-Funktionen
        aneinander (RaceConfig, Optimum, Frontier, Rundenzeiten,
        Verkehrskosten). Jede einzeln ist in test_core.py geprueft - hier
        zaehlt, dass sie auf echten Daten zusammenpassen."""
        erg = f1lab.traffic_scenario(rennen, 3)
        assert erg["moegliche_stopps"] == [2, 3, 4]
        assert erg["hero"].n_stops == 2
        assert erg["alt"].n_stops == 3
        assert len(erg["hero_zeiten"]) == 57
        assert len(erg["rivale_zeiten"]) == 57

    def test_mehr_stopps_kosten_mehr_verkehr(self, rennen):
        """Der Befund aus P41: der Dreistopp faehrt mit frischeren Reifen
        schneller an den Rivalen heran und verbringt dadurch laenger in
        Ueberhol-Reichweite - der Verkehrsaufschlag ist gut doppelt so
        hoch wie beim Zweistopp."""
        erg = f1lab.traffic_scenario(rennen, 3)
        assert erg["alt_kosten"] > erg["hero_kosten"]
        assert erg["hero_kosten"] == pytest.approx(6.5, abs=1.0)
        assert erg["alt_kosten"] == pytest.approx(14.6, abs=2.0)

    def test_unmoegliche_stoppzahl_nennt_die_moeglichen(self, rennen):
        with pytest.raises(ValueError, match=r"\[2, 3, 4\]"):
            f1lab.traffic_scenario(rennen, 9)


class TestStreckengeometrie:
    """Die Kurven-Funktionen brauchen die MultiViewer-API, deren Antwort
    im requests-Cache liegt (14 KB, gezielt in die Fixture geholt).

    Ohne sie war diese ganze Gruppe offline nicht pruefbar - rund 85
    Zeilen, in denen die Kurvenliste einer fremden API auf die
    Referenzrunde projiziert wird. Genau die Sorte Code, bei der ein
    Vorzeichen- oder Achsenfehler ein plausibles, aber falsches Ergebnis
    liefert.
    """

    def test_bahrain_hat_fuenfzehn_kurven(self, quali_tel):
        assert len(f1lab.circuit_info(quali_tel).corners) == 15

    def test_kurven_liegen_der_reihe_nach_auf_der_runde(self, quali_tel):
        """Jede Kurve wird auf den naechstgelegenen Punkt der
        Referenzrunde projiziert. Die Distanzen muessen danach in
        Kurvenreihenfolge aufsteigen - tun sie das nicht, hat die
        Projektion eine Kurve auf die falsche Streckenseite gelegt."""
        cl = f1lab.corner_labels(quali_tel)
        assert len(cl) == 15
        assert cl["label"].tolist() == [f"T{i}" for i in range(1, 16)]
        assert cl["Distance"].is_monotonic_increasing
        laenge = f1lab.lap_speed_profile(quali_tel)[0][-1]
        assert cl["Distance"].between(0, laenge).all()

    def test_kurvengeschwindigkeiten_je_fahrer(self, quali_tel):
        """20 Fahrer x 15 Kurven. T1 ist die langsamste Kurve in Bahrain
        (enge Rechts nach der Start-Ziel-Geraden), T12 die schnellste -
        ein vertauschtes Fenster wuerde diese Ordnung kippen."""
        cs = f1lab.corner_speeds(quali_tel)
        assert cs.shape == (20, 15)
        mittel = cs.mean()
        assert mittel.idxmin() == "T1"
        assert mittel.min() == pytest.approx(68, abs=8)
        assert mittel.idxmax() == "T12"
        assert mittel.max() == pytest.approx(263, abs=10)

    def test_marshal_punkte_auf_derselben_referenzrunde(self, quali_tel):
        """Sektorgrenzen und Lichttafeln kommen aus zwei getrennten
        Punktlisten derselben API und werden mit derselben
        Naechster-Nachbar-Projektion auf die Runde gelegt."""
        for tabelle in (f1lab.marshal_sector_labels(quali_tel),
                        f1lab.marshal_light_labels(quali_tel)):
            assert len(tabelle) == 18
            assert list(tabelle.columns) == ["number", "distance"]
            assert tabelle["distance"].is_monotonic_increasing

    def test_geometrie_misst_die_gefahrene_linie(self, quali_tel):
        """Nicht die offizielle Streckenlaenge: die Ideallinie schneidet
        Kurven und faellt kuerzer aus (5338 m gegen 5412 m offiziell).
        Genau diese Abweichung ist im Docstring der Funktion als
        erwartetes Verhalten festgehalten."""
        g = f1lab.circuit_geometry(quali_tel)
        assert g["corners"] == 15
        assert g["length_m"] == pytest.approx(5338, abs=30)
        assert g["length_m"] < 5412
        assert g["elev_span_m"] == pytest.approx(16.7, abs=2.0)
        assert g["elev_gain_m"] > g["elev_span_m"]   # Summe > Spannweite

    def test_circuit_dimension_als_tabelle(self):
        cd = f1lab.circuit_dimension([(2024, "Bahrain")])
        assert len(cd) == 1
        z = cd.iloc[0]
        assert z["circuit"] == "Sakhir"
        assert z["corners"] == 15
        assert z["length_m"] == pytest.approx(5338, abs=30)
