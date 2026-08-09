"""Reine Rechenfunktionen ohne FastF1-Abhaengigkeit.

Alles hier arbeitet auf numpy-Arrays und ist damit ohne Netzzugriff testbar.
Die FastF1-Anbindung liegt in :mod:`f1lab.session`.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Faustwerte aus der Literatur. Groessenordnung belastbar, keine Messwerte.
FUEL_KG_PER_LAP = 1.8
FUEL_S_PER_KG = 0.03


# --------------------------------------------------------------- Statistik
@dataclass(frozen=True)
class Interval:
    """Punktschaetzer mit Konfidenzintervall."""
    value: float
    lo: float
    hi: float

    @property
    def width(self) -> float:
        return self.hi - self.lo

    def overlaps(self, other: Interval) -> bool:
        """True, wenn sich die Intervalle ueberschneiden.

        Ueberlappende Intervalle heissen: der Unterschied ist mit diesen
        Daten nicht belegbar.
        """
        return not (self.hi < other.lo or other.hi < self.lo)

    def __str__(self) -> str:
        return f"{self.value:.3f} [{self.lo:.3f}, {self.hi:.3f}]"


def bootstrap_median(values, n_resamples: int = 1000, alpha: float = 0.05,
                     seed: int | None = 42) -> Interval:
    """Median mit Bootstrap-Konfidenzintervall.

    Rundenzeiten sind rechtsschief - langsame Runden gibt es viele, schnellere
    als das Optimum nicht. Der Median ist daher robuster als der Mittelwert,
    hat aber keine geschlossene Formel fuer das Konfidenzintervall.

    Args:
        values: Beobachtungen, mindestens 2.
        n_resamples: Anzahl der Bootstrap-Ziehungen.
        alpha: Irrtumswahrscheinlichkeit, 0.05 ergibt ein 95%-Intervall.
        seed: Fuer reproduzierbare Ergebnisse.

    Raises:
        ValueError: bei weniger als 2 Werten.
    """
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    if v.size < 2:
        raise ValueError(f"mindestens 2 Werte noetig, {v.size} erhalten")

    rng = np.random.default_rng(seed)
    draws = rng.choice(v, size=(n_resamples, v.size), replace=True)
    medians = np.median(draws, axis=1)
    return Interval(
        value=float(np.median(v)),
        lo=float(np.percentile(medians, 100 * alpha / 2)),
        hi=float(np.percentile(medians, 100 * (1 - alpha / 2))),
    )


def mad_outlier_mask(values, threshold: float = 3.5) -> np.ndarray:
    """Ausreisser ueber die Median Absolute Deviation.

    Robuster als die Standardabweichung, weil der Schaetzer selbst nicht von
    den Ausreissern verschoben wird. True bedeutet Ausreisser.
    """
    v = np.asarray(values, dtype=float)
    med = np.nanmedian(v)
    mad = np.nanmedian(np.abs(v - med))
    if mad == 0:
        return np.zeros_like(v, dtype=bool)
    modified_z = 0.6745 * (v - med) / mad
    return np.abs(modified_z) > threshold


# --------------------------------------------------------------- Rating
def elo_expected(rating_a: float, rating_b: float) -> float:
    """Erwartete Punktzahl von A gegen B nach dem Elo-Modell, zwischen 0 und 1.

    Gleiche Ratings ergeben 0.5. 400 Punkte Vorsprung bedeuten eine erwartete
    Siegquote von rund 91 Prozent.
    """
    return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))


def elo_update(rating_a: float, rating_b: float, score_a: float,
               k: float = 24.0) -> tuple[float, float]:
    """Neue Ratings nach einem einzelnen Duell.

    Args:
        score_a: Ergebnis aus Sicht von A - 1 fuer einen Sieg, 0 fuer eine
            Niederlage, 0.5 fuer ein Unentschieden.
        k: Wie stark ein einzelnes Duell das Rating bewegt. Klein haelt das
            Rating traege (viele Duelle noetig, um es zu verschieben), gross
            macht es sprunghaft.

    Returns:
        (neues Rating A, neues Rating B). Elo ist ein Nullsummenspiel - was A
        gewinnt, verliert B exakt.
    """
    delta = k * (score_a - elo_expected(rating_a, rating_b))
    return rating_a + delta, rating_b - delta


# --------------------------------------------------------------- Treibstoff
def fuel_correct(lap_times, lap_numbers, total_laps: int,
                 kg_per_lap: float = FUEL_KG_PER_LAP,
                 s_per_kg: float = FUEL_S_PER_KG) -> np.ndarray:
    """Rundenzeiten auf konstante Tankfuellung normieren.

    Ein Auto verliert ueber die Renndistanz rund 100 kg Sprit und wird dadurch
    kontinuierlich schneller. Ohne Korrektur sieht jeder Reifen am Rennende
    besser aus als er ist, und die Degradation wird unterschaetzt.

    Die Korrektur addiert auf jede Runde die Zeit, die das Auto mit der zu
    diesem Zeitpunkt noch vorhandenen Restmenge Sprit schneller gewesen waere.
    Bezugspunkt ist das Rennende (leerer Tank).
    """
    t = np.asarray(lap_times, dtype=float)
    n = np.asarray(lap_numbers, dtype=float)
    remaining_laps = total_laps - n
    return t - remaining_laps * kg_per_lap * s_per_kg


# --------------------------------------------------------------- Degradation
@dataclass(frozen=True)
class DegradationFit:
    """Ergebnis einer Degradationsschaetzung fuer einen Stint."""
    slope: float          # Sekunden pro Runde Reifenalter
    intercept: float      # extrapolierte Zeit bei Reifenalter 0
    r2: float             # Bestimmtheitsmass
    n: int                # ausgewertete Runden

    @property
    def is_reliable(self) -> bool:
        """Grobe Plausibilitaetspruefung.

        Unter 6 Runden ist die Steigung Rauschen, unter R^2 = 0.3 beschreibt
        die Gerade das Verhalten nicht.
        """
        return self.n >= 6 and self.r2 >= 0.3


def fit_degradation(tyre_life, lap_times) -> DegradationFit:
    """Lineare Regression Rundenzeit ueber Reifenalter.

    Die Steigung ist die Degradation in Sekunden pro Runde. Erwartet werden
    fuel-korrigierte Zeiten - sonst mischt sich der Gewichtseffekt hinein und
    kompensiert die Degradation teilweise weg.
    """
    x = np.asarray(tyre_life, dtype=float)
    y = np.asarray(lap_times, dtype=float)
    ok = ~(np.isnan(x) | np.isnan(y))
    x, y = x[ok], y[ok]

    if x.size < 3:
        raise ValueError(f"mindestens 3 Runden noetig, {x.size} erhalten")
    if np.ptp(x) == 0:
        raise ValueError("Reifenalter ist konstant, keine Steigung schaetzbar")

    slope, intercept = np.polyfit(x, y, 1)
    pred = slope * x + intercept
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return DegradationFit(float(slope), float(intercept), r2, int(x.size))


def find_cliff(tyre_life, lap_times, min_segment: int = 4
               ) -> tuple[int | None, DegradationFit, DegradationFit | None]:
    """Sucht den Knickpunkt, ab dem der Reifen deutlich schneller abbaut.

    Reifen degradieren nicht linear. Ab einem gewissen Punkt - dem Cliff -
    bricht der Grip ueberproportional ein. Die Funktion probiert jeden
    moeglichen Bruchpunkt durch und waehlt den mit der kleinsten
    Gesamt-Fehlerquadratsumme.

    Returns:
        (Reifenalter am Knick oder None, Fit davor, Fit danach oder None)

    Ein Ergebnis von None heisst: die zweiteilige Anpassung ist nicht besser
    als die einfache Gerade, der Stint war also linear oder zu kurz.
    """
    x = np.asarray(tyre_life, dtype=float)
    y = np.asarray(lap_times, dtype=float)
    order = np.argsort(x)
    x, y = x[order], y[order]

    single = fit_degradation(x, y)
    if x.size < 2 * min_segment:
        return None, single, None

    single_sse = float(np.sum(
        (y - (single.slope * x + single.intercept)) ** 2))

    best = None
    for i in range(min_segment, x.size - min_segment + 1):
        try:
            left = fit_degradation(x[:i], y[:i])
            right = fit_degradation(x[i:], y[i:])
        except ValueError:
            continue
        sse = (float(np.sum((y[:i] - (left.slope * x[:i] + left.intercept)) ** 2))
               + float(np.sum((y[i:] - (right.slope * x[i:] + right.intercept)) ** 2)))
        if best is None or sse < best[0]:
            best = (sse, i, left, right)

    if best is None:
        return None, single, None

    sse, i, left, right = best
    # Zwei Geraden haben zwei Parameter mehr. Nur akzeptieren, wenn der
    # Fehler deutlich faellt und der zweite Abschnitt staerker ansteigt.
    if sse > 0.8 * single_sse or right.slope <= left.slope:
        return None, single, None

    return int(x[i]), left, right


# --------------------------------------------------------------- Strategie
def estimate_pit_loss(in_lap_deltas, out_lap_deltas) -> float:
    """Zeitverlust eines Boxenstopps aus beobachteten Runden schaetzen.

    Erwartet die Differenz von In- bzw. Out-Lap zur normalen Rundenzeit
    desselben Fahrers. Der Median ist robust gegen die Faelle, in denen
    waehrend des Stopps etwas schiefging.
    """
    a = np.asarray(in_lap_deltas, dtype=float)
    b = np.asarray(out_lap_deltas, dtype=float)
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    if a.size == 0 or b.size == 0:
        raise ValueError("In- und Out-Lap-Deltas duerfen nicht leer sein")
    return float(np.median(a) + np.median(b))


def undercut_gain(deg_old: float, deg_new: float, n_laps: int = 3,
                  out_lap_penalty: float = 0.6) -> float:
    """Zeitgewinn, wenn ein Fahrer n Runden frueher an die Box geht.

    Der Verfolger stoppt frueher und faehrt auf frischen Reifen, waehrend der
    Vordermann weiter altert. Der Pitloss selbst faellt fuer beide an und
    kuerzt sich heraus - entscheidend ist nur die Pace-Differenz im Fenster.

    Args:
        deg_old: Degradation des alten Reifens, s pro Runde.
        deg_new: Degradation des frischen Reifens, s pro Runde.
        n_laps: Wie viele Runden frueher gestoppt wird.
        out_lap_penalty: Aufschlag auf die Out-Lap, weil der Reifen noch kalt
            ist. Groessenordnung eine halbe bis eine Sekunde.

    Returns:
        Sekunden. Positiv heisst, der Undercut lohnt sich.

    Nicht modelliert: Verkehr. Wer hinter einem langsameren Auto herauskommt,
    verliert den Vorteil in einer Runde wieder.
    """
    if n_laps < 1:
        raise ValueError("n_laps muss mindestens 1 sein")

    gain = 0.0
    for i in range(n_laps):
        t_new = deg_new * i + (out_lap_penalty if i == 0 else 0.0)
        t_old = deg_old * (i + 1)
        gain += t_old - t_new
    return float(gain)


def optimal_undercut_window(deg_old: float, deg_new: float,
                            max_laps: int = 8,
                            out_lap_penalty: float = 0.6) -> tuple[int, float]:
    """Findet die Rundenzahl mit dem groessten Undercut-Gewinn.

    Returns:
        (Anzahl Runden, Gewinn in Sekunden)
    """
    gains = [(n, undercut_gain(deg_old, deg_new, n, out_lap_penalty))
             for n in range(1, max_laps + 1)]
    return max(gains, key=lambda t: t[1])


# --------------------------------------------------------------- Telemetrie
def braking_zones(brake, distance, speed, time, min_length_m: float = 20.0
                  ) -> list[dict]:
    """Zerlegt den Bremskanal in einzelne Bremszonen.

    Der Brake-Kanal ist binaer. Ueber die Flanken lassen sich zusammenhaengende
    Bremsphasen finden und je Zone Eintrittsgeschwindigkeit, Laenge und
    mittlere Verzoegerung berechnen.

    Args:
        brake: Bremssignal, wird nach bool konvertiert.
        distance: Zurueckgelegte Distanz in Metern.
        speed: Geschwindigkeit in km/h.
        time: Zeit in Sekunden.
        min_length_m: Kuerzere Zonen sind meist Messrauschen.

    Returns:
        Liste von Zonen mit Kennwerten, sortiert nach Distanz.
    """
    b = np.asarray(brake).astype(bool)
    d = np.asarray(distance, dtype=float)
    v = np.asarray(speed, dtype=float)
    t = np.asarray(time, dtype=float)

    if not (b.size == d.size == v.size == t.size):
        raise ValueError("alle Kanaele muessen gleich lang sein")
    if b.size == 0 or not b.any():
        return []

    # edges[i] vergleicht b[i+1] mit b[i]:
    #   +1 -> b[i+1] ist die erste bremsende Probe  -> start = i + 1
    #   -1 -> b[i]   ist die letzte bremsende Probe -> end   = i
    edges = np.diff(b.astype(np.int8))
    starts = np.flatnonzero(edges == 1) + 1
    ends = np.flatnonzero(edges == -1)
    if b[0]:
        starts = np.r_[0, starts]
    if b[-1]:
        ends = np.r_[ends, b.size - 1]

    zones = []
    for s, e in zip(starts, ends):
        length = d[e] - d[s]
        if length < min_length_m:
            continue
        dt = max(t[e] - t[s], 1e-3)
        dv_ms = (v[s] - v[e]) / 3.6
        zones.append({
            "start_m": round(float(d[s]), 1),
            "end_m": round(float(d[e]), 1),
            "length_m": round(float(length), 1),
            "v_entry_kmh": round(float(v[s]), 1),
            "v_min_kmh": round(float(v[e]), 1),
            "duration_s": round(float(dt), 3),
            "decel_g": round(float(dv_ms / dt / 9.81), 2),
        })
    return zones


def match_by_distance(a, b, tolerance: float) -> list[tuple[int, int]]:
    """Paart Indizes zweier Positionslisten ueber die naechstgelegene Distanz.

    Fuer Ereignisse, die beide Seiten auf derselben Strecke haben - Brems-
    oder Mini-Sektor-Grenzen zweier Fahrer zum Beispiel -, aber nicht exakt
    an derselben Stelle. Je Wert aus ``a`` wird der naechste, noch nicht
    vergebene Wert aus ``b`` gesucht; bleibt keiner innerhalb ``tolerance``,
    bleibt der Wert unverpaart. Das ist kein Fehlerfall - ungleich viele
    Ereignisse (z.B. eine zusaetzliche Bremsung) sind der Normalfall.

    Args:
        a, b: Positionen (z.B. Meter), beliebige Reihenfolge.
        tolerance: Maximaler Abstand fuer eine gueltige Paarung.

    Returns:
        Liste von (Index in a, Index in b), sortiert wie a.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    belegt: set[int] = set()
    paare = []
    for i in range(a.size):
        if b.size == 0:
            break
        diffs = np.abs(b - a[i])
        j = int(np.argmin(diffs))
        if j in belegt or diffs[j] > tolerance:
            continue
        belegt.add(j)
        paare.append((i, j))
    return paare


def active_distance_zones(active, distance, min_length_m: float = 20.0
                          ) -> list[dict]:
    """Zerlegt ein beliebiges binaeres Signal in zusammenhaengende
    Distanz-Zonen - dieselbe Flankenlogik wie :func:`braking_zones`, aber
    ohne die bremsspezifischen Kennwerte (Geschwindigkeit/Verzoegerung).
    Fuer DRS-Aktivzonen gedacht, funktioniert fuer jedes binaere
    Distanzsignal (z.B. auch Throttle > 0).

    Args:
        active: Binaeres Signal, wird nach bool konvertiert.
        distance: Zurueckgelegte Distanz in Metern.
        min_length_m: Kuerzere Zonen sind meist Messrauschen.

    Returns:
        Liste von Zonen (start_m, end_m, length_m), sortiert nach Distanz.
    """
    a = np.asarray(active).astype(bool)
    d = np.asarray(distance, dtype=float)
    if a.size != d.size:
        raise ValueError("alle Kanaele muessen gleich lang sein")
    if a.size == 0 or not a.any():
        return []

    edges = np.diff(a.astype(np.int8))
    starts = np.flatnonzero(edges == 1) + 1
    ends = np.flatnonzero(edges == -1)
    if a[0]:
        starts = np.r_[0, starts]
    if a[-1]:
        ends = np.r_[ends, a.size - 1]

    zones = []
    for s, e in zip(starts, ends):
        length = d[e] - d[s]
        if length < min_length_m:
            continue
        zones.append({"start_m": round(float(d[s]), 1),
                      "end_m": round(float(d[e]), 1),
                      "length_m": round(float(length), 1)})
    return zones


def drs_state(drs_values, open_codes: tuple[int, ...] = (10, 12, 14),
              detected_code: int = 8):
    """Klassifiziert den codierten DRS-Kanal in drei Zustaende.

    0 = zu, 1 = erkannt/im Aktivierungsbereich (Code 8), 2 = offen. FastF1s
    eigene Dokumentation bezeichnet die Codes unterhalb von 10 als unsicher
    ("Unknown Distinction", "Noted Sometimes") - das hier ist die in der
    Community uebliche Lesart, nicht eine offiziell bestaetigte.

    Returns:
        Ganzzahl-Array derselben Laenge wie ``drs_values``.
    """
    v = np.asarray(drs_values)
    status = np.zeros(v.shape, dtype=int)
    status[v == detected_code] = 1
    status[np.isin(v, open_codes)] = 2
    return status


# --------------------------------------------------------------- Streckengeometrie
def path_length(x, y, closed: bool = True) -> float:
    """Laenge eines Streckenzugs als Summe der Segmentlaengen.

    Args:
        x, y: Koordinaten in derselben Einheit; das Ergebnis traegt sie ebenfalls.
        closed: Schliesst den Weg vom letzten zurueck zum ersten Punkt. Eine
            Rennrunde endet dort, wo sie beginnt - ohne das Schlusssegment
            fehlt genau die Luecke zwischen letzter Probe und Start-Ziel.

    Returns:
        Gesamtlaenge. Ein Weg aus weniger als zwei Punkten hat Laenge 0.
    """
    px = np.asarray(x, dtype=float)
    py = np.asarray(y, dtype=float)
    if px.size != py.size:
        raise ValueError("x und y muessen gleich lang sein")

    ok = np.isfinite(px) & np.isfinite(py)
    px, py = px[ok], py[ok]
    if px.size < 2:
        return 0.0

    if closed:
        px = np.r_[px, px[0]]
        py = np.r_[py, py[0]]
    return float(np.hypot(np.diff(px), np.diff(py)).sum())


@dataclass(frozen=True)
class Elevation:
    """Hoehenprofil einer Runde, alle Werte in Metern."""
    gain: float                 # summierter Anstieg
    drop: float                 # summierter Abstieg
    span: float                 # hoechster minus tiefster Punkt

    @property
    def is_flat(self) -> bool:
        """Unter 10 m Spannweite ist eine Strecke praktisch eben."""
        return self.span < 10.0


def elevation_profile(z, min_step: float = 1.0) -> Elevation:
    """Hoehenmeter aus dem Z-Kanal der Positionsdaten.

    Die Rohwerte rauschen um mehrere Dezimeter. Wer einfach alle Betraege der
    Differenzen aufsummiert, zaehlt dieses Rauschen tausendfach mit und landet
    bei Hoehenmetern, die um eine Groessenordnung zu hoch sind. Deshalb hier
    dieselbe Hysterese, die auch GPS-Tracker verwenden: ein Anstieg zaehlt
    erst, wenn er seit der letzten Richtungsumkehr min_step ueberschreitet.

    Die Daempfung wirkt gegen *korreliertes* Rauschen, wie Positionsdaten es
    zeigen - dort faellt sie auf null. Gegen unabhaengiges weisses Rauschen
    hilft sie nur teilweise: dessen Einzelsprunge ueberschreiten die Schwelle
    weiterhin gelegentlich. Auf ungeglaetteten Daten also min_step anheben.

    Args:
        z: Hoehenwerte in Metern.
        min_step: Schwelle in Metern, unterhalb derer eine Aenderung als
            Rauschen gilt.

    Returns:
        Elevation mit Anstieg, Abstieg und Spannweite.
    """
    v = np.asarray(z, dtype=float)
    v = v[np.isfinite(v)]
    if v.size < 2:
        return Elevation(0.0, 0.0, 0.0)

    gain = drop = 0.0
    direction = 0          # +1 steigend, -1 fallend, 0 noch unentschieden
    pivot = v[0]           # letzter bestaetigter Wendepunkt
    peak = v[0]            # laufendes Extremum seit dem Wendepunkt

    for cur in v[1:]:
        if direction > 0:
            if cur > peak:
                peak = cur                      # Anstieg laeuft weiter
            elif cur <= peak - min_step:        # Umkehr bestaetigt
                gain += peak - pivot            # -> Anstieg verbuchen
                pivot, peak, direction = peak, cur, -1
        elif direction < 0:
            if cur < peak:
                peak = cur
            elif cur >= peak + min_step:
                drop += pivot - peak
                pivot, peak, direction = peak, cur, 1
        elif cur >= pivot + min_step:
            direction, peak = 1, cur
        elif cur <= pivot - min_step:
            direction, peak = -1, cur

    # Letzter Abschnitt endet ohne Umkehr und muss noch verbucht werden.
    if direction > 0:
        gain += peak - pivot
    elif direction < 0:
        drop += pivot - peak

    return Elevation(round(float(gain), 1), round(float(drop), 1),
                     round(float(v.max() - v.min()), 1))


# --------------------------------------------------------------- Race Control
def status_intervals(status, time_s
                     ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sparses Status-Log (nur Aenderungen, wie FastF1s ``track_status``) in
    Intervalle umwandeln.

    ``track_status`` traegt eine Zeile pro Zustandswechsel, nicht eine pro
    Zeitschritt - "SCDeployed" erscheint typischerweise genau einmal, auch
    wenn das Safety Car neun Runden lang draussen bleibt. Ein Intervall endet
    deshalb dort, wo das naechste beginnt, nicht am letzten Auftreten
    desselben Werts (das waere derselbe Zeitpunkt wie der Start - ein
    Gruppieren nach gleichem Status ergibt hier fast ausschliesslich
    Intervalle der Laenge 0).

    Args:
        status: Zustandswerte, ein Eintrag pro Aenderung.
        time_s: zugehoerige Zeitpunkte in Sekunden, aufsteigend sortiert.

    Returns:
        (status, start, ende) als drei gleich lange Arrays. ``ende`` des
        letzten Intervalls ist NaN - offen, weil das Log dort endet.
        Unmittelbar wiederholte Werte (zwei Zeilen mit demselben Status ohne
        Wechsel dazwischen) werden zu einem Intervall zusammengefasst.
    """
    s = np.asarray(status)
    t = np.asarray(time_s, dtype=float)
    if s.size == 0:
        return s, t, t.copy()

    behalten = np.empty(s.shape, dtype=bool)
    behalten[0] = True
    behalten[1:] = s[1:] != s[:-1]

    s_out = s[behalten]
    t_out = t[behalten]
    ende = np.append(t_out[1:], np.nan)
    return s_out, t_out, ende
