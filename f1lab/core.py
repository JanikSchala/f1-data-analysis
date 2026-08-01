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
