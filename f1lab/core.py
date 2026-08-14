"""Reine Rechenfunktionen ohne FastF1-Abhaengigkeit.

Alles hier arbeitet auf numpy-Arrays und ist damit ohne Netzzugriff testbar.
Die FastF1-Anbindung liegt in :mod:`f1lab.session`.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, replace

import numpy as np
from scipy.optimize import least_squares
from scipy.signal import savgol_filter

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


# --------------------------------------------------------- Rennstrategie (P35)
# Exakter Boxenstopp-Plan als kuerzester Pfad in einem DAG, plus eine
# Safety-Car-Politik per Rueckwaertsinduktion. Siehe P35 fuer die Herleitung
# (die Stintkosten haengen nur von Mischung und Laenge ab, nicht von der
# Position im Rennen - das macht das Problem zu einem kuerzesten Pfad).
#
# GRUEN/SC sind die zwei Flaggenzustaende, ueber die SafetyCarProcess/
# solve_policy/roll_out sich verstaendigen (0/1 statt Strings - beide Module
# vergleichen sie oft in engen Schleifen).
GRUEN, SC = 0, 1
UNENDLICH = float("inf")


@dataclass(frozen=True)
class TyreModel:
    """Rundenzeitmodell einer Mischung, treibstoffkorrigiert.

    lap_time(alter) = basis + linear * (alter - 1) + quadratisch * (alter - 1)^2

    Das Reifenalter ist einsbasiert: Alter 1 ist die erste Runde auf dem Satz,
    also ist ``base_time`` die Zeit auf frischem Gummi. Die nullbasierte
    Variante waere eine hypothetische Runde auf einem null Runden alten Reifen -
    nicht interpretierbar, und jeder Indexfehler bliebe still.

    Der quadratische Term bildet den Abbau am Stintende ab. Er macht das Modell
    nicht komplizierter, weil die Stintkosten vorab je (Mischung, Laenge)
    ausgerechnet werden: die Rundenzeitfunktion darf beliebig krumm sein,
    solange die Optimierung nur fertige Zahlen sieht.
    """

    compound: str
    base_time: float
    deg_linear: float
    deg_quad: float = 0.0
    max_age: int | None = None

    def lap_time(self, age: int) -> float:
        if age < 1:
            raise ValueError("Reifenalter ist einsbasiert")
        a = age - 1
        return self.base_time + self.deg_linear * a + self.deg_quad * a * a

    def stint_time(self, length: int) -> float:
        """Reine Fahrzeit eines Stints ueber ``length`` Runden, ohne Pitloss."""
        if length < 1:
            raise ValueError("Stintlaenge muss mindestens 1 sein")
        return sum(self.lap_time(a) for a in range(1, length + 1))


@dataclass(frozen=True)
class RaceConfig:
    """Alles, was die Aufgabe festlegt.

    ``fuel_effect`` steht bewusst nicht in der Zielfunktion. Jede Strategie
    faehrt die Runden 1..L genau einmal, also ist der Treibstoffterm fuer alle
    Plaene identisch - eine additive Konstante, die das Optimum nicht bewegen
    kann. Sie wird nur fuer die Anzeige wieder addiert, damit die Rennzeit auf
    einer erkennbaren Skala steht.

    Wichtig: die Basiszeiten muessen treibstoffkorrigiert sein. Der Term
    ``LapNumber`` gehoert in die Degradationsschaetzung (P13), aber nicht
    zusaetzlich in ``base_time`` - sonst zaehlt er doppelt.
    """

    n_laps: int
    pit_loss: float
    tyres: tuple[TyreModel, ...]
    min_stint: int = 1
    max_stint: int | None = None
    require_two_compounds: bool = True
    sc_laps: frozenset[int] = frozenset()
    sc_pit_loss_factor: float = 0.45
    start_compound: str | None = None
    exact_stops: int | None = None
    fuel_effect: float = 0.0

    def __post_init__(self) -> None:
        if self.n_laps < 1:
            raise ValueError("n_laps muss mindestens 1 sein")
        namen = [t.compound for t in self.tyres]
        if not namen:
            raise ValueError("mindestens eine Mischung noetig")
        if len(set(namen)) != len(namen):
            raise ValueError(f"doppelte Mischungen: {namen}")
        if self.require_two_compounds and len(namen) < 2:
            raise ValueError("Zweimischungs-Regel braucht zwei Mischungen")
        if self.start_compound is not None and self.start_compound not in namen:
            raise ValueError(f"unbekannte Startmischung {self.start_compound!r}")
        if self.min_stint < 1:
            raise ValueError("min_stint muss mindestens 1 sein")

    @property
    def fuel_offset(self) -> float:
        """Treibstoffzeit ueber das ganze Rennen. Fuer jede Strategie gleich."""
        return self.fuel_effect * self.n_laps * (self.n_laps - 1) / 2.0


@dataclass(frozen=True)
class Stint:
    compound: str
    start_lap: int
    end_lap: int

    @property
    def length(self) -> int:
        return self.end_lap - self.start_lap + 1

    def __str__(self) -> str:
        return f"{self.compound}[{self.start_lap}-{self.end_lap}]"


@dataclass(frozen=True)
class Strategy:
    stints: tuple[Stint, ...]
    green_time: float
    fuel_offset: float = 0.0

    @property
    def total_time(self) -> float:
        return self.green_time + self.fuel_offset

    @property
    def pit_laps(self) -> tuple[int, ...]:
        """Runden, an deren Ende an die Box gefahren wird."""
        return tuple(s.start_lap - 1 for s in self.stints[1:])

    @property
    def n_stops(self) -> int:
        return len(self.stints) - 1

    @property
    def compounds(self) -> tuple[str, ...]:
        return tuple(s.compound for s in self.stints)

    def describe(self) -> str:
        folge = " -> ".join(str(s) for s in self.stints)
        stopps = ", ".join(str(p) for p in self.pit_laps) or "keine"
        return (f"{self.n_stops}-Stopp  {folge}\n"
                f"  Box am Ende von Runde: {stopps}\n"
                f"  Fahrzeit {self.green_time:9.3f} s | gesamt {self.total_time:9.3f} s")


class InfeasibleRace(RuntimeError):
    """Keine Strategie erfuellt die gesetzten Bedingungen."""


def pit_loss_at(cfg: RaceConfig, lap: int) -> float:
    """Zeitverlust eines Stopps am Ende von ``lap``.

    Unter Safety Car ist das Feld langsam, der relative Preis der Boxengasse
    bricht ein. ``sc_pit_loss_factor`` ist der Anteil, der uebrig bleibt.
    """
    if lap in cfg.sc_laps:
        return cfg.pit_loss * cfg.sc_pit_loss_factor
    return cfg.pit_loss


def stint_arcs(cfg: RaceConfig) -> list[tuple[int, int, int, float]]:
    """Alle legalen Stints als (Mischungsindex, Startrunde, Endrunde, Kosten)."""
    arcs = []
    for ci, tyre in enumerate(cfg.tyres):
        cap = cfg.n_laps if cfg.max_stint is None else min(cfg.max_stint, cfg.n_laps)
        if tyre.max_age is not None:
            cap = min(cap, tyre.max_age)
        for start in range(1, cfg.n_laps + 1):
            if start == 1 and cfg.start_compound not in (None, tyre.compound):
                continue
            for length in range(cfg.min_stint, min(cap, cfg.n_laps - start + 1) + 1):
                end = start + length - 1
                # ein Stint, der vor der Flagge endet, braucht einen Nachfolger,
                # der selbst wieder lang genug ist
                if end < cfg.n_laps and (cfg.n_laps - end) < cfg.min_stint:
                    continue
                eintritt = 0.0 if start == 1 else pit_loss_at(cfg, start - 1)
                arcs.append((ci, start, end, tyre.stint_time(length) + eintritt))
    return arcs


def optimal_strategy(cfg: RaceConfig) -> Strategy:
    """Exaktes Optimum per dynamischer Programmierung ueber den Stint-DAG.

    Der Zustand traegt neben dem Knoten eine Bitmaske der bisher benutzten
    Mischungen. Ohne die waere die Zweimischungs-Regel nicht durchsetzbar: ein
    reiner kuerzester Pfad hat kein Gedaechtnis, und am Ende zu filtern verliert
    die Exaktheit. Bei drei Mischungen kostet die Maske einen Faktor 8.

    Die Stoppzahl steht nur dann im Zustand, wenn sie eingeschraenkt ist. Sie
    immer mitzufuehren vervielfacht den Zustandsraum um die Rundenzahl - fuer
    eine Bedingung, die meistens gar nicht gesetzt ist.
    """
    arcs = stint_arcs(cfg)
    if not arcs:
        raise InfeasibleRace("keine legalen Stints unter diesen Bedingungen")
    aus: dict[int, list] = {}
    for arc in arcs:
        aus.setdefault(arc[1], []).append(arc)

    zaehle = cfg.exact_stops is not None
    L = cfg.n_laps
    # zustand[(maske, stopps)] = (kosten, vorgaenger)
    ebenen: list[dict] = [{} for _ in range(L + 2)]
    ebenen[1][(0, 0)] = (0.0, None)

    for knoten in range(1, L + 1):
        for (maske, stopps), (kosten, _) in ebenen[knoten].items():
            for arc in aus.get(knoten, ()):
                ci, _start, end, preis = arc
                n_stopps = (stopps + (0 if knoten == 1 else 1)) if zaehle else 0
                if cfg.exact_stops is not None and n_stopps > cfg.exact_stops:
                    continue
                schluessel = (maske | (1 << ci), n_stopps)
                ziel = ebenen[end + 1]
                neu = kosten + preis
                alt = ziel.get(schluessel)
                if alt is None or neu < alt[0]:
                    ziel[schluessel] = (neu, (knoten, (maske, stopps), arc))
    besser = None
    for (maske, stopps), eintrag in ebenen[L + 1].items():
        if cfg.require_two_compounds and bin(maske).count("1") < 2:
            continue
        if cfg.exact_stops is not None and stopps != cfg.exact_stops:
            continue
        if besser is None or eintrag[0] < besser[0]:
            besser = eintrag
    if besser is None:
        raise InfeasibleRace(
            "keine Strategie erfuellt die Bedingungen "
            "(min_stint / max_stint / max_age / Zweimischungs-Regel pruefen)")

    gewaehlt = []
    knoten, schluessel, arc = besser[1]
    while True:
        gewaehlt.append(arc)
        vor = ebenen[knoten][schluessel][1]
        if vor is None:
            break
        knoten, schluessel, arc = vor
    stints = tuple(Stint(cfg.tyres[a[0]].compound, a[1], a[2])
                   for a in sorted(gewaehlt, key=lambda a: a[1]))
    return Strategy(stints, sum(a[3] for a in gewaehlt), cfg.fuel_offset)


def frontier_by_stops(cfg: RaceConfig, up_to: int = 4) -> dict[int, Strategy | None]:
    """Bester Plan je exakter Stoppzahl.

    So ist die Frage im Rennen gestellt. "Die fuenf besten Strategien" liefert
    fuenfmal dieselbe Idee mit der Boxenrunde um eins verschoben; "bester
    Einstopper gegen besten Zweistopper und der Abstand dazwischen" ist die
    Entscheidung. Der Abstand misst auch, wie viel Risiko man kauft: drei
    Sekunden sind ein Muenzwurf, fuenfundzwanzig nicht.
    """
    ergebnis: dict[int, Strategy | None] = {}
    for n in range(up_to + 1):
        try:
            ergebnis[n] = optimal_strategy(replace(cfg, exact_stops=n))
        except InfeasibleRace:
            ergebnis[n] = None
    return ergebnis


def pit_loss_crossovers(cfg: RaceConfig, lo: float, hi: float,
                        tol: float = 0.05) -> list[float]:
    """Pitloss-Werte, an denen die optimale Stoppzahl kippt.

    Der Pitloss ist die unsicherste Eingabe - er haengt an Strecke, Verkehr und
    daran, ob der Stopp sauber laeuft. Das Optimum ist stueckweise konstant in
    ihm, interessant ist also nicht der Wert, sondern wo er springt.
    """
    def stopps(v: float) -> int:
        return optimal_strategy(replace(cfg, pit_loss=float(v))).n_stops

    grenzen: list[float] = []
    gitter = [lo + (hi - lo) * i / 20 for i in range(21)]
    v_alt, s_alt = gitter[0], stopps(gitter[0])
    for v in gitter[1:]:
        s = stopps(v)
        if s != s_alt:
            a, b = v_alt, v
            while b - a > tol:
                m = (a + b) / 2
                if stopps(m) == s_alt:
                    a = m
                else:
                    b = m
            grenzen.append(round((a + b) / 2, 2))
        v_alt, s_alt = v, s
    return grenzen


@dataclass(frozen=True)
class SafetyCarProcess:
    """Zweizustands-Markow-Kette auf dem Entscheidungspunkt jeder Runde.

    Grob, aber mit genau zwei Parametern, die sich beide aus FastF1
    ``TrackStatus`` auszaehlen lassen (siehe P18) - und sie erzeugt die zwei
    Eigenschaften, auf die es ankommt: Safety Cars sind selten, und wenn sie
    kommen, bleiben sie ein paar Runden.

    ``sc[l]`` gilt fuer die Entscheidung zu Beginn von Runde l. Der kuerzeste
    Pfad indiziert Safety-Car-Runden dagegen nach der Runde, an deren *Ende*
    gestoppt wird - die beiden passen ueber ``l - 1`` zusammen. Ein Fehler an
    dieser Stelle verschiebt jeden Vergleich um eine Runde und sieht trotzdem
    plausibel aus.
    """

    p_deploy: float = 0.025
    p_end: float = 0.25

    def next_probs(self, state: int) -> tuple[float, float]:
        if state == GRUEN:
            return 1.0 - self.p_deploy, self.p_deploy
        return self.p_end, 1.0 - self.p_end

    def marginals(self, n_laps: int) -> list[float]:
        """q[l] = P(Safety Car am Entscheidungspunkt von Runde l), 1-basiert."""
        q = [0.0] * (n_laps + 2)
        p_gruen = 1.0
        for lap in range(1, n_laps + 1):
            q[lap] = 1.0 - p_gruen
            p_gruen = (p_gruen * self.next_probs(GRUEN)[0]
                       + (1.0 - p_gruen) * self.next_probs(SC)[0])
        return q

    def sample(self, rng: random.Random, n_laps: int) -> list[int]:
        folge = [GRUEN] * (n_laps + 1)
        zustand = GRUEN
        for lap in range(1, n_laps + 1):
            folge[lap] = zustand
            zustand = SC if rng.random() < self.next_probs(zustand)[1] else GRUEN
        return folge


def solve_policy(cfg: RaceConfig, prozess: SafetyCarProcess) -> tuple[float, dict]:
    """Optimale Politik per Rueckwaertsinduktion. Gibt (Erwartungswert, Politik).

    Zustand ist (Runde, Mischung, Reifenalter, benutzte Mischungen, Flagge),
    Aktion ist ausbleiben oder auf Mischung c' wechseln.

    Warum nicht einfach jede Runde neu planen: der kuerzeste Pfad kann
    "vielleicht kommt gleich ein Safety Car" gar nicht ausdruecken. Neuplanen
    mit Punktschaetzung ignoriert deshalb den Optionswert des Wartens - die
    Politik hier dehnt einen Stint manchmal ueber das deterministische Optimum
    hinaus, nur um die Option offenzuhalten.
    """
    if cfg.exact_stops is not None:
        raise NotImplementedError(
            "Stoppzahl-Bedingungen braeuchten einen Zaehler im Zustand")
    n_c = len(cfg.tyres)
    caps = []
    for t in cfg.tyres:
        cap = cfg.n_laps if cfg.max_stint is None else min(cfg.max_stint, cfg.n_laps)
        caps.append(cap if t.max_age is None else min(cap, t.max_age))
    zeiten = [[0.0] + [t.lap_time(a) for a in range(1, caps[i] + 1)]
              for i, t in enumerate(cfg.tyres)]
    stopp = {GRUEN: cfg.pit_loss, SC: cfg.pit_loss * cfg.sc_pit_loss_factor}
    uebergang = {s: prozess.next_probs(s) for s in (GRUEN, SC)}

    def masken(ci: int) -> list[int]:
        bit = 1 << ci
        return [m | bit for m in range(1 << n_c) if not m & bit]

    # Endebene: das Rennen ist vorbei. Zulaessig nur, wenn der letzte Stint lang
    # genug war und die Zweimischungs-Regel erfuellt ist.
    naechste = {}
    for ci in range(n_c):
        for maske in masken(ci):
            ok_maske = (not cfg.require_two_compounds
                        or bin(maske).count("1") >= 2)
            for alter in range(1, caps[ci] + 1):
                gut = ok_maske and alter >= cfg.min_stint
                for s in (GRUEN, SC):
                    naechste[(ci, alter, maske, s)] = 0.0 if gut else UNENDLICH

    politik: dict = {}
    for lap in range(cfg.n_laps, 0, -1):
        aktuell = {}
        for ci in range(n_c):
            alter_liste = (range(1, min(caps[ci], lap - 1) + 1) if lap > 1
                           else range(1))
            for maske in masken(ci):
                for alter in alter_liste:
                    for s in (GRUEN, SC):
                        pg, ps = uebergang[s]
                        best, aktion = UNENDLICH, None
                        neu = alter + 1
                        if neu <= caps[ci]:
                            g = naechste.get((ci, neu, maske, GRUEN), UNENDLICH)
                            h = naechste.get((ci, neu, maske, SC), UNENDLICH)
                            if g < UNENDLICH or h < UNENDLICH:
                                best = zeiten[ci][neu] + pg * g + ps * h
                        if alter >= max(cfg.min_stint, 1):
                            for cj in range(n_c):
                                m2 = maske | (1 << cj)
                                g = naechste.get((cj, 1, m2, GRUEN), UNENDLICH)
                                h = naechste.get((cj, 1, m2, SC), UNENDLICH)
                                if g == UNENDLICH and h == UNENDLICH:
                                    continue
                                wert = stopp[s] + zeiten[cj][1] + pg * g + ps * h
                                if wert < best:
                                    best, aktion = wert, cj
                        aktuell[(ci, alter, maske, s)] = best
                        politik[(lap, ci, alter, maske, s)] = aktion
        naechste = aktuell

    erlaubt = range(n_c)
    if cfg.start_compound is not None:
        erlaubt = [i for i, t in enumerate(cfg.tyres)
                   if t.compound == cfg.start_compound]
    wert, ci_stern = min((naechste.get((ci, 0, 1 << ci, GRUEN), UNENDLICH), ci)
                         for ci in erlaubt)
    if wert == UNENDLICH:
        raise InfeasibleRace("keine Politik erfuellt die Bedingungen")
    politik[("start",)] = ci_stern
    return wert, politik


def roll_out(cfg: RaceConfig, politik: dict, folge: list[int]) -> Strategy:
    """Politik gegen einen konkreten Rennverlauf ausfahren."""
    namen = [t.compound for t in cfg.tyres]
    ci = politik[("start",)]
    maske, alter, start, gesamt = 1 << ci, 0, 1, 0.0
    stints: list[Stint] = []
    for lap in range(1, cfg.n_laps + 1):
        s = folge[lap]
        aktion = politik.get((lap, ci, alter, maske, s))
        if aktion is not None:
            gesamt += cfg.pit_loss * (cfg.sc_pit_loss_factor if s == SC else 1.0)
            stints.append(Stint(namen[ci], start, lap - 1))
            ci, alter, start = aktion, 0, lap
            maske |= 1 << ci
        alter += 1
        gesamt += cfg.tyres[ci].lap_time(alter)
    stints.append(Stint(namen[ci], start, cfg.n_laps))
    return Strategy(tuple(stints), gesamt, cfg.fuel_offset)


def expected_cost_of_plan(cfg: RaceConfig, plan: Strategy,
                          prozess: SafetyCarProcess) -> float:
    """Erwartete Kosten, wenn ein fester Plan stur durchgezogen wird.

    Geschlossene Form, keine Simulation: der einzige zufaellige Anteil ist der
    Pitloss je (fester) Boxenrunde.
    """
    q = prozess.marginals(cfg.n_laps)
    gesamt = sum(next(t for t in cfg.tyres if t.compound == st.compound)
                 .stint_time(st.length) for st in plan.stints)
    for st in plan.stints[1:]:
        p = q[st.start_lap]
        gesamt += cfg.pit_loss * ((1.0 - p) + p * cfg.sc_pit_loss_factor)
    return gesamt


def hindsight_value(cfg: RaceConfig, prozess: SafetyCarProcess,
                    n: int = 200, seed: int = 0) -> tuple[float, float]:
    """Erwartetes Optimum bei *bekanntem* Rennverlauf. Untere Schranke, Mittel und SE."""
    rng = random.Random(seed)
    summe = quadrat = 0.0
    for _ in range(n):
        folge = prozess.sample(rng, cfg.n_laps)
        bekannt = frozenset(lap - 1 for lap in range(1, cfg.n_laps + 1)
                            if folge[lap] == SC)
        wert = optimal_strategy(replace(cfg, sc_laps=bekannt)).green_time
        summe += wert
        quadrat += wert * wert
    mittel = summe / n
    varianz = max(0.0, quadrat / n - mittel * mittel)
    return mittel, (varianz / n) ** 0.5


# --------------------------------------------------------------- Telemetrie
def telemetry_source_quality(source) -> dict:
    """Anteil real gemessener gegen interpolierter Telemetriepunkte
    (siehe P07-Erweiterung, ``Telemetry.Source`` - bislang ungenutzt).

    ``get_telemetry()`` fuehrt zwei unterschiedlich getaktete Stroeme
    zusammen (``car_data``: Motor/Pedale, ``pos_data``: Position, groebere
    Taktung). "car"/"pos" sind echte Messpunkte aus dem jeweiligen Strom,
    "interpolation" sind synthetische Fuellpunkte, die FastF1 beim
    Zusammenfuehren auf ein gemeinsames Zeitraster einfuegt, wo keiner der
    beiden Stroeme einen eigenen Messpunkt hat.
    """
    s = np.asarray(source)
    if s.size == 0:
        return {"n": 0, "car": 0.0, "pos": 0.0, "interpolation": 0.0}
    return {
        "n": int(s.size),
        "car": float(np.mean(s == "car")),
        "pos": float(np.mean(s == "pos")),
        "interpolation": float(np.mean(s == "interpolation")),
    }


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


# --------------------------------------------------------- Rundenzeit-Simulation (P37)
def track_curvature(x, y, dist, window: int = 21) -> np.ndarray:
    """Kruemmung kappa(s) = |x'y'' - y'x''| / (x'^2+y'^2)^1.5 einer Ideallinie,
    numerisch ueber die Distanz differenziert (siehe P37).

    Vor der zweifachen Ableitung geglaettet (Savitzky-Golay): GPS-Rauschen
    in der Rohposition wuerde sonst durch die zweite Ableitung stark
    verstaerkt - differenzieren VOR dem Glaetten ist hier der Fehler, nicht
    danach.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    dist = np.asarray(dist, dtype=float)
    xs = savgol_filter(x, window, 3)
    ys = savgol_filter(y, window, 3)
    dx, dy = np.gradient(xs, dist), np.gradient(ys, dist)
    ddx, ddy = np.gradient(dx, dist), np.gradient(dy, dist)
    return np.abs(dx * ddy - dy * ddx) / np.maximum((dx**2 + dy**2) ** 1.5, 1e-9)


def simulate_lap(dist, kappa, mu_g: float, a_accel: float, a_brake: float,
                 v_top: float) -> tuple[np.ndarray, float]:
    """Quasi-stationaere Punktmassen-Rundenzeitsimulation (siehe P37): das
    Standardverfahren fuer diese Modellklasse (vgl. OptimumLap und aehnliche
    Rundenzeit-Simulatoren).

    Kurvengrenzgeschwindigkeit aus v = sqrt(mu_g / kappa) (Kreisbewegung,
    ``mu_g`` fasst Reibung und Abtrieb in einer effektiven
    Grenzbeschleunigung zusammen). Vorwaertspass: von jedem Punkt aus so
    schnell wie moeglich beschleunigen, gedeckelt durch die Kurvengrenze
    voraus. Rueckwaertspass: von jedem Punkt aus rueckwaerts so, dass eine
    Bremsung rechtzeitig vor der naechsten Kurve fertig ist. Das Minimum
    beider Passes ist die tatsaechlich fahrbare Geschwindigkeit.

    Returns:
        (Geschwindigkeit je Punkt in m/s, Rundenzeit in Sekunden)
    """
    dist = np.asarray(dist, dtype=float)
    kappa = np.asarray(kappa, dtype=float)
    v_max = np.minimum(np.sqrt(mu_g / np.maximum(kappa, 1e-6)), v_top)
    n = len(dist)
    ds = np.diff(dist, prepend=dist[0])
    ds[0] = ds[1] if n > 1 else 1.0

    v_vor = np.empty(n)
    v_vor[0] = v_max[0]
    for i in range(1, n):
        v_vor[i] = min(v_max[i], np.sqrt(v_vor[i - 1] ** 2 + 2 * a_accel * ds[i]))

    v = v_vor.copy()
    for i in range(n - 2, -1, -1):
        v[i] = min(v[i], np.sqrt(v[i + 1] ** 2 + 2 * a_brake * ds[i + 1]))

    dt = np.zeros(n)
    v_mittel = (v[1:] + v[:-1]) / 2
    dt[1:] = ds[1:] / np.maximum(v_mittel, 0.1)
    return v, float(dt.sum())


def calibrate_lap_model(dist, kappa, speed_real) -> dict:
    """Vier Fahrzeugparameter (mu_g, a_accel, a_brake, v_top) per kleinste
    Quadrate an eine echte Geschwindigkeitsspur anpassen (siehe P37).

    Returns:
        dict mit den vier Parametern (SI-Einheiten, m/s bzw. m/s^2) und
        ``rmse_ms`` (Guete des Fits in m/s).
    """
    dist = np.asarray(dist, dtype=float)
    kappa = np.asarray(kappa, dtype=float)
    speed_real = np.asarray(speed_real, dtype=float)

    def residuen(p):
        mu_g, a_b, a_br, v_top = p
        v_sim, _ = simulate_lap(dist, kappa, mu_g, a_b, a_br, v_top)
        return v_sim - speed_real

    start = [15.0, 10.0, 30.0, 95.0]
    grenzen = ([5.0, 3.0, 10.0, 70.0], [40.0, 20.0, 60.0, 120.0])
    ergebnis = least_squares(residuen, start, bounds=grenzen)
    mu_g, a_b, a_br, v_top = ergebnis.x
    return {"mu_g": float(mu_g), "a_accel": float(a_b), "a_brake": float(a_br),
           "v_top": float(v_top), "rmse_ms": float(np.sqrt(np.mean(ergebnis.fun ** 2)))}


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
