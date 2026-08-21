"""reine rechenfunktionen ohne FastF1-abhaengigkeit.

alles hier arbeitet auf numpy-arrays und ist damit ohne netzzugriff testbar.
die FastF1-anbindung liegt in :mod:`f1lab.session`.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, replace

import numpy as np
from scipy.optimize import least_squares
from scipy.signal import savgol_filter

# faustwerte aus der literatur. groessenordnung belastbar, keine messwerte.
FUEL_KG_PER_LAP = 1.8
FUEL_S_PER_KG = 0.03
M_DRY_KG = 798.0  # FIA-mindestgewicht 2024 (auto+fahrer, ohne kraftstoff)


# --------------------------------------------------------------- statistik
@dataclass(frozen=True)
class Interval:
    """punktschaetzer mit konfidenzintervall."""
    value: float
    lo: float
    hi: float

    @property
    def width(self) -> float:
        return self.hi - self.lo

    def overlaps(self, other: Interval) -> bool:
        """true, wenn sich die intervalle ueberschneiden.

        ueberlappende intervalle heissen: der unterschied ist mit diesen
        daten nicht belegbar.
        """
        return not (self.hi < other.lo or other.hi < self.lo)

    def __str__(self) -> str:
        return f"{self.value:.3f} [{self.lo:.3f}, {self.hi:.3f}]"


def bootstrap_median(values, n_resamples: int = 1000, alpha: float = 0.05,
                     seed: int | None = 42) -> Interval:
    """median mit bootstrap-konfidenzintervall.

    rundenzeiten sind rechtsschief. langsame runden gibt es viele, schnellere
    als das optimum nicht. der median ist deshalb robuster als der mittelwert,
    hat aber keine geschlossene formel fuer das konfidenzintervall.

    Args:
        values: beobachtungen, mindestens 2.
        n_resamples: anzahl der bootstrap-ziehungen.
        alpha: irrtumswahrscheinlichkeit, 0.05 ergibt ein 95%-intervall.
        seed: fuer reproduzierbare ergebnisse.

    Raises:
        ValueError: bei weniger als 2 werten.
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
    """ausreisser ueber die median absolute deviation.

    robuster als die standardabweichung: der schaetzer selbst wird nicht von
    den ausreissern verschoben. true bedeutet ausreisser.
    """
    v = np.asarray(values, dtype=float)
    med = np.nanmedian(v)
    mad = np.nanmedian(np.abs(v - med))
    if mad == 0:
        return np.zeros_like(v, dtype=bool)
    modified_z = 0.6745 * (v - med) / mad
    return np.abs(modified_z) > threshold


# --------------------------------------------------------------- rating
def elo_expected(rating_a: float, rating_b: float) -> float:
    """erwartete punktzahl von A gegen B nach dem Elo-modell, zwischen 0 und 1.

    gleiche ratings ergeben 0.5. 400 punkte vorsprung bedeuten eine erwartete
    siegquote von rund 91 prozent.
    """
    return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))


def elo_update(rating_a: float, rating_b: float, score_a: float,
               k: float = 24.0) -> tuple[float, float]:
    """neue ratings nach einem einzelnen duell.

    Args:
        score_a: ergebnis aus sicht von A. 1 fuer einen sieg, 0 fuer eine
            niederlage, 0.5 fuer ein unentschieden.
        k: wie stark ein einzelnes duell das rating bewegt. klein haelt das
            rating traege: es braucht viele duelle, um es zu verschieben.
            gross macht es sprunghaft.

    Returns:
        (neues rating A, neues rating B). Elo ist ein nullsummenspiel: was A
        gewinnt, verliert B exakt.
    """
    delta = k * (score_a - elo_expected(rating_a, rating_b))
    return rating_a + delta, rating_b - delta


# --------------------------------------------------------------- treibstoff
def fuel_correct(lap_times, lap_numbers, total_laps: int,
                 kg_per_lap: float = FUEL_KG_PER_LAP,
                 s_per_kg: float = FUEL_S_PER_KG) -> np.ndarray:
    """rundenzeiten auf konstante tankfuellung normieren.

    ein auto verliert ueber die renndistanz rund 100 kg sprit und wird dadurch
    kontinuierlich schneller. ohne korrektur sieht jeder reifen am rennende
    besser aus als er ist, und die degradation wird unterschaetzt.

    die korrektur addiert auf jede runde die zeit, die das auto mit der zu
    diesem zeitpunkt noch vorhandenen restmenge sprit schneller gewesen waere.
    bezugspunkt ist das rennende (leerer tank).
    """
    t = np.asarray(lap_times, dtype=float)
    n = np.asarray(lap_numbers, dtype=float)
    remaining_laps = total_laps - n
    return t - remaining_laps * kg_per_lap * s_per_kg


# --------------------------------------------------------------- degradation
@dataclass(frozen=True)
class DegradationFit:
    """ergebnis einer degradationsschaetzung fuer einen stint."""
    slope: float          # sekunden pro runde reifenalter
    intercept: float      # extrapolierte zeit bei reifenalter 0
    r2: float             # bestimmtheitsmass
    n: int                # ausgewertete runden

    @property
    def is_reliable(self) -> bool:
        """grobe plausibilitaetspruefung.

        unter 6 runden ist die steigung rauschen. unter R^2 = 0.3 beschreibt
        die gerade das verhalten nicht.
        """
        return self.n >= 6 and self.r2 >= 0.3


def fit_degradation(tyre_life, lap_times) -> DegradationFit:
    """lineare regression rundenzeit ueber reifenalter.

    die steigung ist die degradation in sekunden pro runde. erwartet werden
    fuel-korrigierte zeiten. sonst mischt sich der gewichtseffekt hinein und
    kompensiert die degradation teilweise weg.
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
    """sucht den knickpunkt, ab dem der reifen deutlich schneller abbaut.

    reifen degradieren nicht linear. ab einem gewissen punkt, dem cliff,
    bricht der grip ueberproportional ein. die funktion probiert jeden
    moeglichen bruchpunkt durch und waehlt den mit der kleinsten
    gesamt-fehlerquadratsumme.

    Returns:
        (reifenalter am knick oder None, fit davor, fit danach oder None)

    ein ergebnis von None heisst: die zweiteilige anpassung ist nicht besser
    als die einfache gerade. der stint war also linear oder zu kurz.
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
    if single_sse < 1e-6:
        # praktisch perfekt linear. restfehler ist nur noch
        # gleitkomma-rauschen, keine echte streuung mehr, wie bei einer
        # rauschfreien synthetischen gerade. der vergleich sse > 0.8*single_sse
        # waere dann ein vergleich von rauschen gegen rauschen: je nach
        # BLAS/LAPACK der plattform kippt er in die eine oder andere
        # richtung (in CI mit anderer numpy-version beobachtet, lokal nicht
        # reproduziert). ohne echte streuung gibt es nichts zu erklaeren,
        # also kein cliff.
        return None, single, None

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
    # zwei geraden haben zwei parameter mehr. nur akzeptieren, wenn der
    # fehler deutlich faellt und der zweite abschnitt staerker ansteigt.
    if sse > 0.8 * single_sse or right.slope <= left.slope:
        return None, single, None

    return int(x[i]), left, right


# --------------------------------------------------------------- strategie
def estimate_pit_loss(in_lap_deltas, out_lap_deltas) -> float:
    """zeitverlust eines boxenstopps aus beobachteten runden schaetzen.

    erwartet die differenz von in- bzw. out-lap zur normalen rundenzeit
    desselben fahrers. der median ist robust gegen die faelle, in denen
    waehrend des stopps etwas schiefging.
    """
    a = np.asarray(in_lap_deltas, dtype=float)
    b = np.asarray(out_lap_deltas, dtype=float)
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    if a.size == 0 or b.size == 0:
        raise ValueError("In- und Out-Lap-Deltas duerfen nicht leer sein")
    return float(np.median(a) + np.median(b))


def undercut_gain(deg_old: float, deg_new: float, n_laps: int = 3,
                  out_lap_penalty: float = 0.6) -> float:
    """zeitgewinn, wenn ein fahrer n runden frueher an die box geht.

    der verfolger stoppt frueher und faehrt auf frischen reifen, waehrend der
    vordermann weiter altert. der pitloss selbst faellt fuer beide an und
    kuerzt sich heraus. entscheidend ist nur die pace-differenz im fenster.

    Args:
        deg_old: degradation des alten reifens, s pro runde.
        deg_new: degradation des frischen reifens, s pro runde.
        n_laps: wie viele runden frueher gestoppt wird.
        out_lap_penalty: aufschlag auf die out-lap. der reifen ist noch
            kalt, groessenordnung eine halbe bis eine sekunde.

    Returns:
        sekunden. positiv heisst, der undercut lohnt sich.

    nicht modelliert: verkehr. wer hinter einem langsameren auto herauskommt,
    verliert den vorteil in einer runde wieder.
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
    """findet die rundenzahl mit dem groessten undercut-gewinn.

    Returns:
        (anzahl runden, gewinn in sekunden)
    """
    gains = [(n, undercut_gain(deg_old, deg_new, n, out_lap_penalty))
             for n in range(1, max_laps + 1)]
    return max(gains, key=lambda t: t[1])


# --------------------------------------------------------- rennstrategie (P35)
# exakter boxenstopp-plan als kuerzester pfad in einem DAG, plus eine
# safety-car-politik per rueckwaertsinduktion. siehe P35 fuer die herleitung:
# die stintkosten haengen nur von mischung und laenge ab, nicht von der
# position im rennen. das macht das problem zu einem kuerzesten pfad.
#
# GRUEN/SC sind die zwei flaggenzustaende, ueber die SafetyCarProcess/
# solve_policy/roll_out sich verstaendigen. 0/1 statt strings: beide module
# vergleichen sie oft in engen schleifen.
GRUEN, SC = 0, 1
UNENDLICH = float("inf")


@dataclass(frozen=True)
class TyreModel:
    """rundenzeitmodell einer mischung, treibstoffkorrigiert.

    lap_time(alter) = basis + linear * (alter - 1) + quadratisch * (alter - 1)^2

    das reifenalter ist einsbasiert: alter 1 ist die erste runde auf dem satz,
    also ist ``base_time`` die zeit auf frischem gummi. die nullbasierte
    variante waere eine hypothetische runde auf einem null runden alten
    reifen. das ist nicht interpretierbar, und jeder indexfehler bliebe still.

    der quadratische term bildet den abbau am stintende ab. das macht das
    modell nicht komplizierter: die stintkosten werden vorab je (mischung,
    laenge) ausgerechnet. die rundenzeitfunktion darf beliebig krumm sein,
    solange die optimierung nur fertige zahlen sieht.
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
        """reine fahrzeit eines stints ueber ``length`` runden, ohne pitloss."""
        if length < 1:
            raise ValueError("Stintlaenge muss mindestens 1 sein")
        return sum(self.lap_time(a) for a in range(1, length + 1))


@dataclass(frozen=True)
class RaceConfig:
    """alles, was die aufgabe festlegt.

    ``fuel_effect`` steht bewusst nicht in der zielfunktion. jede strategie
    faehrt die runden 1..L genau einmal, also ist der treibstoffterm fuer
    alle plaene identisch: eine additive konstante, die das optimum nicht
    bewegen kann. sie wird nur fuer die anzeige wieder addiert, damit die
    rennzeit auf einer erkennbaren skala steht.

    wichtig: die basiszeiten muessen treibstoffkorrigiert sein. der term
    ``LapNumber`` gehoert in die degradationsschaetzung (P13), aber nicht
    zusaetzlich in ``base_time``. sonst zaehlt er doppelt.
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
        """treibstoffzeit ueber das ganze rennen. fuer jede strategie gleich."""
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
        """runden, an deren ende an die box gefahren wird."""
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
    """keine strategie erfuellt die gesetzten bedingungen."""


def pit_loss_at(cfg: RaceConfig, lap: int) -> float:
    """zeitverlust eines stopps am ende von ``lap``.

    unter safety car ist das feld langsam, der relative preis der boxengasse
    bricht ein. ``sc_pit_loss_factor`` ist der anteil, der uebrig bleibt.
    """
    if lap in cfg.sc_laps:
        return cfg.pit_loss * cfg.sc_pit_loss_factor
    return cfg.pit_loss


def stint_arcs(cfg: RaceConfig) -> list[tuple[int, int, int, float]]:
    """alle legalen stints als (mischungsindex, startrunde, endrunde, kosten)."""
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
                # ein stint, der vor der flagge endet, braucht einen
                # nachfolger, der selbst wieder lang genug ist
                if end < cfg.n_laps and (cfg.n_laps - end) < cfg.min_stint:
                    continue
                eintritt = 0.0 if start == 1 else pit_loss_at(cfg, start - 1)
                arcs.append((ci, start, end, tyre.stint_time(length) + eintritt))
    return arcs


def optimal_strategy(cfg: RaceConfig) -> Strategy:
    """exaktes optimum per dynamischer programmierung ueber den stint-DAG.

    der zustand traegt neben dem knoten eine bitmaske der bisher benutzten
    mischungen. ohne die waere die zweimischungs-regel nicht durchsetzbar: ein
    reiner kuerzester pfad hat kein gedaechtnis, und am ende zu filtern
    verliert die exaktheit. bei drei mischungen kostet die maske einen
    faktor 8.

    die stoppzahl steht nur dann im zustand, wenn sie eingeschraenkt ist. sie
    immer mitzufuehren vervielfacht den zustandsraum um die rundenzahl, fuer
    eine bedingung, die meistens gar nicht gesetzt ist.
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
    """bester plan je exakter stoppzahl.

    so ist die frage im rennen gestellt. "die fuenf besten strategien" liefert
    fuenfmal dieselbe idee mit der boxenrunde um eins verschoben. "bester
    einstopper gegen besten zweistopper und der abstand dazwischen" ist die
    entscheidung. der abstand misst auch, wie viel risiko man kauft: drei
    sekunden sind ein muenzwurf, fuenfundzwanzig nicht.
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
    """pitloss-werte, an denen die optimale stoppzahl kippt.

    der pitloss ist die unsicherste eingabe. er haengt an strecke, verkehr und
    daran, ob der stopp sauber laeuft. das optimum ist stueckweise konstant in
    ihm, interessant ist also nicht der wert, sondern wo er springt.
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
    """zweizustands-markow-kette auf dem entscheidungspunkt jeder runde.

    grob, aber mit genau zwei parametern, die sich beide aus FastF1
    ``TrackStatus`` auszaehlen lassen (siehe P18). sie erzeugt die zwei
    eigenschaften, auf die es ankommt: safety cars sind selten, und wenn sie
    kommen, bleiben sie ein paar runden.

    ``sc[l]`` gilt fuer die entscheidung zu beginn von runde l. der kuerzeste
    pfad indiziert safety-car-runden dagegen nach der runde, an deren *ende*
    gestoppt wird. die beiden passen ueber ``l - 1`` zusammen. ein fehler an
    dieser stelle verschiebt jeden vergleich um eine runde und sieht trotzdem
    plausibel aus.
    """

    p_deploy: float = 0.025
    p_end: float = 0.25

    def next_probs(self, state: int) -> tuple[float, float]:
        if state == GRUEN:
            return 1.0 - self.p_deploy, self.p_deploy
        return self.p_end, 1.0 - self.p_end

    def marginals(self, n_laps: int) -> list[float]:
        """q[l] = P(safety car am entscheidungspunkt von runde l), 1-basiert."""
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
    """optimale politik per rueckwaertsinduktion. gibt (erwartungswert, politik).

    zustand ist (runde, mischung, reifenalter, benutzte mischungen, flagge),
    aktion ist ausbleiben oder auf mischung c' wechseln.

    warum nicht einfach jede runde neu planen: der kuerzeste pfad kann
    "vielleicht kommt gleich ein safety car" gar nicht ausdruecken. neuplanen
    mit punktschaetzung ignoriert deshalb den optionswert des wartens. die
    politik hier dehnt einen stint manchmal ueber das deterministische
    optimum hinaus, nur um die option offenzuhalten.
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

    # endebene: das rennen ist vorbei. zulaessig nur, wenn der letzte stint
    # lang genug war und die zweimischungs-regel erfuellt ist.
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
    """politik gegen einen konkreten rennverlauf ausfahren."""
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
    """erwartete kosten, wenn ein fester plan stur durchgezogen wird.

    geschlossene form, keine simulation: der einzige zufaellige anteil ist der
    pitloss je (fester) boxenrunde.
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
    """erwartetes optimum bei *bekanntem* rennverlauf. untere schranke, mittel und SE."""
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


# --------------------------------------------------------------------- verkehr (P41)
def lap_times_for_strategy(cfg: RaceConfig, strategy: Strategy) -> np.ndarray:
    """freie rundenzeit je runde 1..n_laps fuer eine feste strategie, ohne
    verkehr (siehe P41, eingabe fuer :func:`gap_evolution`).

    "frei" heisst: exakt die zeiten, aus denen :func:`optimal_strategy` selbst
    rechnet (reifenmodell plus pitloss auf der boxenrunde). keine neue
    rechnung, nur dieselbe kostenstruktur je runde statt nur summiert.
    """
    zeiten = np.empty(cfg.n_laps)
    for st in strategy.stints:
        tyre = next(t for t in cfg.tyres if t.compound == st.compound)
        for lap in range(st.start_lap, st.end_lap + 1):
            zeiten[lap - 1] = tyre.lap_time(lap - st.start_lap + 1)
    for lap in strategy.pit_laps:
        zeiten[lap - 1] += pit_loss_at(cfg, lap)
    return zeiten


def gap_evolution(hero_times, rival_times, initial_gap: float,
                  p_overtake: float, block_gap_s: float = 1.0,
                  rng: random.Random | None = None) -> tuple[np.ndarray, int]:
    """rundenweiser abstand zweier autos, mit ueberholwahrscheinlichkeit statt
    der annahme, dass ein tempovorteil sich sofort in position uebersetzt
    (siehe P41, die in P35 bewusst offen gelassene "verkehr"-luecke, jetzt
    mit P38/P39 als eingabegroesse fuer ``p_overtake``).

    ``gap`` > 0 heisst: der ueberholer (hero) liegt so viele sekunden hinter
    dem vordermann (rival). solange der freie (unblockierte) abstand unter
    ``block_gap_s`` faellt (grob die distanz, auf der DRS/windschatten
    wirken), passiert der ueberholvorgang nicht automatisch. jede solche
    runde ist ein versuch mit erfolgswahrscheinlichkeit ``p_overtake``.
    scheitert er, haelt hero den abstand bei ``block_gap_s`` (kann nicht
    weiter aufschliessen, faellt aber auch nicht zurueck) statt den vollen
    tempovorteil sofort zu realisieren. genau das ist der mechanismus, den
    das reine rundenzeitmodell aus P35 nicht kennt.

    Args:
        hero_times, rival_times: rundenzeiten je runde (inkl. pitloss auf der
            boxenrunde, siehe :func:`lap_times_for_strategy`), gleiche laenge.
        initial_gap: abstand vor runde 1 (positiv = hero hinten).
        p_overtake: erfolgswahrscheinlichkeit je ueberholversuch-runde.
            streckeneigenschaft (siehe P38: kurven/km korreliert mit
            ueberholzahlen; P39: rund drei viertel der ueberholungen in der
            DRS-zone), hier bewusst als externer parameter statt intern
            geschaetzt. siehe P41-docstring fuer die kalibrierungsgrenzen.
        block_gap_s: abstand, unter dem "blockiert" gilt.
        rng: fuer reproduzierbare zufallszahlen. ohne wird ein neuer erzeugt.

    Returns:
        (abstandsverlauf inkl. runde 0 an index 0, anzahl blockierter runden).
    """
    hero_times = np.asarray(hero_times, dtype=float)
    rival_times = np.asarray(rival_times, dtype=float)
    if hero_times.shape != rival_times.shape:
        raise ValueError("hero_times und rival_times muessen gleich lang sein")
    rng = rng if rng is not None else random.Random()
    n = hero_times.size
    verlauf = np.empty(n + 1)
    verlauf[0] = gap = initial_gap
    blockiert = 0
    for lap in range(n):
        frei = gap + (hero_times[lap] - rival_times[lap])
        if gap > 0 and frei <= block_gap_s:
            blockiert += 1
            gap = frei if rng.random() < p_overtake else block_gap_s
        else:
            gap = frei
        verlauf[lap + 1] = gap
    return verlauf, blockiert


def traffic_cost(hero_times, rival_times, initial_gap: float,
                 p_overtake: float, block_gap_s: float = 1.0,
                 n_sim: int = 2000, seed: int = 0) -> tuple[float, float]:
    """erwarteter zeitverlust durch verkehr gegenueber der freien annahme,
    ueber viele zufallslaeufe (siehe P41). mittel und standardfehler, im
    selben stil wie :func:`hindsight_value`.

    "zeitverlust" ist der endabstand MIT blockade minus der endabstand OHNE
    (reine summe der rundenzeiten). positiv heisst, hero verliert durch
    verkehr zeit auf rival, unabhaengig davon, ob am ende noch ueberholt
    wurde oder nicht.
    """
    hero_times = np.asarray(hero_times, dtype=float)
    rival_times = np.asarray(rival_times, dtype=float)
    frei_ende = initial_gap + float((hero_times - rival_times).sum())
    rng = random.Random(seed)
    summe = quadrat = 0.0
    for _ in range(n_sim):
        verlauf, _ = gap_evolution(hero_times, rival_times, initial_gap,
                                   p_overtake, block_gap_s, rng)
        verlust = verlauf[-1] - frei_ende
        summe += verlust
        quadrat += verlust * verlust
    mittel = summe / n_sim
    varianz = max(0.0, quadrat / n_sim - mittel * mittel)
    return mittel, (varianz / n_sim) ** 0.5


# --------------------------------------------------------------- telemetrie
def telemetry_source_quality(source) -> dict:
    """anteil real gemessener gegen interpolierter telemetriepunkte
    (siehe P07-erweiterung, ``Telemetry.Source``, bislang ungenutzt).

    ``get_telemetry()`` fuehrt zwei unterschiedlich getaktete stroeme
    zusammen (``car_data``: motor/pedale, ``pos_data``: position, groebere
    taktung). "car"/"pos" sind echte messpunkte aus dem jeweiligen strom,
    "interpolation" sind synthetische fuellpunkte, die FastF1 beim
    zusammenfuehren auf ein gemeinsames zeitraster einfuegt, wo keiner der
    beiden stroeme einen eigenen messpunkt hat.
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
    """zerlegt den bremskanal in einzelne bremszonen.

    der brake-kanal ist binaer. ueber die flanken lassen sich
    zusammenhaengende bremsphasen finden und je zone
    eintrittsgeschwindigkeit, laenge und mittlere verzoegerung berechnen.

    Args:
        brake: bremssignal, wird nach bool konvertiert.
        distance: zurueckgelegte distanz in metern.
        speed: geschwindigkeit in km/h.
        time: zeit in sekunden.
        min_length_m: kuerzere zonen sind meist messrauschen.

    Returns:
        liste von zonen mit kennwerten, sortiert nach distanz.
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
    #   +1 -> b[i+1] ist die erste bremsende probe  -> start = i + 1
    #   -1 -> b[i]   ist die letzte bremsende probe -> end   = i
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
    """paart indizes zweier positionslisten ueber die naechstgelegene distanz.

    fuer ereignisse, die beide seiten auf derselben strecke haben (brems-
    oder mini-sektor-grenzen zweier fahrer zum beispiel), aber nicht exakt
    an derselben stelle. je wert aus ``a`` wird der naechste, noch nicht
    vergebene wert aus ``b`` gesucht. bleibt keiner innerhalb ``tolerance``,
    bleibt der wert unverpaart. das ist kein fehlerfall: ungleich viele
    ereignisse (z.b. eine zusaetzliche bremsung) sind der normalfall.

    Args:
        a, b: positionen (z.b. meter), beliebige reihenfolge.
        tolerance: maximaler abstand fuer eine gueltige paarung.

    Returns:
        liste von (index in a, index in b), sortiert wie a.
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
    """zerlegt ein beliebiges binaeres signal in zusammenhaengende
    distanz-zonen. dieselbe flankenlogik wie :func:`braking_zones`, aber
    ohne die bremsspezifischen kennwerte (geschwindigkeit/verzoegerung).
    fuer DRS-aktivzonen gedacht, funktioniert fuer jedes binaere
    distanzsignal (z.b. auch throttle > 0).

    Args:
        active: binaeres signal, wird nach bool konvertiert.
        distance: zurueckgelegte distanz in metern.
        min_length_m: kuerzere zonen sind meist messrauschen.

    Returns:
        liste von zonen (start_m, end_m, length_m), sortiert nach distanz.
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


def distance_in_any_zone(distances, zone_starts, zone_ends) -> np.ndarray:
    """fuer jede distanz: liegt sie innerhalb irgendeiner der zonen
    [start, end]? (siehe P39, ueberholorte gegen DRS-zonen).

    allgemein gehalten wie :func:`active_distance_zones`. nimmt beliebige
    start/ende-paare, nicht nur DRS.

    Returns:
        bool-array derselben laenge wie ``distances``.
    """
    d = np.asarray(distances, dtype=float)
    starts = np.asarray(zone_starts, dtype=float)
    ends = np.asarray(zone_ends, dtype=float)
    if starts.size == 0:
        return np.zeros(d.shape, dtype=bool)
    return ((d[:, None] >= starts[None, :])
           & (d[:, None] <= ends[None, :])).any(axis=1)


def lead_distance_to_zone(distances, zone_starts, track_length_m: float) -> np.ndarray:
    """fuer jede distanz: wie weit bis zum beginn der naechsten zone in
    fahrtrichtung? (siehe P39, dritte AUSBAUSTUFE: ueberholorte gegen
    bremszonen).

    mit rundenumbruch: liegt die naechste zone erst in der naechsten
    runde, zaehlt die reststrecke bis zum ziel plus die distanz vom start
    bis zur zone. anders als :func:`distance_in_any_zone` (nur drinnen
    oder draussen) misst das hier einen abstand, auch ausserhalb jeder
    zone. noetig, um zu pruefen, ob ereignisse sich in der ANNAEHERUNG an
    eine zone haeufen, nicht nur exakt darin.

    Returns:
        float-array derselben laenge wie ``distances`` (NaN ohne zonen).
    """
    d = np.asarray(distances, dtype=float)
    starts = np.sort(np.asarray(zone_starts, dtype=float))
    if starts.size == 0:
        return np.full(d.shape, np.nan)
    idx = np.searchsorted(starts, d, side="left")
    wrap = idx >= starts.size
    naechste = np.where(wrap, starts[0] + track_length_m,
                        starts[np.clip(idx, 0, starts.size - 1)])
    return naechste - d


def drs_state(drs_values, open_codes: tuple[int, ...] = (10, 12, 14),
              detected_code: int = 8):
    """klassifiziert den codierten DRS-kanal in drei zustaende.

    0 = zu, 1 = erkannt/im aktivierungsbereich (code 8), 2 = offen. FastF1s
    eigene dokumentation bezeichnet die codes unterhalb von 10 als unsicher
    ("Unknown Distinction", "Noted Sometimes"). das hier ist die in der
    community uebliche lesart, nicht eine offiziell bestaetigte.

    Returns:
        ganzzahl-array derselben laenge wie ``drs_values``.
    """
    v = np.asarray(drs_values)
    status = np.zeros(v.shape, dtype=int)
    status[v == detected_code] = 1
    status[np.isin(v, open_codes)] = 2
    return status


# --------------------------------------------------------------- streckengeometrie
def path_length(x, y, closed: bool = True) -> float:
    """laenge eines streckenzugs als summe der segmentlaengen.

    Args:
        x, y: koordinaten in derselben einheit. das ergebnis traegt sie
            ebenfalls.
        closed: schliesst den weg vom letzten zurueck zum ersten punkt. eine
            rennrunde endet dort, wo sie beginnt. ohne das schlusssegment
            fehlt genau die luecke zwischen letzter probe und start-ziel.

    Returns:
        gesamtlaenge. ein weg aus weniger als zwei punkten hat laenge 0.
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


# --------------------------------------------------------- rundenzeit-simulation (P37)
def track_curvature(x, y, dist, window: int = 21) -> np.ndarray:
    """kruemmung kappa(s) = |x'y'' - y'x''| / (x'^2+y'^2)^1.5 einer ideallinie,
    numerisch ueber die distanz differenziert (siehe P37).

    vor der zweifachen ableitung geglaettet (Savitzky-Golay): GPS-rauschen
    in der rohposition wuerde sonst durch die zweite ableitung stark
    verstaerkt. differenzieren VOR dem glaetten ist hier der fehler, nicht
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
    """quasi-stationaere punktmassen-rundenzeitsimulation (siehe P37): das
    standardverfahren fuer diese modellklasse (vgl. OptimumLap und aehnliche
    rundenzeit-simulatoren).

    kurvengrenzgeschwindigkeit aus v = sqrt(mu_g / kappa) (kreisbewegung,
    ``mu_g`` fasst reibung und abtrieb in einer effektiven
    grenzbeschleunigung zusammen). vorwaertspass: von jedem punkt aus so
    schnell wie moeglich beschleunigen, gedeckelt durch die kurvengrenze
    voraus. rueckwaertspass: von jedem punkt aus rueckwaerts so, dass eine
    bremsung rechtzeitig vor der naechsten kurve fertig ist. das minimum
    beider passes ist die tatsaechlich fahrbare geschwindigkeit.

    Returns:
        (geschwindigkeit je punkt in m/s, rundenzeit in sekunden)
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
    """vier fahrzeugparameter (mu_g, a_accel, a_brake, v_top) per kleinste
    quadrate an eine echte geschwindigkeitsspur anpassen (siehe P37).

    Returns:
        dict mit den vier parametern (SI-einheiten, m/s bzw. m/s^2) und
        ``rmse_ms`` (guete des fits in m/s).
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


def simulate_stint(dist, kappa, mu_g_ref: float, a_accel_ref: float,
                   a_brake_ref: float, v_top: float, fuel_start_kg: float,
                   n_laps: int, kg_per_lap: float = FUEL_KG_PER_LAP,
                   m_dry_kg: float = M_DRY_KG) -> np.ndarray:
    """rundenzeit je runde eines stints, waehrend der tank leerer wird
    (P37, zweite AUSBAUSTUFE).

    die kalibrierten grenzwerte (``*_ref``) gelten fuer eine qualifyingrunde,
    also nahezu leeren tank (siehe ``calibrate_lap_model``). das ist hier der
    bezugspunkt bei kraftstoffmasse 0. modellannahme: die kraefte hinter
    kurvengrip, laengsbeschleunigung und bremsung (abtrieb, reifenhaftung,
    bremsanlage) haengen kaum vom fahrzeuggewicht ab, nur die erreichbare
    beschleunigung a=F/m schon. mit vollerem tank (hoehere masse) sinken alle
    drei grenzwerte deshalb um denselben faktor m_dry/(m_dry+kraftstoff).
    grobe vereinfachung (echte bremskraft haengt z.b. auch von der
    gewichtsabhaengigen normalkraft ab), aber ohne zweiten kalibrierungspunkt
    nicht weiter auftrennbar. deshalb hier bewusst EINE skalierung fuer alle
    drei statt einer pro groesse. v_top bleibt konstant (leistungs-/
    luftwiderstandsbegrenzt, kaum massenabhaengig).

    Args:
        fuel_start_kg: kraftstoffmasse zu stint-beginn.
        n_laps: rundenzahl des stints.
        kg_per_lap: verbrauch je runde, default wie ``fuel_correct``.
        m_dry_kg: fahrzeugmasse ohne kraftstoff (bezugsgewicht der
            skalierung).

    Returns:
        rundenzeit in sekunden je runde, laenge ``n_laps``. faellt monoton,
        waehrend der tank leerer wird, danach konstant.
    """
    fuel = np.maximum(fuel_start_kg - kg_per_lap * np.arange(n_laps), 0.0)
    skala = m_dry_kg / (m_dry_kg + fuel)
    zeiten = np.empty(n_laps)
    for i, s in enumerate(skala):
        _, zeiten[i] = simulate_lap(dist, kappa, mu_g_ref * s, a_accel_ref * s,
                                    a_brake_ref * s, v_top)
    return zeiten


@dataclass(frozen=True)
class Elevation:
    """hoehenprofil einer runde, alle werte in metern."""
    gain: float                 # summierter anstieg
    drop: float                 # summierter abstieg
    span: float                 # hoechster minus tiefster punkt

    @property
    def is_flat(self) -> bool:
        """unter 10 m spannweite ist eine strecke praktisch eben."""
        return self.span < 10.0


def elevation_profile(z, min_step: float = 1.0) -> Elevation:
    """hoehenmeter aus dem Z-kanal der positionsdaten.

    die rohwerte rauschen um mehrere dezimeter. wer einfach alle betraege der
    differenzen aufsummiert, zaehlt dieses rauschen tausendfach mit und landet
    bei hoehenmetern, die um eine groessenordnung zu hoch sind. deshalb hier
    dieselbe hysterese, die auch GPS-tracker verwenden: ein anstieg zaehlt
    erst, wenn er seit der letzten richtungsumkehr min_step ueberschreitet.

    die daempfung wirkt gegen *korreliertes* rauschen, wie positionsdaten es
    zeigen. dort faellt sie auf null. gegen unabhaengiges weisses rauschen
    hilft sie nur teilweise: dessen einzelsprunge ueberschreiten die schwelle
    weiterhin gelegentlich. auf ungeglaetteten daten also min_step anheben.

    Args:
        z: hoehenwerte in metern.
        min_step: schwelle in metern, unterhalb derer eine aenderung als
            rauschen gilt.

    Returns:
        Elevation mit anstieg, abstieg und spannweite.
    """
    v = np.asarray(z, dtype=float)
    v = v[np.isfinite(v)]
    if v.size < 2:
        return Elevation(0.0, 0.0, 0.0)

    gain = drop = 0.0
    direction = 0          # +1 steigend, -1 fallend, 0 noch unentschieden
    pivot = v[0]           # letzter bestaetigter wendepunkt
    peak = v[0]            # laufendes extremum seit dem wendepunkt

    for cur in v[1:]:
        if direction > 0:
            if cur > peak:
                peak = cur                      # anstieg laeuft weiter
            elif cur <= peak - min_step:        # umkehr bestaetigt
                gain += peak - pivot            # -> anstieg verbuchen
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

    # letzter abschnitt endet ohne umkehr und muss noch verbucht werden.
    if direction > 0:
        gain += peak - pivot
    elif direction < 0:
        drop += pivot - peak

    return Elevation(round(float(gain), 1), round(float(drop), 1),
                     round(float(v.max() - v.min()), 1))


# --------------------------------------------------------------- race control
def status_intervals(status, time_s
                     ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """sparses status-log (nur aenderungen, wie FastF1s ``track_status``) in
    intervalle umwandeln.

    ``track_status`` traegt eine zeile pro zustandswechsel, nicht eine pro
    zeitschritt. "SCDeployed" erscheint typischerweise genau einmal, auch
    wenn das safety car neun runden lang draussen bleibt. ein intervall endet
    deshalb dort, wo das naechste beginnt, nicht am letzten auftreten
    desselben werts (das waere derselbe zeitpunkt wie der start). ein
    gruppieren nach gleichem status ergibt hier fast ausschliesslich
    intervalle der laenge 0.

    Args:
        status: zustandswerte, ein eintrag pro aenderung.
        time_s: zugehoerige zeitpunkte in sekunden, aufsteigend sortiert.

    Returns:
        (status, start, ende) als drei gleich lange arrays. ``ende`` des
        letzten intervalls ist NaN. offen, weil das log dort endet.
        unmittelbar wiederholte werte (zwei zeilen mit demselben status ohne
        wechsel dazwischen) werden zu einem intervall zusammengefasst.
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
