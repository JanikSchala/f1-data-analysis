"""flaggt technische probleme aus telemetrie-rundenprofilen per isolation forest und vergleicht das mit einem 1d-conv-autoencoder auf rohen telemetriespuren"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")                      # kein Fenster, nur Dateien

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.ensemble import IsolationForest
from torch import nn

import f1lab
from f1lab.design import FG, GRID, MUTED, PHASE, SERIEN, matplotlib_stil

warnings.filterwarnings("ignore")

OUT = Path(__file__).parent / "out"
OUT.mkdir(exist_ok=True)

SEASON, EVENT, IDENT = 2024, "Baku", "R"
CONTAMINATION = 0.04
FEAT = ["lap_time", "vmax", "vmean", "rpm_max", "rpm_mean", "throttle_mean",
       "brake_pct", "gear_mean"]

# Autoencoder auf rohen Telemetriespuren
TRACE_KANAELE = ["Speed", "RPM", "Throttle", "Brake"]
TRACE_PUNKTE = 96      # durch 4 teilbar: zwei Halbierungen im Encoder passen exakt
AE_BOTTLENECK = 8
AE_EPOCHS = 80

plt.rcParams.update(matplotlib_stil())


def rundenprofile(ses) -> pd.DataFrame:
    """Feature-Profil je Runde für ALLE Runden, nicht nur pick_wo_box().pick_accurate().
    Diese übliche Filterung würde die interessantesten Fälle regelmäßig wegfiltern.
    Box- und Neutralisations-Zugehörigkeit bleiben als Metadaten erhalten, um gezielt statt blind auszuschließen."""
    phasen = f1lab.track_status_phases(ses)
    neutral_laps = set()
    for p in phasen[phasen["label"] != "gruen"].itertuples():
        neutral_laps.update(range(p.lap_start, p.lap_end + 1))

    rows = []
    for lap in ses.laps.itertuples():
        if pd.isna(lap.LapTime):
            continue
        try:
            tel = ses.laps.loc[[lap.Index]].get_car_data()
        except Exception:
            continue
        if tel is None or len(tel) < 50:
            continue
        rows.append({
            "driver": lap.Driver, "lap": lap.LapNumber,
            "lap_time": lap.LapTime.total_seconds(),
            "vmax": tel["Speed"].max(), "vmean": tel["Speed"].mean(),
            "rpm_max": tel["RPM"].max(), "rpm_mean": tel["RPM"].mean(),
            "throttle_mean": tel["Throttle"].mean(),
            "brake_pct": 100 * tel["Brake"].astype(bool).mean(),
            "gear_mean": tel["nGear"].mean(),
            "boxenrunde": pd.notna(lap.PitInTime) or pd.notna(lap.PitOutTime),
            "neutralisiert": lap.LapNumber in neutral_laps,
            "erste_runde": lap.LapNumber == 1,
        })
    return pd.DataFrame(rows).dropna()


class LapAutoencoder(nn.Module):
    """1D-Conv-Autoencoder für Telemetriespuren.

    Zwei Halbierungen im Encoder (96 -> 48 -> 24 Punkte), symmetrisch
    rückgängig gemacht im Decoder. Der schmale Flaschenhals (8 Zahlen für
    4x96=384 Eingabewerte) erzwingt eine komprimierte Repräsentation der
    "normalen" Rundenform. Bei nur einigen hundert Trainingsrunden würde
    ein größeres Netz nur auswendig lernen.
    """

    def __init__(self, n_channels: int = len(TRACE_KANAELE),
                n_points: int = TRACE_PUNKTE, bottleneck: int = AE_BOTTLENECK):
        super().__init__()
        halb = n_points // 4
        self.halb, self.kanaele_innen = halb, 32
        self.encoder = nn.Sequential(
            nn.Conv1d(n_channels, 16, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
        )
        self.zu_bottleneck = nn.Linear(32 * halb, bottleneck)
        self.von_bottleneck = nn.Linear(bottleneck, 32 * halb)
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(16, n_channels, kernel_size=4, stride=2, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        b = self.zu_bottleneck(z.flatten(1))
        z2 = self.von_bottleneck(b).view(-1, self.kanaele_innen, self.halb)
        return self.decoder(z2)


def lap_traces(ses, laps_index) -> tuple[np.ndarray, list]:
    """Rohe Telemetriespuren je Runde, interpoliert auf TRACE_PUNKTE gleich verteilte Distanzpunkte.
    Input für den Autoencoder statt rundenprofile()s aggregierter Kennzahlen.
    Gibt (Spuren, Index) zurück. Index kann kürzer sein als ``laps_index``.
    Nicht für jede Runde ist Telemetrie verfügbar."""
    spuren, index = [], []
    for idx in laps_index:
        try:
            tel = ses.laps.loc[[idx]].get_car_data().add_distance()
        except Exception:
            continue
        if tel is None or len(tel) < 50 or tel["Distance"].max() <= 0:
            continue
        grid = np.linspace(0, tel["Distance"].max(), TRACE_PUNKTE)
        kanaele = [np.interp(grid, tel["Distance"], tel[k].astype(float))
                  for k in TRACE_KANAELE]
        spuren.append(np.stack(kanaele))
        index.append(idx)
    return np.stack(spuren), index


def autoencoder_scores(spuren: np.ndarray, seed: int = 0) -> np.ndarray:
    """trainiert den Autoencoder auf allen ``spuren`` und gibt den Rekonstruktionsfehler je Runde als Anomalie-Score zurück.

    Global je Kanal z-normalisiert, nicht je Fahrer wie bei den Pedal-Features
    oben. Bewusste Vereinfachung: das Netz soll eine normale Runde auf dieser
    Strecke lernen, nicht nur individuelle Abweichungen vom eigenen Mittel.
    """
    torch.manual_seed(seed)
    mean = spuren.mean(axis=(0, 2), keepdims=True)
    std = spuren.std(axis=(0, 2), keepdims=True) + 1e-6
    x = torch.tensor((spuren - mean) / std, dtype=torch.float32)

    modell = LapAutoencoder(n_channels=spuren.shape[1], n_points=spuren.shape[2])
    opt = torch.optim.Adam(modell.parameters(), lr=1e-3, weight_decay=1e-5)
    verlust_fn = nn.MSELoss()

    modell.train()
    for _epoch in range(AE_EPOCHS):
        opt.zero_grad()
        verlust = verlust_fn(modell(x), x)
        verlust.backward()
        opt.step()

    modell.eval()
    with torch.no_grad():
        fehler = ((modell(x) - x) ** 2).mean(dim=(1, 2)).numpy()
    return fehler


def anomalien_autoencoder(sauber: pd.DataFrame, ses,
                          contamination: float = CONTAMINATION) -> pd.DataFrame:
    """Dieselbe 'sauber'-Grundgesamtheit wie ``anomalien_finden()``, aber mit dem Autoencoder-Score statt IsolationForest.
    Liefert dieselben Spalten (u.a. 'anomalie', 'score'). So bleibt ``ausfall_analyse()`` unverändert auf beide Ergebnisse anwendbar."""
    spuren, index = lap_traces(ses, sauber.index)
    fehler = autoencoder_scores(spuren)
    out = sauber.loc[index].copy()
    out["score"] = fehler
    schwelle = np.quantile(fehler, 1 - contamination)
    out["anomalie"] = np.where(fehler >= schwelle, -1, 1)
    return out


def anomalien_finden(df: pd.DataFrame) -> pd.DataFrame:
    """IsolationForest auf sauberen Runden: kein Feld-Ereignis, keine Startrunde, beide strukturell anders und kein Defekt.
    Boxenrunden bleiben bewusst drin."""
    sauber = df[~df["neutralisiert"] & ~df["erste_runde"]].copy()
    z = sauber.groupby("driver")[FEAT].transform(
        lambda s: (s - s.mean()) / (s.std() + 1e-9))
    iso = IsolationForest(contamination=CONTAMINATION, random_state=0)
    sauber["anomalie"] = iso.fit_predict(z)
    sauber["score"] = -iso.score_samples(z)
    return sauber


def mit_race_control_abgleichen(flags: pd.DataFrame, rcm: pd.DataFrame,
                                fenster: int = 2) -> pd.DataFrame:
    """sucht die passende Race-Control-Meldung je Flag: Fahrerkürzel im Text, Rundennummer +/- Fenster."""
    treffer = []
    for r in flags.itertuples():
        nahe = rcm[(rcm["Lap"] >= r.lap - fenster) & (rcm["Lap"] <= r.lap + fenster)]
        passt = nahe[nahe["Message"].str.contains(str(r.driver), na=False)]
        treffer.append(passt.iloc[0]["Message"] if len(passt) else "")
    out = flags.copy()
    out["race_control"] = treffer
    return out


def ausfall_analyse(df: pd.DataFrame, anomalien: pd.DataFrame,
                    ses) -> pd.DataFrame:
    """je Ausfall die letzte(n) Runde(n) und ob vorher eine Anomalie geflaggt wurde.
    Getrennt nach Boxenrunden (fast immer statistisch auffällig durch die
    Boxengasse selbst, unabhängig vom Auto-Zustand) und echten Runden auf der
    Strecke. Nur Letztere zählen als echte Frühwarnung. Sonst würde jeder
    planmäßige Reifenwechsel früh im Rennen fälschlich als "Warnung" durchgehen."""
    ausfaelle = ses.results[ses.results["Status"].isin(["Retired"])]
    zeilen = []
    for r in ausfaelle.itertuples():
        drv = r.Abbreviation
        letzte_runde = df.loc[df["driver"] == drv, "lap"].max()
        eigene = anomalien[anomalien["driver"] == drv].sort_values("lap")
        vor_ausfall = eigene[(eigene["anomalie"] == -1)
                             & (eigene["lap"] < letzte_runde)]
        auf_strecke = vor_ausfall[~vor_ausfall["boxenrunde"]]
        vorlauf = (letzte_runde - auf_strecke["lap"].max()
                  if not auf_strecke.empty else np.nan)
        zeilen.append({
            "driver": drv, "status": r.Status, "letzte_runde": letzte_runde,
            "anomalien_boxenrunden": len(vor_ausfall) - len(auf_strecke),
            "anomalien_auf_strecke": len(auf_strecke),
            "runden_vorlauf_auf_strecke": vorlauf,
        })
    return pd.DataFrame(zeilen)


def zeichne_verlauf(ax, df: pd.DataFrame, anomalien: pd.DataFrame,
                    phasen: pd.DataFrame) -> None:
    # lap_start==0 sind Meldungen vor dem eigentlichen Renn-Beginn
    # (Formationsrunde etc.). Für den Rennverlauf ohne Aussagekraft.
    for p in phasen[(phasen["label"] != "gruen") & (phasen["lap_start"] > 0)].itertuples():
        ax.axvspan(p.lap_start, max(p.lap_end, p.lap_start + 0.3),
                  color=PHASE.get(p.label, MUTED), alpha=0.3, lw=0)
    for _drv, g in anomalien.groupby("driver"):
        g = g.sort_values("lap")
        ax.plot(g["lap"], g["score"], lw=0.7, alpha=0.35, color=MUTED)
    flags = anomalien[anomalien["anomalie"] == -1]
    ax.scatter(flags["lap"], flags["score"], color=SERIEN[1], s=26, zorder=3,
              label="Anomalie (sauber)")
    for lbl in ("gelb", "vsc"):
        if lbl in phasen["label"].values:
            ax.plot([], [], color=PHASE[lbl], lw=8, alpha=0.3, label=lbl)
    ax.set_xlabel("Runde")
    ax.set_ylabel("Anomalie-Score")
    ax.set_title(f"{EVENT} {SEASON} - Anomalie-Score ueber den Rennverlauf",
                loc="left", color=FG, fontsize=12, pad=10)
    ax.legend(loc="upper left", frameon=False, labelcolor=FG, fontsize=9)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(alpha=0.3, linewidth=0.8, color=GRID)
    ax.set_axisbelow(True)


def zeichne_ausfaelle(ax, df: pd.DataFrame, ausfaelle: list[str]) -> None:
    """Rundenzeitverlauf der Ausfaller, letzte Runde markiert. Vier Fahrer > MAX_SERIEN=3 Hausfarben.
    Qualitatives Colormap statt Wiederholung, sonst färben zwei Fahrer identisch wie in P24."""
    farben = plt.get_cmap("tab10")
    for i, drv in enumerate(ausfaelle):
        g = df[df["driver"] == drv].sort_values("lap")
        farbe = farben(i) if len(ausfaelle) > len(SERIEN) else SERIEN[i]
        ax.plot(g["lap"], g["lap_time"], color=farbe, lw=1.6,
               marker=".", ms=4, label=drv)
        letzte = g["lap"].max()
        ax.axvline(letzte, color=farbe, lw=0.8, ls=":")
    ax.set_xlabel("Runde")
    ax.set_ylabel("Rundenzeit [s]")
    ax.set_title("AUSBAUSTUFE: Rundenzeit der vier Ausfaller", loc="left",
                color=FG, fontsize=12, pad=10)
    ax.legend(loc="upper left", frameon=False, labelcolor=FG, fontsize=9)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(alpha=0.3, linewidth=0.8, color=GRID)
    ax.set_axisbelow(True)


def zeichne_score_vergleich(ax, anomalien: pd.DataFrame, anomalien_ae: pd.DataFrame,
                            gemeinsam, korr: float) -> None:
    """IsolationForest- gegen Autoencoder-Score, je Runde ein Punkt. Farbe zeigt welches Verfahren (keins/eins/beide) diese Runde geflaggt hat."""
    iso = anomalien.loc[gemeinsam, "score"]
    ae = anomalien_ae.loc[gemeinsam, "score"]
    iso_flag = anomalien.loc[gemeinsam, "anomalie"] == -1
    ae_flag = anomalien_ae.loc[gemeinsam, "anomalie"] == -1

    gruppen = [
        ("keins", ~iso_flag & ~ae_flag, MUTED, 14),
        ("nur IsolationForest", iso_flag & ~ae_flag, SERIEN[0], 50),
        ("nur Autoencoder", ~iso_flag & ae_flag, SERIEN[1], 50),
        ("beide", iso_flag & ae_flag, SERIEN[2], 90),
    ]
    for label, maske, farbe, groesse in gruppen:
        ax.scatter(iso[maske], ae[maske], s=groesse, color=farbe, alpha=0.75,
                  edgecolors="none", label=f"{label} (n={int(maske.sum())})")
    ax.set_xlabel("IsolationForest-Score (aggregierte Rundenkennzahlen)")
    ax.set_ylabel("Autoencoder-Rekonstruktionsfehler (rohe Telemetriespur)")
    staerke = ("schwacher" if abs(korr) < 0.3 else
              "maessiger" if abs(korr) < 0.6 else "starker")
    ax.set_title(f"ZWEITE AUSBAUSTUFE: Spearman r={korr:.2f} - {staerke} "
                f"Zusammenhang", loc="left", color=FG, fontsize=12, pad=10)
    ax.legend(loc="upper left", frameon=False, labelcolor=FG, fontsize=8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(alpha=0.3, linewidth=0.8, color=GRID)
    ax.set_axisbelow(True)


def main():
    f1lab.enable_cache()

    print(f"[1/4] {EVENT} {SEASON} {IDENT} laden (VORGEHEN 1) ...")
    ses = f1lab.load(SEASON, EVENT, IDENT, telemetry=True, messages=True)
    df = rundenprofile(ses)
    print(f"      {len(df)} Runden gesamt, "
         f"{df['neutralisiert'].sum()} unter Gelb/VSC/SC, "
         f"{df['boxenrunde'].sum()} Boxenrunden (bleiben drin)")

    print("\n[2/4] Isolation Forest auf sauberen Runden (VORGEHEN 2) ...")
    anomalien = anomalien_finden(df)
    print(f"      {len(anomalien)} sauber (ohne Neutralisation/Startrunde), "
         f"{(anomalien['anomalie'] == -1).sum()} geflaggt "
         f"({CONTAMINATION:.0%})")

    print("\n[3/4] Top-Anomalien gegen Race Control abgleichen (VORGEHEN 3) ...")
    top = anomalien[anomalien["anomalie"] == -1].sort_values(
        "score", ascending=False).head(15)
    top = mit_race_control_abgleichen(top, ses.race_control_messages)
    print(top[["driver", "lap", "lap_time", "boxenrunde", "score",
              "race_control"]].round(2).to_string(index=False))

    print("\n[4/5] AUSBAUSTUFE: Ausfallanalyse (IsolationForest) ...")
    ausfall_tab = ausfall_analyse(df, anomalien, ses)
    print(ausfall_tab.to_string(index=False))

    print("\n[5/5] ZWEITE AUSBAUSTUFE: Autoencoder auf rohen Telemetriespuren ...")
    anomalien_ae = anomalien_autoencoder(df[~df["neutralisiert"]
                                            & ~df["erste_runde"]], ses)
    print(f"      {len(anomalien_ae)} Runden mit Telemetriespur, "
         f"{(anomalien_ae['anomalie'] == -1).sum()} geflaggt")

    gemeinsam = anomalien.index.intersection(anomalien_ae.index)
    korr, _p = spearmanr(anomalien.loc[gemeinsam, "score"],
                         anomalien_ae.loc[gemeinsam, "score"])
    iso_flags = set(anomalien.index[anomalien["anomalie"] == -1]) & set(gemeinsam)
    ae_flags = set(anomalien_ae.index[anomalien_ae["anomalie"] == -1]) & set(gemeinsam)
    ueberlappung = len(iso_flags & ae_flags)
    print(f"      Spearman-Korrelation IsolationForest- vs. Autoencoder-Score: "
         f"{korr:.3f}")
    print(f"      Uebereinstimmende Top-Flags: {ueberlappung} von "
         f"{len(iso_flags | ae_flags)} (Vereinigung)")

    ausfall_tab_ae = ausfall_analyse(df, anomalien_ae, ses)
    print("\n      Ausfallanalyse mit Autoencoder-Flags:")
    print(ausfall_tab_ae.to_string(index=False))

    print("\nGrafik ...")
    phasen = f1lab.track_status_phases(ses)
    fig, ax = plt.subplots(2, 1, figsize=(13, 10))
    zeichne_verlauf(ax[0], df, anomalien, phasen)
    zeichne_ausfaelle(ax[1], df, list(ausfall_tab["driver"]))
    fig.suptitle(f"{ses.event['EventName']} {SEASON} - Anomalie-Erkennung",
                x=0.09, ha="left", fontsize=15, color=FG, y=1.0)
    plt.tight_layout()
    path = OUT / "anomalie_erkennung.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"\n      -> {path}")

    print("\nGrafik ZWEITE AUSBAUSTUFE ...")
    fig2, ax2 = plt.subplots(figsize=(8, 7))
    zeichne_score_vergleich(ax2, anomalien, anomalien_ae, gemeinsam, korr)
    plt.tight_layout()
    path2 = OUT / "anomalie_autoencoder_vergleich.png"
    fig2.savefig(path2, dpi=130, bbox_inches="tight")
    plt.close(fig2)
    print(f"      -> {path2}")


if __name__ == "__main__":
    main()
