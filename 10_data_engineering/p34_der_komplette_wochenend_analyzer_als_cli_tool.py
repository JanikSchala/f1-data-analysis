"""fuehrt das installierte f1analyze-CLI end-to-end gegen echte, gecachte Daten aus."""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PAKET_ROOT = Path(__file__).parent / "f1analyze"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(PAKET_ROOT))

warnings.filterwarnings("ignore")

from f1analyze.cli import app  # noqa: E402
from typer.testing import CliRunner  # noqa: E402

import f1lab  # noqa: E402

SEASON, EVENT = 2024, "Bahrain"


def _lauf(runner, schritt: str, argv: list[str], kuerzen: int = 0) -> bool:
    """einen Subcommand ausfuehren und das Ergebnis melden.

    Frueher stand hier ein ``assert r.exit_code == 0``. Das beendet den
    Demolauf beim ersten Subcommand, der nicht durchgeht - auch wenn er
    aus einem guten Grund nicht durchgeht. Mit ``EVENT = "Monaco"``
    liefert ``telemetry ... VER PER`` korrekt einen Fehler, weil Perez
    dort nach der Startkollision nur eine einzige Runde hat und damit
    keine gewertete Bestzeit; der Report danach haette trotzdem laufen
    koennen, kam aber nie dran.

    Jetzt laufen alle vier durch, und am Ende sagt main(), welche
    gescheitert sind - als Aussage mit Exitcode, nicht als Traceback.
    """
    print(f"{schritt} f1analyze {' '.join(argv)} ...")
    r = runner.invoke(app, argv)
    if r.exit_code != 0:
        erste = (r.output.strip().splitlines() or ["ohne Ausgabe"])[0]
        print(f"      Exitcode {r.exit_code}: {erste}")
        return False
    if kuerzen:
        print(r.output[:kuerzen] + "\n      ... (gekuerzt)")
    else:
        print(r.output)
    return True


def main():
    f1lab.enable_cache()
    runner = CliRunner()
    gescheitert = []

    if not _lauf(runner, "[1/4]", ["pace", str(SEASON), EVENT, "--top", "8"]):
        gescheitert.append("pace")
    if not _lauf(runner, "[2/4]", ["strategy", str(SEASON), EVENT],
                 kuerzen=600):
        gescheitert.append("strategy")
    if not _lauf(runner, "[3/4]",
                 ["telemetry", str(SEASON), EVENT, "VER", "PER"]):
        gescheitert.append("telemetry")

    out = Path(__file__).parent / "out"
    out.mkdir(exist_ok=True)
    report_pfad = out / f"{EVENT.lower()}_{SEASON}_report.pdf"
    if _lauf(runner, "[4/4]", ["report", str(SEASON), EVENT,
                               "--out", str(report_pfad)]):
        print(f"      PDF-Groesse: {report_pfad.stat().st_size / 1024:.0f} KB")
    else:
        gescheitert.append("report")

    if gescheitert:
        raise SystemExit(
            f"\n{len(gescheitert)} von 4 Subcommands liefen nicht durch: "
            f"{', '.join(gescheitert)} (Grund siehe oben)")

    print("\nAlle vier Subcommands liefen ueber das installierbare Paket "
         "gegen echte Daten. Details, Architektur, CI: "
         f"{PAKET_ROOT / 'README.md'}")


if __name__ == "__main__":
    main()
