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


def main():
    f1lab.enable_cache()
    runner = CliRunner()

    print(f"[1/4] f1analyze pace {SEASON} {EVENT} --top 8 ...")
    r = runner.invoke(app, ["pace", str(SEASON), EVENT, "--top", "8"])
    assert r.exit_code == 0, r.output
    print(r.output)

    print(f"[2/4] f1analyze strategy {SEASON} {EVENT} ...")
    r = runner.invoke(app, ["strategy", str(SEASON), EVENT])
    assert r.exit_code == 0, r.output
    print(r.output[:600] + "\n      ... (gekuerzt)")

    print(f"[3/4] f1analyze telemetry {SEASON} {EVENT} VER PER ...")
    r = runner.invoke(app, ["telemetry", str(SEASON), EVENT, "VER", "PER"])
    assert r.exit_code == 0, r.output
    print(r.output)

    out = Path(__file__).parent / "out"
    out.mkdir(exist_ok=True)
    report_pfad = out / f"{EVENT.lower()}_{SEASON}_report.pdf"
    print(f"[4/4] f1analyze report {SEASON} {EVENT} --out {report_pfad} ...")
    r = runner.invoke(app, ["report", str(SEASON), EVENT, "--out", str(report_pfad)])
    assert r.exit_code == 0, r.output
    print(r.output)
    print(f"      PDF-Groesse: {report_pfad.stat().st_size / 1024:.0f} KB")

    print("\nAlle vier Subcommands liefen ueber das installierbare Paket "
         "gegen echte Daten. Details, Architektur, CI: "
         f"{PAKET_ROOT / 'README.md'}")


if __name__ == "__main__":
    main()
