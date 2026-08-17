"""Reproduce tests, analyses, simulations, figures, and environment metadata."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "code"
RESULTS = ROOT / "results"
IMAGES = ROOT / "manuscript" / "images"
MANIFEST = RESULTS / "environment.json"
PACKAGES = ("numpy", "scipy", "openpyxl", "matplotlib")


def relative_command(script: str, *arguments: object) -> list[str]:
    return ["python", f"code/{script}", *(str(value) for value in arguments)]


def run(command: list[str]) -> None:
    executable_command = [sys.executable, *command[1:]]
    print("Running:", " ".join(command), flush=True)
    completed = subprocess.run(
        executable_command,
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode:
        if completed.stdout:
            print(completed.stdout, flush=True)
        if completed.stderr:
            print(completed.stderr, file=sys.stderr, flush=True)
        completed.check_returncode()
    print("Completed:", " ".join(command), flush=True)


def environment_record(commands: list[list[str]], status: str, elapsed: float) -> dict:
    return {
        "status": status,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": elapsed,
        "operating_system": platform.platform(),
        "python": platform.python_version(),
        "python_executable": str(Path(sys.executable).resolve()),
        "packages": {
            package: importlib.metadata.version(package) for package in PACKAGES
        },
        "commands": commands,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reproduce all archived IntervalDRMA outputs."
    )
    parser.add_argument("--simulation-replicates", type=int, default=5000)
    parser.add_argument("--skip-simulation", action="store_true")
    parser.add_argument("--skip-figures", action="store_true")
    args = parser.parse_args()

    commands = [
        relative_command("test_intervaldrma.py"),
        relative_command(
            "run_analysis.py", "--output", "results/analysis_results.json"
        ),
        relative_command(
            "radon_example_analysis.py",
            "--output",
            "results/radon_example_results.json",
        ),
    ]
    if not args.skip_simulation:
        commands.append(
            relative_command(
                "mixed_support_simulation.py",
                "--replicates",
                args.simulation_replicates,
                "--output",
                "results/mixed_support_simulation_results.json",
            )
        )
    if not args.skip_figures:
        commands.extend(
            [
                relative_command("make_score_assumption_figure.py"),
                relative_command("make_distance_density_figure.py"),
                relative_command("make_support_to_design_figure.py"),
                relative_command(
                    "make_support_fragility_figure.py",
                    "--results",
                    "results/analysis_results.json",
                    "--output",
                    "manuscript/images/Support_mapping_fragility.pdf",
                ),
            ]
        )

    RESULTS.mkdir(parents=True, exist_ok=True)
    IMAGES.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    status = "complete"
    try:
        for command in commands:
            run(command)
    except Exception:
        status = "failed"
        raise
    finally:
        elapsed = time.perf_counter() - start
        MANIFEST.write_text(
            json.dumps(environment_record(commands, status, elapsed), indent=2),
            encoding="utf-8",
        )
        print(f"Wrote {MANIFEST.relative_to(ROOT)}", flush=True)


if __name__ == "__main__":
    main()
