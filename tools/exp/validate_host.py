#!/usr/bin/env python3
"""Run host checks without building ARM code or launching a simulator."""

import argparse
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]


def run(command):
    subprocess.run(
        command,
        cwd=ROOT,
        check=True,
        timeout=300,
        env=dict(
            os.environ,
            PYTHONDONTWRITEBYTECODE="1",
            PATH=str(Path(sys.executable).parent)
            + os.pathsep
            + os.environ.get("PATH", ""),
        ),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--lint", action="store_true", help="Also run format and static checks"
    )
    args = parser.parse_args()
    run(["bash", "-n", "tools/exp/exp.sh"])
    run(
        [
            sys.executable,
            "-B",
            "-m",
            "unittest",
            "discover",
            "-s",
            "tests",
            "-p",
            "*_test.py",
        ]
    )
    run([sys.executable, "-B", "tools/analysis/test_check_spills.py"])
    if args.lint:
        sources = [
            "tools",
            "tests",
            "transformer_profiling/add_experiment.py",
            "transformer_profiling/report.py",
            "transformer_profiling/cache_comparison.py",
        ]
        run([sys.executable, "-m", "black", "--workers", "1", "--check", *sources])
        run([sys.executable, "-m", "ruff", "check", *sources])
        run(
            [
                "shellcheck",
                "compile_transformer.sh",
                "tools/exp/exp.sh",
                "tools/exp/lib/boot_ckpt.rcS",
                "tools/exp/lib/run.rcS.tmpl",
                "tools/exp/lib/smoke.rcS.tmpl",
                "tests/run_gemm_widening_test.sh",
            ]
        )


if __name__ == "__main__":
    main()
