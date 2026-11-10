#!/usr/bin/env python3
"""Prepare, execute and analyze a reproducible SDOT Roofline scan."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    sub.add_parser("plan", help="Print the exact full scan without running it")
    prepare = sub.add_parser(
        "prepare", help="Create a new directory; build, validate and freeze inputs"
    )
    for name in (
        "output",
        "machine-root",
        "host-cxx",
        "arm-cxx",
        "objdump",
        "m5-include",
        "libm5",
        "m5-path",
    ):
        prepare.add_argument("--" + name, type=Path, required=True)
    prepare.add_argument("--machine-freeze-sha256", required=True)
    for name in ("run", "analyze", "results", "export", "calibrate"):
        command = sub.add_parser(name)
        command.add_argument("--root", type=Path, required=True)
        if name == "run":
            command.add_argument("--workers", type=int, default=3)
        if name == "calibrate":
            command.add_argument("--output", type=Path, required=True)
    replay = sub.add_parser(
        "reprocess",
        help="Read a completed full scan and write verified numerical results",
    )
    replay.add_argument("--root", type=Path, required=True)
    replay.add_argument("--output", type=Path, required=True)
    replay.add_argument("--freeze-sha256", required=True)
    args = parser.parse_args()
    if args.action == "plan":
        from plan import make_plan

        result = make_plan()
    elif args.action == "reprocess":
        from reprocess import reprocess

        result = reprocess(
            args.root.resolve(), args.output.resolve(), args.freeze_sha256
        )
    elif args.action == "prepare":
        from build import prepare

        result = prepare(args)
    elif args.action == "run":
        from run import run

        result = run(args.root.resolve(), args.workers)
    elif args.action == "analyze":
        from analyze import analyze

        result = analyze(args.root.resolve(), args.root.resolve() / "analysis")
    elif args.action == "results":
        from scan_results import write_results

        result = write_results(args.root.resolve(), args.root.resolve() / "results")
    elif args.action == "calibrate":
        from calibrate import calibrate

        result = calibrate(args.root.resolve(), args.output.resolve())
    else:
        from export import export

        result = export(args.root.resolve())
    print(json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()
