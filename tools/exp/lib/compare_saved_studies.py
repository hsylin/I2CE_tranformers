"""Compare bound saved observations without pooling or launching measurements."""

import argparse
import hashlib
import json
from pathlib import Path


PIN_FIELDS = ("scientific_inputs", "runtime_pins", "checkpoint_pins")
PHASES = ("BLOCK", "CODEBOOK", "ATTENTION", "NORMALIZATION_PHASES")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_study(registration_path, summary_path, rows_path):
    registration = json.loads(registration_path.read_text())
    summary = json.loads(summary_path.read_text())
    registered = {row["id"]: row for row in registration["rows"]}
    observations = {}
    for row in summary["rows"]:
        stage = row["stage"]
        if stage not in registered or row["status"] != "valid-single-run":
            raise ValueError("Unknown or incomplete saved stage")
        path = rows_path / stage / "audit/counters.json"
        binding = summary["bindings"][stage]
        if digest(path) != binding["counters_sha256"]:
            raise ValueError("Saved counter hash differs")
        counters = json.loads(path.read_text())
        if not counters["passed"]:
            raise ValueError("Saved counter audit did not pass")
        if row["whole_ticks"] != counters["metrics"]["BLOCK"]["sim_ticks"]:
            raise ValueError("Summary and counter time differ")
        for field in ("binary_sha256", "source_commit", "tag_object"):
            if field in binding and binding[field] != registered[stage][field]:
                raise ValueError("Summary and registration identity differ")
        observations[stage] = dict(
            identity={
                key: registered[stage][key]
                for key in (
                    "run_id",
                    "tag",
                    "tag_object",
                    "source_commit",
                    "binary_sha256",
                )
            },
            metrics=counters["metrics"],
            frequency=counters["sim_frequency"],
            counter_sha256=binding["counters_sha256"],
        )
    if set(observations) != set(registered):
        raise ValueError("Registered stages are missing")
    return dict(
        study=registration["study"],
        registration_sha256=digest(registration_path),
        summary_sha256=digest(summary_path),
        pins={field: registration[field] for field in PIN_FIELDS},
        observations=observations,
    )


def compare(reference, current):
    if reference["pins"] != current["pins"]:
        raise ValueError("Scientific inputs or runtime/checkpoint pins differ")
    if set(reference["observations"]) != set(current["observations"]):
        raise ValueError("Stage sets differ")
    rows = []
    for stage in sorted(reference["observations"]):
        left = reference["observations"][stage]
        right = current["observations"][stage]
        phases = {}
        for phase in PHASES:
            a, b = left["metrics"][phase], right["metrics"][phase]
            if set(a) != set(b):
                raise ValueError("Counter fields differ")
            phases[phase] = {
                field: (
                    None
                    if a[field] is None or b[field] is None
                    else b[field] - a[field]
                )
                for field in a
            }
        ticks = phases["BLOCK"]["sim_ticks"]
        frequency = left["frequency"]
        if frequency <= 0 or right["frequency"] != frequency:
            raise ValueError("Simulation frequencies differ or are invalid")
        rows.append(
            dict(
                stage=stage,
                identical_binary=left["identity"]["binary_sha256"]
                == right["identity"]["binary_sha256"],
                reference=left["identity"],
                current=right["identity"],
                reference_counter_sha256=left["counter_sha256"],
                current_counter_sha256=right["counter_sha256"],
                delta_milliseconds=ticks * 1000 / frequency,
                delta_percent=100 * ticks / left["metrics"]["BLOCK"]["sim_ticks"],
                aggregate_counter_deltas=phases,
            )
        )
    return dict(
        schema_version=1,
        reference_study=reference["study"],
        current_study=current["study"],
        source_bindings={
            name: {key: study[key] for key in ("registration_sha256", "summary_sha256")}
            for name, study in (("reference", reference), ("current", current))
        },
        matching_registered_pins=True,
        rows=rows,
        identical_binary_stages=[
            row["stage"] for row in rows if row["identical_binary"]
        ],
        interpretation="Paired observations from separate studies; no pooling, calibrated uncertainty, noise floor, confidence interval or equivalence threshold.",
        new_measurements_launched=0,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for prefix in ("reference", "current"):
        for field in ("registration", "summary", "rows"):
            parser.add_argument("--" + prefix + "-" + field, type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    studies = [
        load_study(
            *(
                getattr(args, prefix + "_" + field)
                for field in ("registration", "summary", "rows")
            )
        )
        for prefix in ("reference", "current")
    ]
    result = compare(*studies)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
