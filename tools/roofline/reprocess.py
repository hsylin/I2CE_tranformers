"""Recompute numerical results without changing completed scan evidence."""

import re
from common import digest, freeze_check, load, require, save
from plan import make_plan


def validate_plan(recorded, expected):
    """Match ordered cases and block boundaries, not just total counts."""
    case_fields = ("id", "block", "kind", "n", "footprint", "d", "u", "passes")
    block_fields = ("block", "first_id", "kind", "n", "footprint", "d", "u", "passes")
    for name, fields in [("cases", case_fields), ("blocks", block_fields)]:
        require(
            [{k: x[k] for k in fields} for x in recorded[name]]
            == [{k: x[k] for k in fields} for x in expected[name]],
            "Recorded " + name + " do not match the full scan",
        )
    require(
        recorded["experiments"] == expected["experiments"] == 3,
        "Recorded repetitions differ",
    )


def reprocess(root, output, expected_freeze):
    require(
        digest(root / "freeze.json") == expected_freeze,
        "Recorded freeze identity differs",
    )
    freeze_check(root)
    require(
        load(root / "execution/complete.json")["status"] == "passed",
        "Incomplete source scan",
    )
    require(
        load(root / "automated-qa.json")["status"] == "passed", "Incomplete source QA"
    )
    require(
        not (root / "execution/failure.json").exists(), "Source failure requires review"
    )
    original = load(root / "analysis/verification.json")
    require(original["status"] == "passed", "Source analysis is unverified")
    for name, expected in original["raw_hashes"].items():
        require(digest(root / name) == expected, "Recorded evidence changed: " + name)
    plan = make_plan()
    validate_plan(load(root / "plan.json"), plan)
    verification = (root / "runs/verify/system.terminal").read_text()
    prefixes = re.findall(
        r"^([A-Z][A-Z0-9]*)_VERIFY_PASS cases=144$", verification, re.M
    )
    require(len(prefixes) == 1, "Unrecognized recorded terminal protocol")
    output.mkdir(parents=True, exist_ok=False)
    save(output / "plan.json", plan)
    save(
        output / "start.json",
        dict(
            source_freeze_sha256=expected_freeze,
            source_plan_sha256=digest(root / "plan.json"),
            source_binary_sha256=digest(root / "build/ert_sdot"),
            operation="Reanalyze completed evidence; no simulation and no source mutation",
        ),
    )
    from analyze import analyze
    from calibrate import calibrate
    from scan_results import write_results

    analysis = output / "analysis"
    analyze(root, analysis, plan=plan, marker_prefix=prefixes[0])
    # Compare every pre-existing quantity and gate against the recorded analysis.
    comparisons = {}
    for filename in ("rows.json", "groups.json", "largest-pass.json"):
        before, after = load(root / "analysis" / filename), load(analysis / filename)
        require(len(before) == len(after), "Changed analysis coverage")
        for a, b in zip(before, after):
            for key, value in a.items():
                require(
                    key in b and b[key] == value,
                    "Changed observation: " + filename + ":" + key,
                )
        comparisons[filename] = len(before)
    save(
        output / "observation-preservation.json",
        dict(
            status="passed",
            compared_records=comparisons,
            recorded_quantities_and_stability_flags_unchanged=True,
        ),
    )
    calibrate(root, output / "calibration.json", analysis=analysis, plan=plan)
    write_results(
        root,
        output / "results",
        analysis=analysis,
        plan=plan,
        qa_output=output / "automated-qa.json",
    )
    freeze_check(root)
    save(
        output / "complete.json",
        dict(
            status="passed",
            source_freeze_sha256=expected_freeze,
            calibration_sha256=digest(output / "calibration.json"),
            simulation_performed=False,
        ),
    )
    return dict(status="passed", output=str(output))
