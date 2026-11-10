"""Export verified numerical scan results without a visualization dependency."""

import shutil

from common import digest, load, require, save


def write_results(root, output, analysis=None, plan=None, qa_output=None):
    analysis = analysis or root / "analysis"
    summary = load(analysis / "summary.json")
    verification = load(analysis / "verification.json")
    require(summary["status"] == verification["status"] == "passed", "Unverified scan")
    for name, expected in verification["raw_hashes"].items():
        require(digest(root / name) == expected, "Raw evidence changed: " + name)
    groups = load(analysis / "groups.json")
    largest = load(analysis / "largest-pass.json")
    require(
        summary["rows"] == 2436 and len(groups) == 812 and len(largest) == 137,
        "Incomplete scan coverage",
    )
    output.mkdir(parents=True, exist_ok=False)
    save(
        output / "sdot-scan-results.json",
        dict(
            plan=plan or load(root / "plan.json"),
            summary=summary,
            verification=verification,
            groups=groups,
            largest_pass=largest,
        ),
    )
    for name in ("rows", "groups", "largest-pass"):
        shutil.copyfile(
            analysis / (name + ".csv"), output / ("sdot-scan-" + name + ".csv")
        )
    hashes = {p.name: digest(p) for p in sorted(output.iterdir())}
    save(
        qa_output or root / "automated-qa.json",
        dict(
            status="passed",
            rows=summary["rows"],
            conditions=len(groups),
            largest_pass=len(largest),
            result_hashes=hashes,
        ),
    )
    return dict(status="passed", output=str(output))
