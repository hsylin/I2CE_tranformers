"""Export a portable measured reference record from a completed SDOT scan."""

from pathlib import Path
import statistics
import subprocess
import sys

from common import digest, freeze_check, load, require, save


def calibrate(root, output, analysis=None, plan=None):
    sys.path.insert(
        0, str(Path(__file__).resolve().parents[2] / "transformer_profiling")
    )
    import roofline_calibration as rc

    require(
        load(root / "execution/complete.json")["status"] == "passed",
        "Scan is incomplete",
    )
    require(load(root / "automated-qa.json")["status"] == "passed", "QA is incomplete")
    require(not (root / "execution/failure.json").exists(), "Failure requires review")
    freeze_check(root)
    analysis = analysis or root / "analysis"
    summary, verification = load(analysis / "summary.json"), load(
        analysis / "verification.json"
    )
    require(
        summary["status"] == verification["status"] == "passed",
        "Analysis is unverified",
    )
    hashes = dict(verification["raw_hashes"])
    for name in (
        "freeze.json",
        "plan.json",
        "registration.json",
        "build/ert_sdot",
        "analysis/summary.json",
        "analysis/rows.json",
        "analysis/verification.json",
        "execution/complete.json",
        "automated-qa.json",
    ):
        hashes[name] = digest(root / name)
    for name, expected in hashes.items():
        require(digest(root / name) == expected, "Evidence changed: " + name)
    plan, rows = plan or load(root / "plan.json"), load(analysis / "rows.json")
    for name in ("summary.json", "rows.json", "verification.json"):
        hashes["derived/" + name] = digest(analysis / name)
    config = load(root / "runs/exp0/config.json")
    frequency = rows[0]["sim_freq"]
    identity = rc.hardware_identity(config, frequency)
    empties = {
        ex: statistics.median(
            r["sim_ticks"] for r in summary["empty_windows"] if r["experiment"] == ex
        )
        for ex in range(3)
    }

    def selected(metric, prefix=None):
        chosen = summary["peaks"][metric]["best_stable"]
        require(chosen is not None, "No stable reference: " + metric)
        block = plan["blocks"][chosen["block"]]
        samples = []
        for r in rows:
            if r["block"] != chosen["block"] or r["passes"] not in block["passes"][-2:]:
                continue
            count = (
                r["integer_ops"]
                if prefix is None
                else r[prefix + "_read_bytes"] + r[prefix + "_write_bytes"]
            )
            samples.append(
                dict(
                    id=r["id"],
                    passes=r["passes"],
                    experiment=r["experiment"],
                    sim_ticks=r["sim_ticks"],
                    count=count,
                    empty_ticks=empties[r["experiment"]],
                    stats_sha256=hashes["runs/exp%d/stats.txt" % r["experiment"]],
                )
            )
        result = dict(
            condition=block,
            passes=block["passes"][-2:],
            samples=samples,
            rate_per_second=chosen[metric + "_median"] * 1e9,
            unit="operations/second" if prefix is None else "bytes/second",
        )
        if prefix is not None:
            result.update(direction="total", source_type="measured")
        rc.stable_rate(result, frequency, summary["quality"], hashes)
        return result

    registration = load(root / "registration.json")
    gem5 = Path(registration["gem5_root"])
    require(
        digest(gem5 / "build/ARM/gem5.fast")
        == registration["runtime_pins"][str(gem5 / "build/ARM/gem5.fast")],
        "gem5 binary differs from the registered binary",
    )
    record = dict(
        schema_version=2,
        id="sdot-full-scan-" + hashes["plan.json"][:12],
        pattern="streaming-int8-sve128-sdot",
        gem5_revision=subprocess.check_output(
            ["git", "-C", str(gem5), "rev-parse", "HEAD"], text=True
        ).strip(),
        hardware=identity,
        hardware_sha256=rc.identity_hash(identity),
        hardware_ini=rc.ini_identity(root / "runs/exp0/config.ini"),
        source_hashes=hashes,
        quality=summary["quality"],
        coverage=dict(conditions=812, measured_ROIs=2436, empty_windows=9),
        compute=selected("integer_GOPs"),
        references={
            level: selected(metric, prefix)
            for level, metric, prefix in [
                ("Core–L1D", "cpu_l1d_GBs", "cpu_l1d"),
                ("L1D–L2", "l1d_l2_GBs", "l1d_l2"),
                ("Physical DRAM", "dram_GBs", "dram"),
            ]
        },
    )
    rc.scan_references(
        record,
        config,
        root / "runs/exp0/config.ini",
        frequency,
        record["gem5_revision"],
    )
    save(output, record)
    return dict(status="passed", path=str(output), sha256=digest(output))
