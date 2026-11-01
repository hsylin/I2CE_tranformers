"""Export a completed registered run for the offline Roofline report."""

from pathlib import Path
import shutil
import sys

from study_records import digest, load, require, save, verify_files


def collect(root, registration, row):
    """Run only after ordinary completion gates; never launch a measurement."""
    policy = registration["roofline"]
    metadata = row["roofline"]
    formal = Path(row["output"]) / "formal"
    state = Path(row["state"])
    gate = load(formal / "config-gate.json")
    require(
        gate["passed"]
        and gate["config_json_sha256"] == digest(formal / "config.json")
        and gate["config_ini_sha256"] == digest(formal / "config.ini"),
        "Completed configurations differ from the hardware gate",
    )
    require(digest(row["binary"]) == row["binary_sha256"], "Binary identity changed")
    build_record = row["build_record"]
    require(
        digest(build_record["path"]) == build_record["sha256"], "Build record changed"
    )
    build = load(build_record["path"])
    require(
        build["source_commit"] == metadata["source_commit"]
        and build["binary_sha256"] == row["binary_sha256"]
        and build["commands"]
        and build["compiler_identity"],
        "Build/source/binary identity is incomplete or inconsistent",
    )
    verify_files(registration["runtime_pins"])
    verify_files(registration["checkpoint_pins"])
    require(
        load(state / "formal-exit.json")["returncode"] == 0,
        "Host process did not complete successfully",
    )
    inputs = {}
    for name, record in registration["scientific_inputs"].items():
        require(digest(record["path"]) == record["sha256"], "Input changed: " + name)
        inputs[name] = record["sha256"]
    terminal = (formal / "system.terminal").read_text()
    stdout = (state / "formal-process.log").read_text()
    require(
        "transformer.o: OK" in terminal
        and "[guest] " + row["id"] + " exit code: 0" in terminal
        and "[guest] " + row["id"] + " done" in terminal
        and "m5_exit instruction encountered (0)" in stdout
        and "m5_fail" not in stdout,
        "Guest completion/identity evidence is incomplete",
    )
    output = root / "evidence" / row["id"] / "roofline"
    output.mkdir(exist_ok=False)
    # A failed export remains available for inspection; it is never overwritten.
    save(output / "inputs.json", inputs)
    save(
        output / "measurement.json",
        dict(
            gem5_revision=policy["gem5_revision"],
            roi=metadata["roi"],
            operation_model=metadata["operation_model"],
            monitors="native-data-boundaries",
            runtime_sha256=sorted(registration["runtime_pins"].values()),
            checkpoint_sha256=sorted(registration["checkpoint_pins"].values()),
        ),
    )
    artifacts = {}
    sources = dict(
        stats=formal / "stats.txt",
        config=formal / "config.json",
        config_ini=formal / "config.ini",
        regions=formal / "gem5_profile_regions.tsv",
        terminal=formal / "system.terminal",
        process_log=state / "formal-process.log",
        host_exit=state / "formal-exit.json",
        config_gate=formal / "config-gate.json",
        build=Path(build_record["path"]),
    )
    for name, source in sources.items():
        target = output / (name + source.suffix)
        with source.open("rb") as src, target.open("xb") as dst:
            shutil.copyfileobj(src, dst)
        require(digest(source) == digest(target), "Evidence changed during copy")
        artifacts[name] = dict(path=target.name, sha256=digest(target))
    for name in ("inputs", "measurement"):
        artifacts[name] = dict(
            path=name + ".json", sha256=digest(output / (name + ".json"))
        )
    receipt = dict(
        source_commit=metadata["source_commit"],
        binary_sha256=row["binary_sha256"],
        inputs_sha256=artifacts["inputs"]["sha256"],
        measurement_sha256=artifacts["measurement"]["sha256"],
        artifact_sha256={key: value["sha256"] for key, value in artifacts.items()},
        host_returncode=0,
        guest_returncode=0,
        simulation_code=0,
        simulation_cause="m5_exit instruction encountered",
        config_gate_passed=True,
        guest_binary_hash_ok=True,
    )
    save(output / "receipt.json", receipt)
    artifacts["receipt"] = dict(
        path="receipt.json", sha256=digest(output / "receipt.json")
    )
    run = dict(
        id=policy["study_id"] + "/" + row["id"],
        label=metadata["label"],
        status=metadata["status"],
        gem5_revision=policy["gem5_revision"],
        source_commit=metadata["source_commit"],
        binary_sha256=row["binary_sha256"],
        inputs_sha256=receipt["inputs_sha256"],
        measurement_sha256=receipt["measurement_sha256"],
        roi=metadata["roi"],
        operation_model=metadata["operation_model"],
        limitations=metadata["limitations"],
        artifacts=artifacts,
    )
    manifest = dict(
        schema_version=2, method="gem5_measured_cache_references", experiments=[run]
    )
    if "calibration" in policy:
        calibration = policy["calibration"]
        require(
            digest(calibration["path"]) == calibration["sha256"], "Calibration changed"
        )
        with (
            Path(calibration["path"]).open("rb") as src,
            (output / "calibration.json").open("xb") as dst,
        ):
            shutil.copyfileobj(src, dst)
        manifest["calibration"] = dict(
            path="calibration.json", sha256=calibration["sha256"]
        )
    save(output / "roofline.json", manifest)
    sys.path.insert(
        0, str(Path(__file__).resolve().parents[3] / "transformer_profiling")
    )
    import roofline

    # Preserve unsupported status alongside ordinary profiling; never invent bytes.
    result = roofline.load(output / "roofline.json")
    save(
        output / "collection.json",
        dict(
            manifest_sha256=digest(output / "roofline.json"),
            observations=len(result["observations"]),
            skipped=result["skipped"],
            reference_issues=[
                issue for o in result["observations"] for issue in o["reference_issues"]
            ],
        ),
    )
