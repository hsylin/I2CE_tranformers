"""Prepare, gate and collect one registered run without retries."""

import io
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tarfile
import time
import hashlib

from study_records import require, digest, load, save, verify_files
from study_counters import parse


def command(root, registration, row, configuration=False):
    output = Path(row["output"]) / ("configuration" if configuration else "formal")
    inputs = Path(row["inputs"])
    args = [
        registration["simulator"],
        "-d",
        str(output),
        "--stats-file=stats.txt",
        "--dump-config=config.ini",
        "--json-config=config.json",
        str(root / "tools/exp/lib/study_configuration.py"),
        "--study-root=" + str(root),
        "--study-stage=" + row["id"],
        "--study-contract=" + digest(root / "helper-contract.json"),
        "--cpu=minor",
        "--kernel=" + registration["kernel"],
        "--disk-image=" + registration["disk"],
        "--restore=" + registration["checkpoint"],
        "--script=" + str(inputs / "run.rcS"),
        "--vio-9p=" + str(inputs),
        "--cpu-freq=4GHz",
        "--mem-size=2GiB",
        "--mem-type=DDR3_1600_8x8",
        "--mem-channels=1",
        "--num-cores=1",
        "--sve-vl=1",
        "--l1d-size=32KiB",
        "--l2-size=1MiB",
    ]
    if configuration:
        args.append("--configuration-check")
    return args


def environment(registration):
    env = os.environ.copy()
    env.pop("PYTHONOPTIMIZE", None)
    env.update(M5_PATH=registration["gem5_resources"], PYTHONDONTWRITEBYTECODE="1")
    return env


def prepare(row, registration):
    state, inputs = Path(row["state"]), Path(row["inputs"])
    state.mkdir(parents=True, exist_ok=False)
    inputs.mkdir(parents=True, exist_ok=False)
    save(state / "preparation-start.json", dict(stage=row["id"], epoch=time.time()))
    sources = dict(registration["scientific_inputs"])
    sources["transformer.o"] = dict(path=row["binary"], sha256=row["binary_sha256"])
    for relative, source in sources.items():
        require(digest(source["path"]) == source["sha256"], relative)
        target = inputs / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source["path"], target)
        require(digest(target) == source["sha256"], relative)
    (inputs / "weights/multiple_learner_outputs").mkdir(exist_ok=False)
    # The guest uses identical mount points and command lines in every stage.
    # Only the host 9p namespace, marker identifier and bound binary hash vary.
    script = (
        """#!/bin/sh
export PATH=/sbin:/usr/sbin:/bin:/usr/bin
fail() { echo "[guest] STAGE failure $1"; /sbin/m5 fail "$1"; exit "$1"; }
echo "[guest] E01-E13 STAGE once-only formal"
mkdir -p /mnt
mount -t 9p -o trans=virtio,version=9p2000.L,aname=INPUT gem5 /mnt || fail 1
cd /mnt || fail 2
test ! -e gem5_profile_regions.tsv || fail 3
test -d /mnt/weights/generated_from_notebook || fail 4
mkdir -p /home/thu/TiC-SAT/weights || fail 5
mount --bind /mnt/weights /home/thu/TiC-SAT/weights || fail 6
echo "BINARY  transformer.o" | sha256sum -c - || fail 7
./transformer.o
RC=$?
echo "[guest] STAGE exit code: $RC"
test "$RC" -eq 0 || fail "$RC"
cat gem5_profile_regions.tsv || fail 8
/sbin/m5 writefile gem5_profile_regions.tsv gem5_profile_regions.tsv
echo "[guest] STAGE done"
/sbin/m5 exit
""".replace(
            "STAGE", row["id"]
        )
        .replace("INPUT", str(inputs))
        .replace("BINARY", row["binary_sha256"])
    )
    (inputs / "run.rcS").write_text(script)
    subprocess.run(["/bin/sh", "-n", str(inputs / "run.rcS")], check=True)
    files = {str(p): digest(p) for p in inputs.rglob("*") if p.is_file()}
    require(
        len(files) == len(sources) + 1,
        "Study check failed: len(files) == len(sources) + 1",
    )
    save(
        state / "input-binding.json",
        dict(stage=row["id"], files=files, generated_regions_absent=True),
    )


def entry_markers(terminal):
    return {
        text: text in terminal
        for text in [
            "transformer.o: OK",
            "D_SEQ = 512\n",
            "CODEBOOK_SIZE = 4\n",
            "N_LEARNERS = 2\n",
            "SVE_LENGTH on Gem5/QEMU = 16 bytes = 128 bits",
            "[GEM5_PROFILE] reset scope=group2_full_interleaved_transformer_block",
        ]
    }


def normalized_diagnostics(text):
    prefixes = (
        "[bounded-runtime-W]",
        "[static-activation-layout]",
        "[attention-sdot]",
        "[codebook-sdot]",
        "[codebook-last-output]",
        "[s25-tile-family]",
        "[compressed-indices]",
    )
    return [
        re.sub(r"setup_ns=\d+", "setup_ns=POSITIVE", line)
        for line in text.splitlines()
        if line.startswith(prefixes)
    ]


def archive_once(root, row, evidence):
    sources = {}
    for prefix, directory in [
        ("formal", Path(row["output"]) / "formal"),
        ("state", Path(row["state"])),
        ("audit", evidence),
    ]:
        for path in directory.rglob("*"):
            if path.is_file():
                sources[prefix + "/" + str(path.relative_to(directory))] = path
    members = {
        name: dict(sha256=digest(path), bytes=path.stat().st_size)
        for name, path in sources.items()
    }
    data = (json.dumps(members, indent=2) + "\n").encode()
    archive = root / "archives" / (row["id"] + "-raw.tar.gz")
    with (
        archive.open("xb") as stream,
        tarfile.open(fileobj=stream, mode="w:gz", compresslevel=1) as bundle,
    ):
        for name, path in sorted(sources.items()):
            bundle.add(path, arcname=name, recursive=False)
        item = tarfile.TarInfo("manifest.json")
        item.size = len(data)
        item.mode = 0o600
        bundle.addfile(item, io.BytesIO(data))
    with tarfile.open(archive) as bundle:
        require(
            len(bundle.getnames()) == len(members) + 1,
            "Study check failed: len(bundle.getnames()) == len(members)+1",
        )
        require(
            set(bundle.getnames()) == set(members) | {"manifest.json"},
            "Study check failed: set(bundle.getnames()) == set(members)|{'manifest.json'}",
        )
        require(
            bundle.extractfile("manifest.json").read() == data,
            "Study check failed: bundle.extractfile('manifest.json').read() == data",
        )
        for name, metadata in members.items():
            checksum, size = hashlib.sha256(), 0
            with bundle.extractfile(name) as stream:
                for block in iter(lambda: stream.read(1024**2), b""):
                    checksum.update(block)
                    size += len(block)
            require(
                checksum.hexdigest() == metadata["sha256"]
                and size == metadata["bytes"],
                name,
            )
    result = dict(
        passed=True,
        stage=row["id"],
        path=str(archive),
        sha256=digest(archive),
        bytes=archive.stat().st_size,
        members=len(members) + 1,
        manifest_sha256=hashlib.sha256(data).hexdigest(),
    )
    save(root / "evidence" / (row["id"] + "-archive.json"), result)
    return result


def collect(root, registration, row):
    evidence = root / "evidence" / row["id"]
    evidence.mkdir(exist_ok=False)
    save(
        evidence / "start.json",
        dict(
            epoch=time.time(),
            stage=row["id"],
            parser_sha256=digest(Path(__file__).with_name("study_counters.py")),
        ),
    )
    state, output, inputs = (
        Path(row["state"]),
        Path(row["output"]) / "formal",
        Path(row["inputs"]),
    )
    require(
        load(state / "formal-exit.json")["returncode"] == 0,
        "Study check failed: load(state/'formal-exit.json')['returncode'] == 0",
    )
    verify_files(load(state / "prelaunch-gate.json")["pins"])
    terminal = (output / "system.terminal").read_text()
    stdout = (state / "formal-process.log").read_text()
    require(
        all(entry_markers(terminal).values()),
        "Study check failed: all(entry_markers(terminal).values())",
    )
    require(
        "[guest] " + row["id"] + " exit code: 0" in terminal,
        "Study check failed: '[guest] '+row['id']+' exit code: 0' in terminal",
    )
    require(
        "[guest] " + row["id"] + " done" in terminal,
        "Study check failed: '[guest] '+row['id']+' done' in terminal",
    )
    require(
        terminal.count("[GEM5_PROFILE] dump ") == 22,
        "Study check failed: terminal.count('[GEM5_PROFILE] dump ') == 22",
    )
    require(
        "m5_exit instruction encountered (0)" in stdout and "m5_fail" not in stdout,
        "Study check failed: 'm5_exit instruction encountered (0)' in stdout and 'm5_fail' not in stdout",
    )
    gate = load(output / "config-gate.json")
    require(
        gate["passed"] and gate["config_ini_sha256"] == digest(output / "config.ini"),
        "Study check failed: gate['passed'] and gate['config_ini_sha256'] == digest(output/'config.ini')",
    )
    require(
        gate["config_json_sha256"] == digest(output / "config.json"),
        "Study check failed: gate['config_json_sha256'] == digest(output/'config.json')",
    )
    require(
        (inputs / "gem5_profile_regions.tsv").read_bytes()
        == (output / "gem5_profile_regions.tsv").read_bytes()
        == (root / "reference/regions.tsv").read_bytes(),
        "Study check failed: (inputs/'gem5_profile_regions.tsv').read_bytes() == (output/'gem5_profile_regions.tsv').read_bytes() == (root/'reference/regions.tsv').read_bytes()",
    )
    require(
        not any(
            p.is_file()
            for p in (inputs / "weights/multiple_learner_outputs").rglob("*")
        ),
        "Study check failed: not any(p.is_file() for p in (inputs/'weights/multiple_learner_outputs').rglob('*'))",
    )
    require(
        normalized_diagnostics(terminal) == row["validated_diagnostics"],
        "Study check failed: normalized_diagnostics(terminal) == row['validated_diagnostics']",
    )
    setup = [
        line
        for line in terminal.splitlines()
        if line.startswith("[compressed-indices]")
    ]
    if row["id"] >= "E03":
        require(
            len(setup) == 1 and "clock=simulator_rpns" in setup[0],
            "Study check failed: len(setup) == 1 and 'clock=simulator_rpns' in setup[0]",
        )
        require(
            int(re.search(r"setup_ns=(\d+)", setup[0])[1]) > 0,
            "Study check failed: int(re.search(r'setup_ns=(\\d+)', setup[0])[1]) > 0",
        )
    else:
        require(not setup, "Study check failed: not setup")
    parsed = parse(output, evidence)
    save(evidence / "counters.json", parsed)
    save(
        evidence / "completion-audit.json",
        dict(
            passed=True,
            stage=row["id"],
            formal_attempts=1,
            binary_sha256=row["binary_sha256"],
            runtime_unchanged=True,
            diagnostic_match=True,
            setup_outside_ROI=setup,
            raw_dumps=23,
            metrics={
                key: parsed["metrics"][key]
                for key in ["BLOCK", "CODEBOOK", "ATTENTION", "NORMALIZATION_PHASES"]
            },
            counter_fields=len(parsed["metrics"]["BLOCK"]),
            epoch=time.time(),
        ),
    )
    archive = archive_once(root, row, evidence)
    save(
        state / "complete.json",
        dict(
            passed=True,
            stage=row["id"],
            epoch=time.time(),
            archive_sha256=archive["sha256"],
            completion_audit_sha256=digest(evidence / "completion-audit.json"),
        ),
    )
