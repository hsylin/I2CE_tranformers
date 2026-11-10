"""Build and register a fresh scan against an explicitly pinned reference machine."""

import os
import re
import shutil
import struct
import subprocess
import time

from common import digest, load, require, save, freeze_check
from plan import SOURCE, emit, make_plan


def command(root, name, argv):
    start = time.time()
    with (root / "build" / (name + ".log")).open("x") as log:
        result = subprocess.run(
            list(map(str, argv)), cwd="/tmp", stdout=log, stderr=subprocess.STDOUT
        )
    save(
        root / "build" / (name + ".json"),
        dict(
            command=list(map(str, argv)),
            returncode=result.returncode,
            elapsed_seconds=time.time() - start,
        ),
    )
    require(result.returncode == 0, "Build/validation failed: " + name)


def prepare(args):
    require(
        os.uname().sysname == "Linux",
        "Build and validation must run on the simulation server",
    )
    root, baseline = args.output.resolve(), args.machine_root.resolve()
    require(
        digest(baseline / "freeze.json") == args.machine_freeze_sha256,
        "Reference machine package identity differs",
    )
    freeze_check(baseline)
    require(
        load(baseline / "execution/complete.json")["status"] == "passed",
        "Reference package is not complete",
    )
    root.mkdir(parents=True, exist_ok=False)
    for name in (
        "src",
        "control",
        "build",
        "inputs",
        "runs",
        "execution",
        "provenance",
    ):
        (root / name).mkdir()
    # Copy the entire tool before preparing immutable run inputs. Active logs
    # and eventual outputs are never added to the frozen source manifest.
    for path in SOURCE.glob("*.py"):
        shutil.copyfile(path, root / "control" / path.name)
    shutil.copytree(SOURCE / "data", root / "control/data")
    shutil.copyfile(SOURCE / "driver_sdot.cpp", root / "src/driver_sdot.cpp")
    shutil.copyfile(baseline / "reference.ini", root / "reference.ini")
    plan = make_plan()
    emit(root, plan)
    # Check all original function bodies, not merely instruction counts.
    old = (baseline / "src/kernel_sdot.S").read_text()
    new = (root / "src/kernel_sdot.S").read_text()
    for entry in plan["kernels"]:
        name = entry["name"]
        pattern = rf"{name}:\n.*?\.size {name},\.-{name}"
        require(
            re.search(pattern, old, re.S).group()
            == re.search(pattern, new, re.S).group(),
            "Streaming kernel changed: " + name,
        )
    save(
        root / "provenance/reference.json",
        dict(
            freeze_sha256=args.machine_freeze_sha256,
            plan_sha256=digest(baseline / "plan.json"),
            reference_ini_sha256=digest(root / "reference.ini"),
            kernel_bodies_identical=True,
            historical_results_inherited=False,
        ),
    )
    original = load(baseline / "plan.json")
    require(
        original["working_sets"] == plan["source_worksets"]["tiles"]
        and digest(baseline / "plan.json")
        == plan["source_worksets"]["source_plan_sha256"],
        "Working sets do not match the frozen source plan",
    )
    host, arm = args.host_cxx.resolve(), args.arm_cxx.resolve()
    command(
        root,
        "host-build",
        [
            host,
            "-std=c++17",
            "-O2",
            "-g",
            "-DHOST_ORACLE",
            "-fsanitize=address,undefined",
            "-fno-omit-frame-pointer",
            root / "src/driver_sdot.cpp",
            "-o",
            root / "build/oracle",
        ],
    )
    command(root, "host-verify", [root / "build/oracle", "--verify"])
    require(
        "SDOT_VERIFY_PASS cases=144" in (root / "build/host-verify.log").read_text(),
        "Host functional coverage incomplete",
    )
    command(root, "host-goldens", [root / "build/oracle", "--golden"])
    command(
        root,
        "arm-build",
        [
            arm,
            "-std=c++17",
            "-O3",
            "-static",
            "-march=armv8.2-a+sve",
            "-msve-vector-bits=128",
            "-I" + str(args.m5_include.resolve()),
            root / "src/driver_sdot.cpp",
            root / "src/kernel_sdot.S",
            args.libm5.resolve(),
            "-o",
            root / "build/ert_sdot",
        ],
    )
    assembly = subprocess.check_output(
        [str(args.objdump.resolve()), "-d", str(root / "build/ert_sdot")], text=True
    )
    (root / "build/disassembly.txt").write_text(assembly)
    audits = []
    for e in plan["kernels"]:
        body = re.search(r"<" + e["name"] + r">:\n(.*?)(?=\n\n)", assembly, re.S).group(
            1
        )
        counts = {
            op: len(re.findall(r"\b" + op + r"\b", body))
            for op in ("sdot", "ld1b", "ld1w", "st1w")
        }
        require(
            counts
            == dict(sdot=e["d"] * e["u"], ld1b=2 * e["u"], ld1w=e["u"], st1w=e["u"]),
            "Unexpected generated instructions: " + e["name"],
        )
        require(
            not re.search(r"\bz(?:8|9|1[0-5])\.|\b(?:sp|x(?:19|2[0-9]))\b", body),
            "Stack or preserved-register use in kernel",
        )
        audits.append(dict(e, counts=counts, no_stack_spills=True))
    save(root / "build/assembly-audit.json", dict(status="passed", functions=audits))
    goldens = []
    for block in plan["blocks"]:
        total = sum(block["passes"]) + len(block["passes"])
        h = 14695981039346656037
        # Independent scalar arithmetic, little-endian byte hashing.
        for i in range(4 * block["n"]):
            dot = sum(
                (((4 * i + q) * 17 + 23) % 256 - 128)
                * (((4 * i + q) * 29 + 71) % 256 - 128)
                for q in range(4)
            )
            value = (
                0xF0000000 + 2654435761 * i + dot * block["d"] * total
            ) & 0xFFFFFFFF
            for byte in struct.pack("<I", value):
                h = ((h ^ byte) * 1099511628211) & 0xFFFFFFFFFFFFFFFF
        goldens.append(
            dict(block=block["block"], hash=f"{h:016x}", cumulative_passes=total)
        )
    cpp = {
        int(b): h
        for b, h in re.findall(
            r"SDOT_BLOCK experiment=-1 block=(\d+) hash=([0-9a-f]+) valid=1",
            (root / "build/host-goldens.log").read_text(),
        )
    }
    require(
        len(cpp) == 137 and all(cpp[g["block"]] == g["hash"] for g in goldens),
        "Independent golden hashes disagree",
    )
    save(root / "build/independent-goldens.json", dict(status="passed", blocks=goldens))
    save(
        root / "build/complete.json",
        dict(
            status="passed",
            binary_sha256=digest(root / "build/ert_sdot"),
            compiler_sha256={
                str(p): digest(p) for p in (host, arm, args.objdump.resolve())
            },
            compiler_identity=subprocess.check_output(
                [str(arm), "--version"], text=True
            ),
            sanitizer="address,undefined",
            functional_cases=144,
            golden_blocks=137,
        ),
    )
    registration = load(baseline / "registration.json")
    registration.pop("recovery_provenance", None)
    registration.pop("sdot_provenance", None)
    registration["created_at"] = time.time()
    registration["scope"] = "Full SDOT bandwidth and compute scan; 3*812 ROIs"
    registration["reference_ini"] = str(root / "reference.ini")
    registration["runtime_pins"] = {
        p: h
        for p, h in registration["runtime_pins"].items()
        if not p.startswith(str(baseline) + "/")
    }
    for name in [
        "reference.ini",
        "control/gem5_configuration.py",
        "plan.json",
        "build/complete.json",
    ]:
        registration["runtime_pins"][str(root / name)] = digest(root / name)
    for tag in ("verify", "exp0", "exp1", "exp2"):
        target = root / "inputs" / tag
        target.mkdir()
        shutil.copyfile(root / "build/ert_sdot", target / "ert_sdot")
        (target / "ert_sdot").chmod(0o755)
        invocation = (
            "./ert_sdot --verify"
            if tag == "verify"
            else "./ert_sdot --experiment " + tag[-1]
        )
        # Paths are quoted for the guest shell, independently from argv handling.
        import shlex

        script = (
            "#!/bin/sh\nexport PATH=/sbin:/usr/sbin:/bin:/usr/bin\n"
            'fail() { echo SDOT_GUEST_FAILED "$1"; /sbin/m5 fail "$1"; exit "$1"; }\n'
            "mkdir -p /mnt\nmount -t 9p -o "
            + shlex.quote("trans=virtio,version=9p2000.L,aname=" + str(target))
            + " gem5 /mnt || fail 1\n"
            "cd /mnt || fail 2\necho "
            + shlex.quote(digest(target / "ert_sdot") + "  ert_sdot")
            + " | sha256sum -c - || fail 3\n"
            + invocation
            + " || fail 4\necho SDOT_GUEST_COMPLETE\n/sbin/m5 exit\n"
        )
        (target / "run.rcS").write_text(script)
        registration["runs"][tag].update(
            inputs=str(target),
            max_ticks=2000000000000 if tag == "verify" else 120000000000000,
            input_pins={
                str(target / name): digest(target / name)
                for name in ("ert_sdot", "run.rcS")
            },
        )
    save(root / "registration.json", registration)
    commands = load(baseline / "commands.json")
    for tag, argv in commands.items():
        commands[tag] = [
            v.replace(str(baseline), str(root)).replace(
                digest(baseline / "registration.json"),
                digest(root / "registration.json"),
            )
            for v in argv
        ]
    save(root / "commands.json", commands)
    save(
        root / "runtime.json",
        dict(m5_path=str(args.m5_path.resolve())),
    )
    files = {
        str(p.relative_to(root)): digest(p)
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }
    save(
        root / "freeze.json",
        dict(
            files=files,
            modes={
                f"inputs/{tag}/ert_sdot": 0o755
                for tag in ("verify", "exp0", "exp1", "exp2")
            },
        ),
    )
    return dict(
        status="prepared",
        root=str(root),
        freeze_sha256=digest(root / "freeze.json"),
        binary_sha256=digest(root / "build/ert_sdot"),
        conditions=812,
    )
