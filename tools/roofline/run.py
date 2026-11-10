"""One-shot owner: configuration, ARM checks, three independent scans, analysis."""

import concurrent.futures
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

from common import digest, freeze_check, load, require, save


def resources(root):
    memory = (
        int(
            next(
                line.split()[1]
                for line in Path("/proc/meminfo").read_text().splitlines()
                if line.startswith("MemAvailable:")
            )
        )
        * 1024
    )
    disk = shutil.disk_usage(root).free
    require(
        memory >= 64 * 1024**3 and disk >= 100 * 1024**3, "Resource admission failed"
    )
    return dict(
        available_memory_bytes=memory, free_disk_bytes=disk, cpu_count=os.cpu_count()
    )


def run(root, workers):
    require(os.uname().sysname == "Linux", "Simulation must run on the server")
    require(1 <= workers <= 3, "At most three independent scan repetitions exist")
    execution = root / "execution"
    # Exclusive claim occurs before process creation. Failure is retained, never retried.
    save(
        execution / "start.json",
        dict(epoch=time.time(), pid=os.getpid(), workers=workers),
    )
    try:
        os.nice(19)
        os.sched_setscheduler(0, os.SCHED_IDLE, os.sched_param(0))
        frozen = freeze_check(root)
        save(
            execution / "executable-mode-gate.json",
            dict(status="passed", modes=frozen["modes"]),
        )
        runtime = load(root / "runtime.json")
        env = dict(
            os.environ,
            M5_PATH=runtime["m5_path"],
            PYTHONDONTWRITEBYTECODE="1",
            OMP_NUM_THREADS="1",
            OPENBLAS_NUM_THREADS="1",
            MKL_NUM_THREADS="1",
        )
        commands = load(root / "commands.json")

        def stage(name, argv):
            with (execution / (name + ".log")).open("x") as log:
                p = subprocess.Popen(
                    argv, cwd="/tmp", env=env, stdout=log, stderr=subprocess.STDOUT
                )
                save(
                    execution / (name + "-start.json"),
                    dict(epoch=time.time(), pid=p.pid, command=argv),
                )
                code = p.wait()
            save(
                execution / (name + "-exit.json"),
                dict(epoch=time.time(), returncode=code),
            )
            require(code == 0, "Stage failed: " + name)

        def simulate(tag):
            freeze_check(root)
            save(execution / (tag + "-admission.json"), resources(root))
            (root / "runs" / tag).mkdir()
            stage(tag, commands[tag])
            require(
                load(root / f"runs/{tag}/config-gate.json")
                == dict(passed=True, differences=[]),
                "Machine configuration differs",
            )
            if tag == "configuration":
                require(
                    load(root / "runs/configuration/initialization-check.json")[
                        "passed"
                    ],
                    "Initialization failed",
                )
            else:
                end = load(root / f"runs/{tag}/simulation-exit.json")
                require(
                    end["cause"] == "m5_exit instruction encountered"
                    and end["code"] == 0,
                    "Abnormal exit",
                )
                text = (root / f"runs/{tag}/system.terminal").read_text()
                require(
                    "FAILED" not in text and text.count("SDOT_GUEST_COMPLETE") == 1,
                    "Guest failure",
                )
                expected = (
                    "SDOT_VERIFY_PASS cases=144"
                    if tag == "verify"
                    else f"SDOT_COMPLETE experiment={tag[-1]} blocks=137 cases=812"
                )
                require(text.count(expected) == 1, "Guest coverage incomplete")

        simulate("configuration")
        simulate("verify")
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(simulate, f"exp{i}") for i in range(3)]
            errors = []
            for future in futures:
                try:
                    future.result()
                except Exception as error:
                    errors.append(str(error))
            require(not errors, "; ".join(errors))
        freeze_check(root)
        save(
            execution / "simulations-complete.json",
            dict(status="passed", measured_ROIs=2436, empty_ROIs=9),
        )
        stage(
            "analysis",
            [
                sys.executable,
                "-B",
                str(root / "control/scan.py"),
                "analyze",
                "--root",
                str(root),
            ],
        )
        stage(
            "results",
            [
                sys.executable,
                "-B",
                str(root / "control/scan.py"),
                "results",
                "--root",
                str(root),
            ],
        )
        require(load(root / "automated-qa.json")["status"] == "passed", "QA failed")
        freeze_check(root)
        save(
            execution / "complete.json",
            dict(
                status="passed",
                epoch=time.time(),
                freeze_sha256=digest(root / "freeze.json"),
            ),
        )
    except Exception:
        save(
            execution / "failure.json",
            dict(epoch=time.time(), traceback=traceback.format_exc()),
        )
        raise
