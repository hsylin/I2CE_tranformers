"""Synthetic cumulative counters; never a performance observation."""

import json
import configparser
import shutil
from pathlib import Path

import roofline

ROOT = Path(__file__).resolve().parents[1]
LABELS = ["MHA_QKV", "MHA_QK", "MHA_softmax", "MHA_SV"] * 4 + [
    "MHA_out",
    "Projection",
    "non_GEMM_after_projection",
    "FF1",
    "FF2",
    "non_GEMM_after_ff2",
]


def bundle(folder):
    folder.mkdir(parents=True)
    calibration = ROOT / "tests/fixtures/roofline/legacy-cache-calibration.json"
    shutil.copyfile(calibration, folder / "calibration.json")
    c = json.loads(calibration.read_text())
    config = dict(system=c["hardware"]["system"])
    (folder / "config.json").write_text(json.dumps(config))
    ini = configparser.ConfigParser(interpolation=None)
    ini.optionxform = str
    ini.read_dict(c["hardware_ini"])
    with (folder / "config.ini").open("w") as stream:
        ini.write(stream)
    roi = dict(
        counter_mode="cumulative_since_reset",
        begin_block=None,
        end_block=21,
        expected_blocks=23,
        labels=LABELS,
    )
    op = dict(
        kind="transformer_gemm",
        learners=2,
        sequence=512,
        dimension=256,
        feedforward=1024,
    )
    measurement = dict(
        gem5_revision=roofline.MODEL_REVISION, roi=roi, operation_model=op
    )
    (folder / "measurement.json").write_text(json.dumps(measurement))
    (folder / "inputs.json").write_text(json.dumps({"synthetic": True}))
    (folder / "regions.tsv").write_text(
        "# synthetic labels\nindex\tgroup\tname\n"
        + "".join(f"{i}\t0\t{label}\n" for i, label in enumerate(LABELS, 1))
    )
    total = dict(simTicks=72086366500, simFreq=10**12, finalTick=72086366500)
    total[roofline.DOT] = 67108864
    total[roofline.MEMBUS_ROUTE] = 11804736
    total[roofline.DRAM + ".peakBW"] = 12800
    for level, counts in zip(
        roofline.TRAFFIC,
        [
            (439617833, 97232136),
            (188274659, 187275620),
            (6716288, 5073472),
            (6720000, 5084736),
        ],
    ):
        total.update(zip(roofline.TRAFFIC[level], counts))
    total[roofline.DRAM + ".readBursts"] = 6716288 // 64
    total[roofline.DRAM + ".writeBursts"] = 5073472 // 64
    lines = []
    for i in range(1, 24):
        v = {
            k: (
                value
                if k in ("simFreq", roofline.DRAM + ".peakBW")
                else value * min(i, 22) // 22
            )
            for k, value in total.items()
        }
        for direction, key in zip(("read", "write"), roofline.TRAFFIC["Physical DRAM"]):
            v[key] = v[roofline.DRAM + "." + direction + "Bursts"] * 64
        for level, (r, w, scale) in roofline.RATES.items():
            for rate, key in zip((r, w), roofline.TRAFFIC[level]):
                v[rate] = v[key] / (v["simTicks"] / v["simFreq"]) / scale
        # Match controller route exactly, including integer rounding in intermediate dumps.
        v[roofline.MEMBUS_ROUTE] = sum(
            v[k] for k in roofline.TRAFFIC["Membus/controller"]
        )
        lines.append("---------- Begin Simulation Statistics ----------")
        lines.extend(
            f"{k} {value:.6f}" if k in roofline.FORMULAS else f"{k} {value}"
            for k, value in v.items()
        )
        lines.append("---------- End Simulation Statistics ----------")
    (folder / "stats.txt").write_text("\n".join(lines) + "\n")
    artifacts = {}
    for key, name in dict(
        stats="stats.txt",
        config="config.json",
        config_ini="config.ini",
        regions="regions.tsv",
        inputs="inputs.json",
        measurement="measurement.json",
    ).items():
        artifacts[key] = dict(path=name, sha256=roofline.digest(folder / name))
    receipt = dict(
        host_returncode=0,
        guest_returncode=0,
        simulation_code=0,
        simulation_cause="m5_exit instruction encountered",
        config_gate_passed=True,
        guest_binary_hash_ok=True,
        source_commit="a" * 40,
        binary_sha256="c" * 64,
        inputs_sha256=artifacts["inputs"]["sha256"],
        measurement_sha256=artifacts["measurement"]["sha256"],
        artifact_sha256={k: v["sha256"] for k, v in artifacts.items()},
    )
    (folder / "receipt.json").write_text(json.dumps(receipt))
    artifacts["receipt"] = dict(
        path="receipt.json", sha256=roofline.digest(folder / "receipt.json")
    )
    run = dict(
        id="fixture",
        label="Synthetic counters for parser testing",
        status="diagnostic_measurement",
        gem5_revision=roofline.MODEL_REVISION,
        source_commit=receipt["source_commit"],
        binary_sha256=receipt["binary_sha256"],
        roi=roi,
        operation_model=op,
        artifacts=artifacts,
        inputs_sha256=receipt["inputs_sha256"],
        measurement_sha256=receipt["measurement_sha256"],
        limitations=["Synthetic test fixture; not an experiment."],
    )
    manifest = dict(
        schema_version=2,
        method="gem5_measured_cache_references",
        calibration=dict(
            path="calibration.json", sha256=roofline.digest(folder / "calibration.json")
        ),
        experiments=[
            dict(
                id="pending-stage",
                status="not_measured",
                unavailable_reason="No matching native CommMonitor evidence",
            ),
            run,
        ],
    )
    path = folder / "roofline.json"
    path.write_text(json.dumps(manifest))
    return path
