"""Synthetic checks of immutable records, finite budgets and parsed counters."""

import copy
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools/exp/lib"))
import study_config_gate as gate
import study_counters as counters
import study_resources as resources
from study_records import save, study_budget, verify_contract, digest


class StudyContracts(unittest.TestCase):
    def test_registered_budget_and_duplicate_paths(self):
        for count in [1, 3, 13]:
            registration = dict(
                maximum_formal_runs=count,
                automatic_retries=0,
                rows=[
                    dict(
                        id=f"E{i:02}",
                        state=f"/state/E{i:02}",
                        output=f"/output/E{i:02}",
                    )
                    for i in range(1, count + 1)
                ],
            )
            self.assertEqual(study_budget(registration), count)
            registration["maximum_formal_runs"] = count + 1
            with self.assertRaises(ValueError):
                study_budget(registration)
        registration["maximum_formal_runs"] = count
        registration["rows"][1]["state"] = registration["rows"][0]["state"]
        with self.assertRaises(ValueError):
            study_budget(registration)

    def test_gates_survive_optimized_python(self):
        script = "from study_records import require; require(False, 'gate active')"
        for flags in [[], ["-O"], ["-OO"]]:
            result = subprocess.run(
                [sys.executable, *flags, "-c", script],
                cwd=REPO / "tools/exp/lib",
                capture_output=True,
                text=True,
                env={**__import__("os").environ, "PYTHONDONTWRITEBYTECODE": "1"},
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("ValueError: gate active", result.stderr)

    def test_contract_rejects_changed_inputs_and_records_are_immutable(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            save(
                root / "registration.json",
                dict(
                    maximum_formal_runs=1,
                    automatic_retries=0,
                    rows=[dict(id="E01", state="/state/E01", output="/out/E01")],
                ),
            )
            save(
                root / "helper-contract.json",
                dict(files={"registration.json": digest(root / "registration.json")}),
            )
            expected = digest(root / "helper-contract.json")
            verify_contract(root, expected)
            with self.assertRaises(FileExistsError):
                save(root / "registration.json", {})
            (root / "registration.json").write_text("{}")
            with self.assertRaises(ValueError):
                verify_contract(root, expected)

    def test_only_host_paths_are_mutable(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "reference").mkdir()
            output = root / "output"
            output.mkdir()
            ini = "[system]\nreadfile=old\n[system.pci_devices1.vio]\nroot=old\nsocketPath=old\n[system.workload]\ndtb_filename=old\n[cache]\nsize=32768\n"
            (root / "reference/config.ini").write_text(ini)
            source = {
                "system": {
                    "path": "system",
                    "readfile": "old",
                    "vio": {
                        "path": "system.pci_devices1.vio",
                        "root": "old",
                        "socketPath": "old",
                    },
                    "workload": {"path": "system.workload", "dtb_filename": "old"},
                    "cache": {"size": 32768},
                }
            }
            save(root / "reference/config.json", source)
            expected_ini, expected_json, allowed = gate.expected_config(
                root, root / "inputs", output
            )
            self.assertEqual(len(allowed), 4)
            self.assertFalse(
                gate.differences(expected_json, copy.deepcopy(expected_json))
            )
            bad = copy.deepcopy(expected_json)
            bad["system"]["cache"]["size"] = 65536
            self.assertEqual(len(gate.differences(expected_json, bad)), 1)
            del bad["system"]["cache"]
            self.assertTrue(gate.differences(expected_json, bad))
            self.assertEqual(expected_ini["cache"]["size"], "32768")

    def test_counters_preserve_all_intervals_and_post_roi(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            evidence = root / "audit"
            evidence.mkdir()
            (root / "gem5_profile_regions.tsv").write_text(
                "header\nheader\n"
                + "".join(
                    (f"{i}\t0\t{label}\n" for i, label in enumerate(counters.LABELS))
                )
            )
            mapping = dict(counters.MAP, **counters.EXTRA, sim_ticks="simTicks")
            blocks = []
            for i in range(1, 24):
                values = {stat: i * 10 for stat in mapping.values()}
                values["simFreq"] = 10**12
                values["simTicks"] = i * 1000
                values[counters.MAP["toL2_total_packet_bytes"]] = i * 30
                for prefix in counters.PREFIX.values():
                    values[prefix + ".demandAvgMissLatency::total"] = 1
                    values[prefix + ".demandAvgMshrMissLatency::total"] = 1
                blocks.append(
                    "---------- Begin Simulation Statistics ----------\n"
                    + "".join((f"{k} {v}\n" for k, v in values.items()))
                    + "---------- End Simulation Statistics ----------\n"
                )
            (root / "stats.txt").write_text("".join(blocks))
            result = counters.parse(root, evidence)
            self.assertEqual(result["metrics"]["BLOCK"]["sim_ticks"], 22000)
            self.assertEqual(
                result["post_ROI_dump23_delta_separate"]["sim_ticks"], 1000
            )
            self.assertEqual(len(result["metrics"]["BLOCK"]), 65)
            with self.assertRaises(FileExistsError):
                counters.parse(root, evidence)
            with self.assertRaises(ValueError):
                counters.deltas([{"x": "2"}, {"x": "1"}], {"v": "x"})
            self.assertEqual(
                counters.deltas([{"x": "9007199254740993"}], {"v": "x"})[0]["v"],
                9007199254740993,
            )
            (root / "stats.txt").write_text("".join(blocks[:22]))
            with self.assertRaises(ValueError):
                counters.raw_blocks(root / "stats.txt")

    def test_resource_limits_and_no_fixed_cap(self):
        args = dict(
            cores=128,
            busy=0,
            available=512 * resources.GIB,
            memory_pending=0,
            free_disk=1024 * resources.GIB,
            package_free=32 * resources.GIB,
            active=0,
            per_cpu=2,
            per_memory=8 * resources.GIB,
            io_passed=True,
        )
        self.assertGreater(resources.capacity(**args)["additional_slots"], 13)
        for changes in [
            dict(busy=125),
            dict(available=16 * resources.GIB),
            dict(free_disk=16 * resources.GIB),
            dict(io_passed=False),
            dict(package_free=1),
            dict(memory_pending=500 * resources.GIB),
        ]:
            self.assertFalse(resources.capacity(**dict(args, **changes))["admitted"])
        self.assertEqual(
            resources.busy_cpus(
                {0: [10, 0, 10, 70, 10, 0, 0, 0]},
                {0: [20, 0, 20, 140, 20, 0, 0, 0]},
                [0],
            ),
            (0.2, 0.1),
        )


if __name__ == "__main__":
    unittest.main()
