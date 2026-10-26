import copy
import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location(
    "comparison",
    Path(__file__).resolve().parents[1] / "tools/exp/lib/compare_saved_studies.py",
)
comparison = importlib.util.module_from_spec(spec)
spec.loader.exec_module(comparison)


class ComparisonTests(unittest.TestCase):
    def fixture(self):
        return dict(
            study="fixture",
            registration_sha256="registration",
            summary_sha256="summary",
            pins={"fixed": 1},
            observations={
                "E01": dict(
                    identity={"binary_sha256": "same"},
                    counter_sha256="counter",
                    frequency=1_000_000_000_000,
                    metrics={
                        name: dict(
                            sim_ticks=1_000_000_000, instructions=100, missing=None
                        )
                        for name in comparison.PHASES
                    },
                )
            },
        )

    def test_identical_binary_does_not_erase_observed_difference(self):
        first = self.fixture()
        second = copy.deepcopy(first)
        second["observations"]["E01"]["metrics"]["BLOCK"]["sim_ticks"] += 10_000_000
        result = comparison.compare(first, second)
        self.assertEqual(result["identical_binary_stages"], ["E01"])
        self.assertEqual(result["rows"][0]["delta_milliseconds"], 0.01)
        self.assertIsNone(
            result["rows"][0]["aggregate_counter_deltas"]["BLOCK"]["missing"]
        )
        self.assertEqual(first, self.fixture())

    def test_distinct_binary_is_not_a_repeat(self):
        first = self.fixture()
        second = copy.deepcopy(first)
        second["observations"]["E01"]["identity"]["binary_sha256"] = "different"
        self.assertEqual(
            comparison.compare(first, second)["identical_binary_stages"], []
        )

    def test_different_pins_are_rejected(self):
        first = self.fixture()
        second = copy.deepcopy(first)
        second["pins"] = {"fixed": 2}
        with self.assertRaisesRegex(ValueError, "pins differ"):
            comparison.compare(first, second)

    def test_missing_stage_is_rejected(self):
        first = self.fixture()
        second = copy.deepcopy(first)
        second["observations"] = {}
        with self.assertRaisesRegex(ValueError, "Stage sets"):
            comparison.compare(first, second)


if __name__ == "__main__":
    unittest.main()
