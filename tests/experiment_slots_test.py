import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location('slots', Path(__file__).resolve().parents[1] / 'tools/exp/lib/simulation_slots.py')
slots = importlib.util.module_from_spec(spec)
spec.loader.exec_module(slots)


class AdmissionTests(unittest.TestCase):
    def decide(self, live=0, limit='auto', **changes):
        info = dict(physical_cpus=36, affinity_cpus=72, cpu_budget=36,
                    memory_available=320 * slots.GIB, load1=0)
        info.update(changes)
        return slots.admission(info, live, limit, 8 * slots.GIB, 4, 16 * slots.GIB)

    def test_auto_can_exceed_eight_without_counting_smt_as_cores(self):
        self.assertEqual(self.decide()['cap'], 32)
        self.assertTrue(self.decide(live=31, load1=31)['admitted'])
        self.assertFalse(self.decide(live=32, load1=32)['admitted'])

    def test_memory_pressure_stops_even_manual_cap(self):
        self.assertFalse(self.decide(limit=64, memory_available=20 * slots.GIB)['admitted'])

    def test_quota_and_external_load(self):
        self.assertEqual(self.decide(cpu_budget=12)['cap'], 8)
        self.assertEqual(self.decide(live=4, load1=10)['cap'], 26)
        self.assertEqual(self.decide(limit=100, cpu_budget=12)['cap'], 12)

    def test_bad_limit(self):
        for value in ['0', '-1', 'unlimited', '2.5']:
            with self.assertRaises(Exception):
                slots.positive(value)
        self.assertEqual(slots.positive('16'), 16)


if __name__ == '__main__':
    unittest.main()
