"""Negative checks for portable publication validation, including python -O."""
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]/'transformer_profiling/hsylin'
spec = importlib.util.spec_from_file_location('verify_results', ROOT/'reproduction/verify_results.py')
verify = importlib.util.module_from_spec(spec)
spec.loader.exec_module(verify)


class PublicationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)/'publication'
        shutil.copytree(ROOT, self.root, ignore=shutil.ignore_patterns('__pycache__'))

    def change_json(self, relative, mutate):
        path = self.root/relative
        value = json.loads(path.read_text())
        mutate(value)
        path.write_text(json.dumps(value))
        # Recompute envelope hashes so semantic gates, not just checksums, run.
        self.rehash()

    def rehash(self):
        (self.root/'SHA256SUMS').write_text(''.join(
            hashlib.sha256(p.read_bytes()).hexdigest()+'  '+str(p.relative_to(self.root))+'\n'
            for p in sorted(self.root.rglob('*')) if p.is_file() and p.name!='SHA256SUMS'))

    def test_changed_source_is_rejected(self):
        self.change_json('manifest.json', lambda value: value['experiments'][0].update(server_sha='0'*40))
        with self.assertRaisesRegex(RuntimeError, 'identity mismatch server_sha'):
            verify.verify(self.root)

    def test_changed_binary_is_rejected(self):
        self.change_json('manifest.json', lambda value: value['experiments'][0].update(binary_sha256='0'*64))
        with self.assertRaisesRegex(RuntimeError, 'binary mismatch'):
            verify.verify(self.root)

    def test_changed_hardware_is_rejected(self):
        self.change_json('cb4-baseline-optimization/experiments/E01/run.json',
                         lambda value: value['hardware'].update(unexpected_cache_change=True))
        with self.assertRaisesRegex(RuntimeError, 'cross-run difference hardware'):
            verify.verify(self.root)

    def test_duplicate_identity_is_rejected(self):
        self.change_json('manifest.json', lambda value: value['experiments'].__setitem__(0, value['experiments'][1]))
        with self.assertRaisesRegex(RuntimeError, 'nine distinct experiments'):
            verify.verify(self.root)

    def test_incomplete_raw_block_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, 'Incomplete statistics block'):
            verify.blocks('---------- Begin Simulation Statistics ----------\nsimTicks 100\n')

    def test_corrupt_file_is_rejected(self):
        (self.root/'README.md').write_text('unexpected modification')
        with self.assertRaisesRegex(RuntimeError, 'Checksum mismatch'):
            verify.verify(self.root)


if __name__ == '__main__':
    unittest.main()
