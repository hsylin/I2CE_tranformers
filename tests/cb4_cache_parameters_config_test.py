"""Manual host-side interface checks; never builds or starts gem5."""
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class Configuration(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.fake = self.root / 'compiler'
        self.fake.write_text('#!/bin/bash\nif [[ "$1" == -print-sysroot ]]; then exit 0; fi\nprintf "%s\\n" "$@" > "$ARGS_LOG"\n')
        self.fake.chmod(0o755)
        self.env = {k: v for k, v in os.environ.items() if not k.startswith('I2CE_CACHE_')}
        self.env.update(A64CXX=str(self.fake), CONDA_PREFIX='', ARGS_LOG=str(self.root/'args'),
                        SIMD_FLAG='1', USE_CODEBOOK_GEMM_FLAG='1', I2CE_USE_LIBM5_FLAG='0')

    def compile(self, **changes):
        return subprocess.run(['bash', str(ROOT/'compile_transformer.sh')], cwd=self.root,
                              env=dict(self.env, **changes), text=True, capture_output=True)

    def test_compile_default_and_override(self):
        result = self.compile()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn('-DI2CE_CACHE_', (self.root/'args').read_text())
        result = self.compile(I2CE_CACHE_K1='64', I2CE_CACHE_O1='16')
        self.assertEqual(result.returncode, 0, result.stderr)
        args = (self.root/'args').read_text().splitlines()
        self.assertIn('-DI2CE_CACHE_K1=64', args)
        self.assertIn('-DI2CE_CB4_CACHE_CONFIG_REQUESTED=1', args)

    def test_bad_shell_values_and_disabled_path(self):
        for bad in ['', '0', '-1', '016', '65536', '16 -DX=1', '$(touch forbidden)']:
            result = self.compile(I2CE_CACHE_S1=bad)
            self.assertNotEqual(result.returncode, 0, bad)
        for flags in [{'SIMD_FLAG':'0'}, {'USE_CODEBOOK_GEMM_FLAG':'0'}, {'CORE_NUM_FLAG':'2'}]:
            self.assertNotEqual(self.compile(I2CE_CACHE_K1='64', **flags).returncode, 0)
        self.assertFalse((self.root/'forbidden').exists())

    def runner(self, overrides, cb=4, sve=128, source=ROOT):
        folder = self.root/'tools/exp'
        folder.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT/'tools/exp/exp.sh', folder/'exp.sh')
        cfg = self.root/'gem5.py'
        cfg.write_text('# CowDiskImage --l1d-size --sve-vl\n')
        (folder/'runner.conf').write_text('\n'.join(f'{k}={shlex.quote(str(v))}' for k,v in {
            'EXP_ROOT':self.root/'runs', 'GEM5_BIN':'/usr/bin/true', 'GEM5_CWD':self.root,
            'GEM5_CFG':'gem5.py', 'KERNEL':cfg, 'DISK':cfg, 'REPO_ROOT':source,
            'GEN_PYTHON':'/usr/bin/true', 'USE_LIBM5':0}.items())+'\n')
        (folder/'experiments.tsv').write_text(f'99\t{cb}\t2\t{sve}\tcodebook_int8\t{overrides}\ttest only\n')
        return subprocess.run(['bash', str(folder/'exp.sh'), 'dryrun', '99'],
                              env=self.env, text=True, capture_output=True)

    def test_runner_records_tiles_and_keeps_hardware_separate(self):
        result = self.runner('tile_k1=64,tile_o1=16,l1d=64KiB,l2=2MiB')
        self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
        self.assertIn('I2CE_CACHE_K1=64', result.stdout)
        self.assertIn('--l1d-size=64KiB', result.stdout)
        self.assertIn('--l2-size=2MiB', result.stdout)
        self.assertFalse((self.root/'runs').exists())

    def test_runner_rejects_unsupported_or_ignored_overrides(self):
        for overrides, cb, sve in [('tile_k1=0',4,128), ('tile_k1=64',8,128),
                                   ('tile_k1=64',4,256), ('cores=2',4,128)]:
            self.assertNotEqual(self.runner(overrides,cb,sve).returncode, 0)
        old = self.root/'old'; old.mkdir()
        (old/'compile_transformer.sh').write_text('# no tile support\n')
        result = self.runner('tile_k1=64',source=old)
        self.assertNotEqual(result.returncode,0)
        self.assertIn('ignored override',result.stderr)


if __name__ == '__main__':
    unittest.main()
