#!/usr/bin/env python3
"""Exercise collection and portable comparison against a COPY of real old stats.

I2CE_RESULTS_FIXTURE points at an exported CB4-B0-r1 run directory.
Synthetic E03/E04 records below test tooling only; they are not measurements.
"""
import csv
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('cache_study',ROOT/'tools/exp/cache_study.py')
study=importlib.util.module_from_spec(spec);spec.loader.exec_module(study)


class Workflow(unittest.TestCase):
    def test_collection_identity_and_portable_comparison(self):
        fixture=Path(os.environ['I2CE_RESULTS_FIXTURE'])
        with tempfile.TemporaryDirectory() as temp:
            exports=Path(temp); namespace=study.study('E03'); result_root=exports/'results'/namespace
            for eid in ['E03','E04']:
                run=exports/'runs'/eid;shutil.copytree(fixture,run)
                for name in ['collected_as','collect.log','provenance.tsv']:
                    (run/name).unlink(missing_ok=True)
                command=['bash',str(ROOT/'tools/exp/exp.sh'),'collect','38','--run',str(run),
                         '--study',namespace,'--exp-id',eid,'--output-root',str(result_root),'--gem5-timestamp','20000101_000000']
                result=subprocess.run(command,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
                self.assertEqual(result.returncode,0,result.stdout)
                duplicate=subprocess.run(command,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
                self.assertNotEqual(duplicate.returncode,0,'Duplicate collection must be rejected')
                row=next(r for r in study.table(result_root/'manifest.tsv') if r['exp_id']==eid)
                self.assertEqual(row['study'],namespace)
                self.assertEqual(row['runner_id'],'38')
                record={'server_sha':study.B0,'result_file':row['output_file'],
                        'result_sha256':study.sha(Path(row['output_file'])),
                        **{key:{'synthetic_test_fixture':True} for key in ['hardware','generated','machine_config','controller','checkpoint']}}
                (run/'run.json').write_text(json.dumps(record))
            study.compare(exports,exports/'comparison')
            report=(exports/'comparison/README.md').read_text()
            self.assertIn('| E03 | E04 |',report);self.assertIn('1.000000×',report)
            record['checkpoint']={'different':True}
            (exports/'runs/E04/run.json').write_text(json.dumps(record))
            with self.assertRaisesRegex(RuntimeError,'fingerprint mismatch: checkpoint'):
                study.compare(exports,exports/'bad-comparison')
            record['checkpoint']={'synthetic_test_fixture':True}
            (exports/'runs/E04/run.json').write_text(json.dumps(record))
            Path(row['output_file']).write_text('corrupted')
            with self.assertRaisesRegex(RuntimeError,'checksum changed'):
                study.compare(exports,exports/'corrupt-comparison')


if __name__=='__main__':unittest.main()
