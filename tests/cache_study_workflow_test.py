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


class RevisionControllerTests(unittest.TestCase):
    def test_only_frozen_baseline_can_cross_controller(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            old={'sha':'old','files':{'exp':'old-hash'}}
            new={'sha':'new','files':{'exp':'new-hash'}}
            (root/'reference-controller.json').write_text(json.dumps(old))
            (root/'controller.json').write_text(json.dumps(new))
            study.verify_comparison_controllers('E05','E06',{'controller':old},{'controller':new},root)
            for base,cand,b,c in [('E06','E07',old,new),('E05','E06',old,{'sha':'tampered'}),('E05','E06',new,old)]:
                with self.assertRaisesRegex(RuntimeError,'Comparison controller'):
                    study.verify_comparison_controllers(base,cand,{'controller':b},{'controller':c},root)

    def test_original_baselines_cannot_be_prepared_or_submitted(self):
        for eid in ['E01','E02','E03','E04','E05']:
            for fn in [study.prepare,study.submit]:
                with self.assertRaisesRegex(RuntimeError,'read-only references'):
                    fn(eid)

if __name__=='__main__':unittest.main()
