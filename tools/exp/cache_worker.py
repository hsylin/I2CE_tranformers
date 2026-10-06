#!/usr/bin/env python3
"""Finite dispatcher/collector for one approved cache-study phase.

It never resubmits a reserved action or retries a failed simulation. A failed
validation/build needs inspection; partial evidence is retained. Restart only
after checking the existing worker and run receipts.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
import cache_study as study


def invoke(*args):
    subprocess.run([sys.executable,'-B',str(study.EXP/'cache_study.py'),*args],check=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase',choices=['hierarchy']);a=p.parse_args()
    ids=['E06','E07','E08','E09']
    prepare_order=ids
    with study.lock('worker'):
        study.checked()
        for eid in ids:study.gate_for(eid)
        for eid in prepare_order:
            if not (study.STATE/'prepared'/(eid+'.json')).exists():invoke('prepare',eid)
            if not (study.STATE/'correctness'/eid/'result.json').exists():
                subprocess.run([sys.executable,'-B',str(study.EXP/'cache_correctness.py'),eid],check=True)
        ready=json.loads((study.STATE/'packing-ready.json').read_text())
        for eid in ids:
            prepared=json.loads((study.STATE/'prepared'/(eid+'.json')).read_text())
            study.require(ready[eid]['source_sha']==prepared['server_sha'] and
                          ready[eid]['binary_sha256']==prepared['build']['binary_sha256'] and
                          ready[eid]['passed'] is True, 'Production packing gate not approved')
        for eid in ids:
            if (study.STATE/'runs'/(eid+'.json')).exists():continue
            if (study.STATE/'pending'/(eid+'.json')).exists():invoke('recover',eid)
            else:invoke('submit',eid)
        deadline=time.monotonic()+72*3600
        while time.monotonic()<deadline:
            invoke('collect','--finished')
            if all('result_file' in json.loads((study.STATE/'runs'/(eid+'.json')).read_text()) for eid in ids):
                invoke('compare','--all')
                invoke('export','--out',str(study.STATE/'exports'))
                study.save(study.STATE/(a.phase+'-worker-complete.json'),dict(eids=ids,finished_at=time.time()))
                print('Phase complete; review results before the next phase.',flush=True)
                return
            time.sleep(60)
        raise RuntimeError('72-hour collector deadline reached; inspect existing jobs, never automatically repeat')


if __name__=='__main__':main()
