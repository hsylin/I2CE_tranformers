#!/usr/bin/env python3
"""Isolated packing revision: reuse E01-E05; one new run per E06-E09."""
import argparse
import configparser
from contextlib import contextmanager
import csv
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import time
ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT/'tools/exp'
STATE = Path.home()/'i2ce/cb4-cache-packing-revision'
B0 = '33d76aa29d8e199caee430f45ae59ae1ae16fd3b'
ROW = '38'


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def git(*args):
    return subprocess.check_output(['git','-C',str(ROOT),*args],text=True).strip()


def save(path, value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps(value,indent=2)+'\n');temp.replace(path)


def kv(path):
    return dict(line.split('\t',1) for line in Path(path).read_text().splitlines() if '\t' in line)


def table(path):
    with Path(path).open() as file:
        return list(csv.DictReader(file, delimiter='\t'))


def config():
    keys=['EXP_ROOT','REPO_ROOT','GEM5_BIN','GEM5_CWD','GEM5_CFG','GEM5_ROOT',
          'LIBM5_A','KERNEL','DISK','GEN_PYTHON','MAX_PARALLEL','USE_LIBM5',
          'RUN_CPU','BOOT_CPU','NUM_CORES','GEM5_MACHINE_ARGS']
    script='set -a\nsource "$1"\npython3 -c \'import json,os,sys; print(json.dumps({k:os.environ.get(k,"") for k in sys.argv[1:]}))\' "${@:2}"'
    c=json.loads(subprocess.check_output(['bash','-c',script,'bash',str(EXP/'runner.conf'),*keys],text=True))
    require(Path(c['REPO_ROOT']).resolve()==ROOT,'runner.conf points at another checkout')
    require(c['USE_LIBM5']=='1','Study requires libm5')
    require(Path(c['EXP_ROOT']).resolve()==Path.home()/'i2ce/exp-cb4-cache-packing-revision', 'Use the isolated cache-study experiment root')
    c['GEM5_CFG']=c['GEM5_CFG'] or 'configs/example/arm/starter_fs.py'
    c['GEM5_ROOT']=c['GEM5_ROOT'] or c['GEM5_CWD']
    c['LIBM5_A']=c['LIBM5_A'] or c['GEM5_ROOT']+'/util/m5/build/arm64/out/libm5.a'
    return c


def environment(c):
    env=os.environ.copy()
    env['I2CE_IDX_BITS_BYTE_ALIGNED']='1'
    env['PYTHONDONTWRITEBYTECODE']='1'
    # Use the existing server's generator environment and its cross compiler.
    prefix=Path(c['GEN_PYTHON']).parent.parent
    compiler=prefix/'bin/aarch64-conda-linux-gnu-g++'
    require(compiler.is_file(),'Configured generator environment lacks the cross compiler')
    env['CONDA_PREFIX']=str(prefix);env['A64CXX']=str(compiler)
    env['PATH']=str(prefix/'bin')+os.pathsep+env['PATH']
    return env


@contextmanager
def lock(name):
    STATE.mkdir(parents=True,exist_ok=True)
    with (STATE/(name+'.lock')).open('a') as f:
        try: fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError: raise RuntimeError('Another '+name+' operation is active') from None
        yield


def controller():
    return {'sha':git('rev-parse','HEAD'), 'files':{str(p.relative_to(ROOT)):sha(p) for p in
            [EXP/'runner.conf',EXP/'exp.sh',EXP/'experiments.tsv',EXP/'cache_study.py', EXP/'cache_correctness.py', EXP/'cache_worker.py', EXP/'lib/simulation_slots.py',
             EXP/'lib/run.rcS.tmpl',
             EXP/'lib/gen_artifacts.py',ROOT/'transformer_profiling/add_experiment.py']}}


def checked():
    expected=json.loads((STATE/'controller.json').read_text())
    require(controller()==expected,'Frozen controller changed')
    require(subprocess.run(['git','-C',str(ROOT),'diff','--quiet','HEAD','--','tools','transformer_profiling/add_experiment.py']).returncode==0,
            'Tracked controller code has local changes')
    return expected


def fingerprint(c):
    # Re-hash an asset whenever its file metadata changes; keep large disk hashes
    # outside the repository. No environment variables or credentials are stored.
    path=STATE/'asset-hashes.json'
    cache=json.loads(path.read_text()) if path.exists() else {}
    paths={k:Path(c[k]) for k in ['GEM5_BIN','KERNEL','DISK','LIBM5_A']}
    paths['starter_fs']=Path(c['GEM5_CWD'])/c['GEM5_CFG']
    paths['devices']=Path(c['GEM5_CWD'])/'configs/example/arm/devices.py'
    paths['compiler']=Path(environment(c)['A64CXX'])
    result={}
    for key,p in paths.items():
        st=p.stat(); stamp=[st.st_size,st.st_mtime_ns,st.st_ino]
        cached=cache.get(str(p),{})
        if cached.get('stamp')!=stamp:
            print('Fingerprinting',key,flush=True)
            cached={'stamp':stamp,'sha256':sha(p)};cache[str(p)]=cached
        result[key]={'path':str(p),'sha256':cached['sha256']}
    result['gem5_sha']=subprocess.check_output(['git','-C',c['GEM5_CWD'],'rev-parse','HEAD'],text=True).strip()
    diff=subprocess.check_output(['git','-C',c['GEM5_CWD'],'diff','HEAD'])
    result['gem5_diff_sha256']=hashlib.sha256(diff).hexdigest()
    result['compiler_version']=subprocess.check_output([str(paths['compiler']),'--version'],text=True)
    save(path,cache)
    return result


def study(eid):
    require(re.fullmatch(r'E0[1-9]', eid), 'Expected E01 through E09')
    return 'cb4-baseline-optimization' if int(eid[1:]) <= 5 else 'cb4-hierarchical-cache-packing'


def manifest():
    return json.loads((STATE/'manifest.json').read_text())


def source(eid):
    row = manifest()['actions'][eid]
    require(re.fullmatch(r'[0-9a-f]{40}', row.get('server_sha') or ''), 'Source has not been implemented/registered')
    require(git('rev-parse', row['server_sha']+'^{tree}') == row['source_tree'], 'Source tree mismatch')
    return row


def verify(path):
    with lock('submit'):
        data = json.loads(path.read_text())
        require(set(data['actions']) == {f'E{i:02}' for i in range(1,10)}, 'Manifest must contain all nine fixed slots')
        require(data['baseline_sha'] == B0 and data['repeats'] == 1, 'Wrong baseline/repetition contract')
        for eid, row in data['actions'].items():
            require(row['study'] == study(eid), 'Incorrect study namespace')
            if row.get('server_sha'):
                require(git('rev-parse', row['server_sha']+'^{tree}') == row['source_tree'], 'Source tree mismatch: '+eid)
            for folder in ['prepared', 'pending', 'runs']:
                existing = STATE/folder/(eid+'.json')
                if existing.exists():
                    old = json.loads(existing.read_text())
                    require(old['server_sha'] == row['server_sha'] and old['source_tree'] == row['source_tree'], 'Cannot change a prepared/submitted source: '+eid)
        c = config()
        hw = fingerprint(c)
        require(hw == json.loads((STATE/'expected-hardware.json').read_text()), 'Machine/compiler differs from preserved CB4 reference')
        if (STATE/'controller.json').exists():
            checked()
        else:
            require(subprocess.run(['git','-C',str(ROOT),'diff','--quiet','HEAD','--','tools','transformer_profiling/add_experiment.py']).returncode == 0, 'Commit controller before freezing')
            save(STATE/'controller.json', controller())
            save(STATE/'hardware.json', hw)
        rows = [r.split('\t') for r in (EXP/'experiments.tsv').read_text().splitlines() if r.split('\t')[0] == ROW]
        require(len(rows)==1 and rows[0][1:6]==['4','2','128','codebook_int8','-'], 'Wrong runner 38')
        save(STATE/'manifest.json', data)
        save(STATE/'manifest-origin.json', dict(path=str(path.resolve()),sha256=sha(path)))
        print('Verified manifest and frozen controller', git('rev-parse','HEAD'))


def prepared_data(build):
    return {'headers':{p.name:sha(p) for p in sorted((build/'headers').glob('*.h'))},
            'weights':{str(p.relative_to(build/'weights')):sha(p) for p in sorted((build/'weights').rglob('*')) if p.is_file()}}


def prepare(eid):
    require(eid in ['E06','E07','E08','E09'], 'Original baseline runs are read-only references')
    with lock('submit'):
        checked(); gate_for(eid); row=source(eid); c=config()
        require(not (STATE/'prepared'/(eid+'.json')).exists(), 'Already prepared; reuse the recorded immutable build')
        require(fingerprint(c)==json.loads((STATE/'hardware.json').read_text()), 'Hardware/compiler changed')
        log=STATE/'logs'/(eid+'.build.log'); log.parent.mkdir(exist_ok=True)
        with log.open('x') as output:
            subprocess.run(['bash','./exp.sh','build',ROW,'--at',row['server_sha']],cwd=EXP,
                           env=environment(c),stdout=output,stderr=subprocess.STDOUT,check=True)
        build=(Path(c['EXP_ROOT'])/ROW/'share').resolve(); bc=kv(build/'build_config.tsv')
        require(bc['repo_commit_full']==row['server_sha'] and '+dirty' not in bc['repo_commit'], 'Wrong/dirty source')
        require([bc[k] for k in ['codebook_size','n_learners','sve_bits','idx_bits_byte_aligned']]==['4','2','128','1'], 'Wrong generated configuration')
        require('I2CE_USE_LIBM5_FLAG=1' in bc['compile_flags'], 'Wrong profiling mechanism')
        require(re.search(r'^\s*#define\s+BITS_PER_CB\s+2\b',(build/'codebooks_def.h.provenance').read_text(),re.M), 'Not shared I2 layout')
        require(sha(build/'transformer.o')==bc['binary_sha256'], 'Binary hash mismatch')
        generated=prepared_data(build)
        require(generated['headers'] and generated['weights'], 'Missing generated artifacts')
        expected=STATE/'generated-data.json'
        if expected.exists():
            require(generated==json.loads(expected.read_text()), 'Generated inputs/weights/headers differ: '+eid)
        else:
            require(eid=='E03', 'Prepare E03 first to establish identical generated data')
            save(expected, generated)
        save(STATE/'prepared'/(eid+'.json'),dict(row,eid=eid,build=bc,build_dir=str(build),generated=generated))
        print('Prepared',eid,build)


def gate_for(eid):
    if int(eid[1:]) >= 6:
        gate=json.loads((STATE/'baseline-gate.json').read_text())
        for number in range(1,6):
            prior=json.loads((STATE/'runs'/f'E{number:02}.json').read_text())
            require(prior.get('result_sha256')==sha(Path(prior['result_file'])), 'Baseline phase is incomplete/changed')
        require(gate.get('approved') is True, 'Non-tiling gate not approved')
        require(gate['E05_server_sha']==source('E05')['server_sha'], 'Gate source mismatch')
        run=json.loads((STATE/'runs/E05.json').read_text())
        require(gate['E05_result_sha256']==sha(Path(run['result_file'])), 'Gate result changed')


def record(eid):
    pending=json.loads((STATE/'pending'/(eid+'.json')).read_text()); c=config()
    matches=[]
    for run in (Path(c['EXP_ROOT'])/ROW).glob('out_*'):
        marker=run/'study-action.json'
        if marker.exists() and json.loads(marker.read_text()).get('eid')==eid:
            matches.append(run)
    require(len(matches)==1, 'Expected exactly one immutable run marker; inspect pending launch, do not repeat')
    run=matches[0]; bc=kv(run/'build_config.tsv')
    require(bc['repo_commit_full']==pending['server_sha'] and bc['binary_sha256']==pending['build']['binary_sha256'], 'Launch used wrong build')
    require((run/'launch.json').exists(), 'No scheduler receipt')
    receipt=json.loads((run/'launch.json').read_text())
    require(receipt['status']=='launched', 'Inspect interrupted scheduler launch')
    save(STATE/'runs'/(eid+'.json'),dict(pending,run_dir=str(run),launch=receipt))
    print('Recorded',eid,run)


def checkpoint(c):
    directory=Path(c['EXP_ROOT'])/'_cpt/sve128'
    candidates=sorted(directory.glob('cpt.*'))
    require(len(candidates)==1 and candidates[0].is_dir(), 'Install exactly one preserved boot checkpoint before submission')
    return {p.name:sha(p) for p in sorted(candidates[0].iterdir()) if p.is_file()}


def submit(eid):
    require(eid in ['E06','E07','E08','E09'], 'Original baseline runs are read-only references')
    with lock('submit'):
        checked(); gate_for(eid); row=source(eid); c=config()
        for folder in ['runs','pending']:
            require(not (STATE/folder/(eid+'.json')).exists(), 'Already submitted/reserved; recover instead of relaunching')
        prepared=json.loads((STATE/'prepared'/(eid+'.json')).read_text())
        validation=json.loads((STATE/'correctness'/eid/'result.json').read_text())
        require(validation['source_sha']==row['server_sha'] and validation.get('passed') is True, 'Correctness is not verified')
        require(validation['input_generation']==prepared['generated'], 'Correctness used different generated inputs')
        ready=json.loads((STATE/'packing-ready.json').read_text())[eid]
        require(ready.get('passed') is True and ready['source_sha']==row['server_sha'] and
                ready['binary_sha256']==prepared['build']['binary_sha256'], 'Production packing gate not approved')
        require(prepared_data(Path(prepared['build_dir']))==prepared['generated'], 'Prepared data changed')
        require(sha(Path(prepared['build_dir'])/'transformer.o')==prepared['build']['binary_sha256'], 'Prepared binary changed')
        hardware=fingerprint(c); require(hardware==json.loads((STATE/'hardware.json').read_text()), 'Hardware/compiler changed')
        context=dict(prepared,controller=checked(),hardware=hardware,checkpoint=checkpoint(c),
                     correctness=validation,submitted_at=time.time())
        save(STATE/'pending'/(eid+'.json'),context)
        env=environment(c);env['I2CE_RUN_BUILD_DIR']=prepared['build_dir'];env['I2CE_STUDY_ACTION']=eid
        log=STATE/'logs'/(eid+'.submit.log')
        with log.open('x') as output:
            subprocess.run(['bash','./exp.sh','checkpoint',ROW,'--at',row['server_sha']],cwd=EXP,env=env,stdout=output,stderr=subprocess.STDOUT,check=True)
            subprocess.run(['bash','./exp.sh','run',ROW,'--at',row['server_sha']],cwd=EXP,env=env,stdout=output,stderr=subprocess.STDOUT,check=True)
        record(eid)


def live(run):
    pid=Path(run)/'pid'
    if not pid.exists(): return False
    p=Path('/proc')/pid.read_text().strip()/'cmdline'
    return p.exists() and str(run).encode() in p.read_bytes()


def machine(run):
    c=configparser.ConfigParser(interpolation=None,strict=False);c.read(run/'config.ini')
    for section,size,assoc in [('system.cpu_cluster.cpus.dcache','32768','2'),('system.cpu_cluster.l2','1048576','16')]:
        require(c.get(section,'size')==size and c.get(section,'assoc')==assoc,'Wrong actual cache geometry')
    return {s:{k:v for k,v in c.items(s) if re.search(r'type|clock|freq|size|assoc|latency|mshrs|limit|width|sve|channel|ranges',k,re.I)} for s in c.sections()}


def collect_one(eid):
    path=STATE/'runs'/(eid+'.json');r=json.loads(path.read_text());run=Path(r['run_dir'])
    if 'result_file' in r:
        require(sha(Path(r['result_file']))==r['result_sha256'], 'Collected result changed')
        return
    if live(run):
        print(eid,'RUNNING');return
    require('m5_exit instruction encountered' in (run/'gem5_stdout.log').read_text(), 'Simulation failed: '+eid)
    terminal=(run/'system.terminal').read_text(errors='replace')
    require('[guest] exp 38 exit code: 0' in terminal and '[guest] done' in terminal,'Guest failed: '+eid)
    actual=machine(run)
    require(actual==json.loads((STATE/'expected-machine.json').read_text()),'Actual machine differs from reference')
    result_root=ROOT/'transformer_profiling/hsylin'/study(eid)
    manifest_path=result_root/'manifest.tsv'
    rows=table(manifest_path) if manifest_path.exists() else []
    selected=[x for x in rows if x['exp_id']==eid]
    if not selected:
        require(not (run/'collected_as').exists(),'Run was collected into another dataset')
        subprocess.run(['bash','./exp.sh','collect',ROW,'--run',str(run),'--study',study(eid),
                        '--exp-id',eid,'--output-root',str(result_root)],cwd=EXP,env=environment(config()),check=True)
        selected=[x for x in table(manifest_path) if x['exp_id']==eid]
    require(len(selected)==1,'Duplicate/missing result ID')
    row=selected[0]
    require(row['study']==study(eid) and row['repo_commit']==r['server_sha'][:12] and Path(row['stats_file'])==run/'stats.txt','Collected source/namespace mismatch')
    result=Path(row['output_file'])
    r.update(result_file=str(result),result_sha256=sha(result),machine_config=actual,
             host_elapsed_seconds_approx=(run/'gem5_stdout.log').stat().st_mtime-r['launch']['time'])
    save(path,r);print('Collected',eid)


def collect():
    with lock('collect'):
        checked()
        for path in sorted((STATE/'runs').glob('E*.json')):
            collect_one(path.stem)


def status():
    for eid in manifest()['actions']:
        path=STATE/'runs'/(eid+'.json')
        if path.exists():
            r=json.loads(path.read_text());state='RUNNING' if live(r['run_dir']) else ('COLLECTED' if 'result_file' in r else 'FINISHED: inspect/collect')
            print(eid,state,r['server_sha'],r['run_dir'])
        elif (STATE/'pending'/(eid+'.json')).exists(): print(eid,'PENDING: recover, never resubmit')
        else: print(eid,'PREPARED' if (STATE/'prepared'/(eid+'.json')).exists() else 'NOT SUBMITTED')


METRICS=['sim_seconds','cpu_cycles','instructions','ops','load_instructions','store_instructions',
         'dcache_demand_accesses','dcache_demand_misses','l2_demand_accesses','l2_demand_misses',
         'committed_branches','branch_mispredictions']
PAIRS=[('E01','E02'),('E02','E03'),('E03','E04'),('E04','E05'),('E03','E05'),
       ('E05','E06'),('E06','E07'),('E06','E08'),('E07','E09'),('E08','E09'),
       ('E05','E07'),('E05','E08'),('E05','E09'),('E03','E06'),('E03','E07'),('E03','E08'),('E03','E09')]


def read_result(eid, exports=None):
    path=exports/'runs'/eid/'run.json' if exports else STATE/'runs'/(eid+'.json')
    r=json.loads(path.read_text()); file=Path(r['result_file'])
    if exports:file=exports/'results'/study(eid)/file.name
    require(sha(file)==r['result_sha256'], 'Result checksum changed')
    rows=table(file)
    require(sum(x['row_kind']=='final_total' for x in rows)==1,'Expected one final_total')
    keys=['implementation','n_learners','codebook_size','sve_bits','model','d_q','d_seq','d_model','num_head','d_ff','compile_flags','cores','l1i','l1d','l2','overrides','runner_id']
    common={k:rows[0][k] for k in keys};phases={}
    for row in rows:
        require(row['exp_id']==eid and row['study']==study(eid) and r['server_sha'].startswith(row['repo_commit']),'Wrong row identity')
        require({k:row[k] for k in keys}==common,'Inconsistent table metadata')
        if row['row_kind'] not in ['interval_delta','final_total']:continue
        phase='BLOCK' if row['row_kind']=='final_total' else row['interval']
        dest=phases.setdefault(phase,{m:0.0 for m in METRICS})
        for metric in METRICS:
            value=row.get(metric,'NA')
            if value in ['', 'NA','N/A']:
                require(metric not in ['sim_seconds','cpu_cycles'],'Missing required counter')
                dest[metric]=None
            else:
                number=float(value);require(math.isfinite(number) and number>=0,'Invalid metric')
                if dest[metric] is not None:dest[metric]+=number
    return r,common,phases


def verify_comparison_controllers(base, cand, b, c, exports=None):
    if b['controller'] == c['controller']:
        return
    directory = exports if exports else STATE
    reference = json.loads((directory/'reference-controller.json').read_text())
    revision = json.loads((directory/'controller.json').read_text())
    require(base in ['E01','E02','E03','E04','E05'] and cand in ['E06','E07','E08','E09']
            and b['controller'] == reference and c['controller'] == revision,
            'Comparison controller is neither the frozen baseline nor this revision')
    # Only the explicitly preserved E01-E05 may cross the two tool versions.
    # Hardware, generated data, actual machine, ROI and checkpoint stay strict.


def compare(exports=None, output=None, only=None):
    output=output or (exports/'comparisons' if exports else STATE/'comparisons');output.mkdir(parents=True,exist_ok=True)
    summary=['# CB4 single-run comparisons','', 'Simulated ROI seconds; no median or repeat-stability claim. Host elapsed time is separate.','', '| Control | Candidate | Control s | Candidate s | Speedup | Time change |','|---|---|---:|---:|---:|---:|']
    for base,cand in PAIRS:
        if only and study(cand)!=only:continue
        paths=[exports/'runs'/e/'run.json' if exports else STATE/'runs'/(e+'.json') for e in [base,cand]]
        if any(not p.exists() or 'result_file' not in json.loads(p.read_text()) for p in paths):continue
        b,bc,bp=read_result(base,exports);c,cc,cp=read_result(cand,exports)
        require(bc==cc and bp.keys()==cp.keys(),'Different table configuration/ROI phases')
        for k in ['hardware','generated','machine_config','checkpoint']:
            require(b[k]==c[k],'Comparison fingerprint mismatch: '+k)
        verify_comparison_controllers(base, cand, b, c, exports)
        result=[]
        for phase in bp:
            x,y=bp[phase],cp[phase]
            require(x['sim_seconds']>0 and y['sim_seconds']>0,'Zero runtime')
            row=dict(baseline=base,candidate=cand,phase=phase,speedup=x['sim_seconds']/y['sim_seconds'])
            for metric in METRICS:
                row['baseline_'+metric]=x[metric];row['candidate_'+metric]=y[metric]
                row[metric+'_change_percent']=100*(y[metric]/x[metric]-1) if x[metric] not in (None,0) and y[metric] is not None else 'NA'
            result.append(row)
        with (output/(base+'--'+cand+'.tsv')).open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(result[0]),delimiter='\t');writer.writeheader();writer.writerows(result)
        x,y=bp['BLOCK']['sim_seconds'],cp['BLOCK']['sim_seconds']
        summary.append(f'| {base} | {cand} | {x:.6f} | {y:.6f} | {x/y:.6f}× | {100*(y/x-1):+.4f}% |')
    (output/'README.md').write_text('\n'.join(summary)+'\n');print('\n'.join(summary))


def export(output):
    with lock('collect'):
        checked();output.mkdir(parents=True,exist_ok=True)
        for name in ['manifest.json','controller.json','reference-controller.json','hardware.json','generated-data.json','baseline-gate.json']:
            if (STATE/name).exists():shutil.copy2(STATE/name,output/name)
        for name in ['cache_study.py','cache_correctness.py']:
            (output/'tools').mkdir(exist_ok=True);shutil.copy2(EXP/name,output/'tools'/name)
        for path in sorted((STATE/'runs').glob('E*.json')):
            r=json.loads(path.read_text())
            if 'result_file' not in r:continue
            dest=output/'runs'/path.stem;dest.mkdir(parents=True,exist_ok=True)
            shutil.copy2(path,dest/'run.json')
            for name in ['stats.txt','config.ini','build_config.tsv','provenance.tsv','gem5_profile_regions.tsv','gem5_stdout.log','system.terminal','collected_as','collect.log','launch.json','study-action.json']:
                shutil.copy2(Path(r['run_dir'])/name,dest/name)
            shutil.copytree(ROOT/'transformer_profiling/hsylin'/study(path.stem),output/'results'/study(path.stem),dirs_exist_ok=True)
        compare(exports=output)
        files=[p for p in sorted(output.rglob('*')) if p.is_file() and p.name!='SHA256SUMS']
        (output/'SHA256SUMS').write_text(''.join(sha(p)+'  '+str(p.relative_to(output))+'\n' for p in files))
        print('Exported',output)


def main():
    p=argparse.ArgumentParser(description=__doc__);subs=p.add_subparsers(dest='cmd',required=True)
    v=subs.add_parser('verify');v.add_argument('--manifest',type=Path,required=True)
    for name in ['prepare','submit','recover']:
        s=subs.add_parser(name);s.add_argument('eid')
    s=subs.add_parser('status');s.add_argument('--all',action='store_true')
    s=subs.add_parser('collect');s.add_argument('--finished',action='store_true')
    s=subs.add_parser('compare');s.add_argument('--all',action='store_true');s.add_argument('--study');s.add_argument('--exports',type=Path);s.add_argument('--out',type=Path)
    s=subs.add_parser('export');s.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    if a.cmd=='verify':verify(a.manifest)
    elif a.cmd=='prepare':prepare(a.eid)
    elif a.cmd=='submit':submit(a.eid)
    elif a.cmd=='recover':
        with lock('submit'):
            checked();require(not (STATE/'runs'/(a.eid+'.json')).exists(),'Already recorded');record(a.eid)
    elif a.cmd=='status':status()
    elif a.cmd=='collect':collect()
    elif a.cmd=='compare':compare(a.exports,a.out,a.study)
    elif a.cmd=='export':export(a.out)


if __name__=='__main__':main()
