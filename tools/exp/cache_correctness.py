#!/usr/bin/env python3
"""QEMU functional check of the exact source and generated data of a prepared action.

Only the validation copy's weight directory and profiling flags are changed.
This is not a performance run and its wall time must never be reported as one.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import cache_study as study


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('eid');parser.add_argument('--compare',default='E03')
    args=parser.parse_args();study.study(args.eid);study.checked();study.gate_for(args.eid)
    prepared=json.loads((study.STATE/'prepared'/(args.eid+'.json')).read_text())
    build=Path(prepared['build_dir']);root=study.STATE/'correctness'/args.eid
    study.require(not root.exists(),'Validation already exists; inspect evidence before retrying')
    study.require(study.prepared_data(build)==prepared['generated'],'Prepared inputs changed')
    root.mkdir(parents=True);source=root/'source';source.mkdir()
    with (root/'source.tar').open('wb') as output:
        subprocess.run(['git','-C',str(study.ROOT),'archive',prepared['server_sha'],
                        'Full_NN','transformer_layers','accelerator','transformer.cpp',
                        'transformer.h','compile_transformer.sh'],stdout=output,check=True)
    with tarfile.open(root/'source.tar') as archive:
        for member in archive.getmembers():
            study.require(not member.issym() and not member.islnk() and not Path(member.name).is_absolute()
                          and '..' not in Path(member.name).parts,'Unsafe archive entry')
        archive.extractall(source)
    for path in (build/'headers').glob('*.h'):
        shutil.copy2(path,source/'Full_NN/gemm_definitions'/path.name)
    weights=root/'weights';shutil.copytree(build/'weights',weights)
    file=source/'transformer.cpp';text=file.read_text()
    needle='std::string dir_name = "/home/thu/TiC-SAT/weights";'
    study.require(text.count(needle)==1,'Unknown validation weights path')
    file.write_text(text.replace(needle,'std::string dir_name = '+json.dumps(str(weights))+';'))
    env=study.environment(study.config())
    env.update(USE_FP32_TRANSFORMER_FLAG='0',FULL_INTERLEAVED_PIPELINE_FLAG='1',
               SIMD_FLAG='1',RELOAD_WEIGHT_FLAG='1',USE_NOTEBOOK_GENERATED_WEIGHTS_FLAG='1',
               USE_CODEBOOK_GEMM_FLAG='1',ENABLE_CODEBOOK_REFERENCE_FLAG='0',
               ENABLE_DEBUG_PRINT_FLAG='0',PROFILE_GEMM_ONLY_FLAG='0',GEM5_PROFILE_REGIONS_FLAG='0',
               DENSE_NO_SIMD_BASELINE_FLAG='0',I2CE_USE_LIBM5_FLAG='0',CORE_NUM_FLAG='1',OMP_NUM_THREADS='1')
    with (root/'build.log').open('w') as log:
        subprocess.run(['bash','compile_transformer.sh'],cwd=source,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    with (source/'transformer.o').open('rb') as stdin,(root/'run.log').open('w') as log:
        subprocess.run(['/home/thu/opt/qemu-sve/bin/qemu-aarch64','-cpu',
                        'max,sve=on,sve-default-vector-length=16',str(source/'transformer.o')],
                       stdin=stdin,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    log=(root/'run.log').read_text(errors='replace')
    study.require(' Not loaded' not in log and 'Weight file size mismatch:' not in log,'Failed to load generated data')
    outputs=weights/'multiple_learner_outputs/c'
    tensors={str(p.relative_to(outputs)):study.sha(p) for p in sorted(outputs.rglob('*')) if p.is_file()}
    study.require(len(tensors)==84,'Expected all 84 pipeline tensors')
    if args.eid!=args.compare:
        expected=json.loads((study.STATE/'correctness'/args.compare/'result.json').read_text())
        study.require(tensors==expected['tensors'],'Output tensors differ from '+args.compare)
    result=dict(eid=args.eid,source_sha=prepared['server_sha'],passed=True,backend='qemu-sve128',
                tensors=tensors,input_generation=prepared['generated'],
                validation_binary_sha256=study.sha(source/'transformer.o'),
                validation_only_changes=['weights path','profiling disabled'],reference=args.compare)
    study.save(root/'result.json',result)
    print('PASS',args.eid,len(tensors),'tensor hashes;', 'reference' if args.eid==args.compare else 'equal to '+args.compare)


if __name__=='__main__':main()
