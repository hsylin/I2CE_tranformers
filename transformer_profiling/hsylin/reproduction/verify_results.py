#!/usr/bin/env python3
"""Verify preserved single-run evidence and print comparisons; never simulate."""
import csv
from decimal import Decimal, ROUND_HALF_UP
import gzip
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CODEBOOK = {'MHA_QKV', 'Projection', 'FF1', 'FF2'}
METRICS = [
    ('sim_seconds', 'simSeconds'), ('instructions', 'simInsts'), ('ops', 'simOps'),
    ('cpu_cycles', 'system.cpu_cluster.cpus.numCycles'),
    ('memory_references', 'system.cpu_cluster.cpus.commitStats0.numMemRefs'),
    ('load_instructions', 'system.cpu_cluster.cpus.commitStats0.numLoadInsts'),
    ('store_instructions', 'system.cpu_cluster.cpus.commitStats0.numStoreInsts'),
    ('dcache_demand_accesses', 'system.cpu_cluster.cpus.dcache.demandAccesses::total'),
    ('dcache_demand_misses', 'system.cpu_cluster.cpus.dcache.demandMisses::total'),
    ('icache_demand_accesses', 'system.cpu_cluster.cpus.icache.demandAccesses::total'),
    ('icache_demand_misses', 'system.cpu_cluster.cpus.icache.demandMisses::total'),
    ('l2_demand_accesses', 'system.cpu_cluster.l2.demandAccesses::total'),
    ('l2_demand_misses', 'system.cpu_cluster.l2.demandMisses::total'),
    ('committed_branches', 'system.cpu_cluster.cpus.branchPred.committed_0::total'),
    ('branch_mispredictions', 'system.cpu_cluster.cpus.branchPred.mispredicted_0::total'),
]


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def member(root, name):
    path = (root/name).resolve()
    require(path.is_relative_to(root.resolve()), 'Path outside publication: '+name)
    return path


def blocks(text):
    answer, current = [], None
    for line in text.splitlines():
        if 'Begin Simulation Statistics' in line:
            require(current is None, 'Nested statistics block')
            current = {}
        elif 'End Simulation Statistics' in line:
            require(current is not None, 'Unmatched statistics end')
            answer.append(current)
            current = None
        elif current is not None:
            parts = line.split('#', 1)[0].split()
            if len(parts) >= 2:
                try:
                    current[parts[0]] = Decimal(parts[1])
                except ArithmeticError:
                    pass
    require(current is None, 'Incomplete statistics block')
    return answer


def verify(root=ROOT):
    for line in (root/'SHA256SUMS').read_text().splitlines():
        expected, name = line.split('  ', 1)
        require(digest(member(root, name).read_bytes()) == expected, 'Checksum mismatch: '+name)
    records = json.loads((root/'manifest.json').read_text())['experiments']
    require(len(records) == 9 and {r['result_id'] for r in records} == {f'E{i:02}' for i in range(1, 10)}, 'Expected nine distinct experiments')
    receipts = {r['result_id']: json.loads((member(root, r['evidence_dir'])/'run.json').read_text()) for r in records}
    reference = receipts['E03']
    results = {}
    for rec in records:
        eid = rec['result_id']; run = receipts[eid]
        directory = member(root, rec['evidence_dir'])
        for key in ['server_sha', 'local_sha', 'source_tree', 'study']:
            require(run[key] == rec[key], eid+': identity mismatch '+key)
        require(run['controller']['sha'] == rec['controller_sha'], eid+': controller mismatch')
        for key in ['generated', 'hardware', 'machine_config', 'checkpoint']:
            require(run[key] == reference[key], eid+': undeclared cross-run difference '+key)
        correctness = run['correctness']
        require(correctness['passed'] and correctness['source_sha'] == run['server_sha'], eid+': wrong correctness source')
        require(len(correctness['tensors']) == 84 and correctness['tensors'] == reference['correctness']['tensors'], eid+': tensor mismatch')
        data = member(root, rec['result_file']).read_bytes()
        require(digest(data) == rec['result_sha256'] == run['result_sha256'], eid+': result hash mismatch')
        rows = list(csv.DictReader(data.decode().splitlines(), delimiter='\t'))
        require(len(rows) == 11 and len(rows[0]) == 52, eid+': unexpected table schema')
        for row in rows:
            require(row['exp_id'] == eid and row['study'] == run['study'], eid+': row identity mismatch')
            require(row['binary_sha256'] == rec['binary_sha256'] == run['build']['binary_sha256'], eid+': binary mismatch')
            require(run['server_sha'].startswith(row['repo_commit']), eid+': row source mismatch')
        for patch in rec['ordered_patch_files']:
            require(digest(member(root, patch['file']).read_bytes()) == patch['sha256'], eid+': patch hash mismatch')
        text = gzip.decompress((directory/'stats.txt.gz').read_bytes())
        require(digest(text) == rec['raw_stats_sha256'], eid+': raw statistics hash mismatch')
        raw = blocks(text.decode())
        require(len(raw) == 23, eid+': expected 22 ROI dumps plus one exit dump')
        require('m5_exit instruction encountered' in (directory/'gem5_stdout.log').read_text(), eid+': abnormal simulator exit')
        terminal = (directory/'system.terminal').read_text()
        require('[guest] exp 38 exit code: 0' in terminal and '[guest] done' in terminal, eid+': abnormal guest exit')
        labels = [line.split('\t')[2] for line in (directory/'gem5_profile_regions.tsv').read_text().splitlines()[2:]]
        require(len(labels) == 22, eid+': incorrect region map')
        phases = {p: {column: Decimal(0) for column, _ in METRICS} for p in labels}
        previous = {column: Decimal(0) for column, _ in METRICS}
        for block, label in zip(raw[:22], labels):
            seconds = block['simTicks']/block['simFreq']
            if 'simSeconds' not in block or abs(block['simSeconds']-seconds) > Decimal('0.000001'):
                block['simSeconds'] = seconds.quantize(Decimal('0.000001'), rounding=ROUND_HALF_UP)
            for column, stat in METRICS:
                require(stat in block, eid+': missing raw counter '+stat)
                value = block[stat]
                require(value >= previous[column], eid+': counter reset '+stat)
                phases[label][column] += value-previous[column]
                previous[column] = value
        for row in rows:
            require(row['row_kind'] in ['interval_delta', 'final_total'], eid+': unexpected row kind')
            expected = previous if row['row_kind'] == 'final_total' else phases[row['interval']]
            for column, _ in METRICS:
                value = expected[column]
                if column == 'sim_seconds':
                    value = value.quantize(Decimal('0.000001'), rounding=ROUND_HALF_UP)
                require(Decimal(row[column]) == value, eid+': raw/table mismatch '+row['interval']+' '+column)
        whole = previous['sim_seconds']
        codebook = sum(phases[p]['sim_seconds'] for p in CODEBOOK)
        require(whole == Decimal(str(rec['simulated_ROI_seconds'])), eid+': manifest runtime mismatch')
        results[eid] = {'whole': whole, 'codebook': codebook, 'baseline': rec['comparison_baseline_id']}
    print('PASS: nine single runs, 84 tensor hashes each, identities, fingerprints, raw counters and exits.')
    print('ID\twhole_s\tcodebook_s\tbaseline\twhole_speedup\tcodebook_speedup')
    for eid, result in results.items():
        base = results.get(result['baseline'])
        speed = ['NA', 'NA'] if base is None else [f"{base[k]/result[k]:.6f}" for k in ['whole', 'codebook']]
        print('\t'.join([eid, f"{result['whole']:.6f}", f"{result['codebook']:.6f}", str(result['baseline']), *speed]))
    return results


if __name__ == '__main__':
    verify()
