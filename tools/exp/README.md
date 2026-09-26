# tools/exp — config-driven gem5 experiment runner

One TSV row = one experiment. The runner regenerates build artifacts when the
configuration needs it, compiles an isolated binary, boots (or reuses) a
checkpoint, launches gem5 detached, records provenance, and feeds finished
runs into the `transformer_profiling` tables.

## Setup (once per machine)

```bash
cd tools/exp
cp runner.conf.example runner.conf     # edit paths for your machine
conda activate gem5_env                # aarch64 cross compiler; regeneration needs numpy+torch+tqdm
./exp.sh patch-gem5                    # only needed for sve_bits != 128 or cache-size overrides
```

`patch-gem5` adds `--sve-vl`, `--l1i-size`, `--l1d-size`, `--l2-size` options
to the gem5 tree's `starter_fs.py` (idempotent; original kept as `.orig`).
Without it, 128-bit runs with stock cache sizes work as-is.

## Parameters

| parameter | column / override key | kind | effect |
|---|---|---|---|
| codebook size | `cb` | **build** | notebook regenerates headers+weights, new binary |
| number of learners | `n_learners` | **build** | same |
| SVE vector length | `sve_bits` (128/256/512) | build **and** sim | sets notebook `SVE_LANES = bits/32` and gem5 `--sve-vl = bits/128` together |
| sequence length | `seq_len` | **build** | notebook `TRANSFORMER_SEQ_LEN` (default 512) |
| model dimension | `d_model` | **build** | `TRANSFORMER_D_MODEL` (256); head hidden size = d_model/num_heads |
| number of heads | `num_heads` | **build** | `TRANSFORMER_NUM_HEADS` (4) |
| FF hidden size | `d_ff` | **build** | `TRANSFORMER_D_FF` (1024) |
| per-head Q/K dim | `d_q` | **build** | `TRANSFORMER_D_Q` (64) |
| L1I / L1D / L2 size | `l1i` / `l1d` / `l2` | sim | gem5 cache sizes (stock 48KiB/32KiB/1MiB); same binary, same checkpoint |
| core count | `cores` | sim | gem5 `--num-cores`; checkpoints are keyed per (sve, cores) |

**Build** parameters are baked into notebook-generated headers and weights —
they are never runtime flags, so each configuration gets its own binary under
`$EXP_ROOT/<id>/share/`. **Sim** parameters only change the gem5 command line.
Non-default *model dimensions* are an untested path: run `./exp.sh smoke <id>`
and sanity-check a debug build before trusting numbers.

## Adding an experiment

Add one line to `experiments.tsv` (TAB-separated, `overrides` is `-` or
`key=value,key=value`):

```
39	8	4	128	codebook_int8	l1d=64KiB,l2=2MiB	bigger caches
40	4	2	256	codebook_int8	seq_len=256,d_model=128,num_heads=2	small model @ SVE-256
```

No script changes needed.

## Commands

```bash
./exp.sh list             # show the table
./exp.sh dryrun 39        # resolve every step incl. gem5 flags; touches nothing
./exp.sh smoke [id]       # end-to-end pipeline test in minutes (no full sim)
./exp.sh submit 37 38 39  # build + checkpoint + launch, throttled (MAX_PARALLEL)
./exp.sh status           # gem5 processes + last run per experiment (DONE/RUNNING)
./exp.sh results 39       # where the latest run's logs/stats are
./exp.sh collect 39       # finished run -> tables + refreshed HTML report
./exp.sh report           # all collected experiments -> one offline HTML
```

Long jobs are launched with `nohup setsid` (they survive disconnects), but run
`submit` itself inside `screen`/`tmux` when it has to regenerate artifacts or
boot a new checkpoint, so the build phase survives too.

## Harvesting results (collect)

When `status` shows `DONE`, `./exp.sh collect <id>` calls
`transformer_profiling/add_experiment.py` with the run's own recorded
parameters (`--stats … --n-learners … --codebook-size|--dense … --sve-bits …
--exp-id E<id>`, plus `--model/--dims` when the model dimensions were
overridden). Extra arguments pass through and win on conflict, e.g.:

```bash
./exp.sh collect 38 --study "Codebook-size scaling" --dry-run   # preview rows
./exp.sh collect 38 --study "Codebook-size scaling"             # write tables + HTML
cd ../.. && git add transformer_profiling/final \
  && git commit -m "chore(profiling): add E38 cb=4 nl=2 sve=128 run" && git push
```

## Interactive report

Install the plotting additions once in the existing `gem5_env`, then generate
all collected experiments with one command (shown from the repository root):

```bash
python -m pip install -r transformer_profiling/requirements-visualization.txt
bash tools/exp/exp.sh report
```

Output: `transformer_profiling/reports/profiling_report.html`. The English page
contains charts and controls only, with Plotly.js embedded for offline viewing.
Each invocation scans `transformer_profiling/final/` for individual experiment
TSVs and the combined table. Individual files take precedence for the same ID,
so a newly added `e37_*.tsv` is included even if the combined table is stale.
Reporting works without `runner.conf` or gem5. `REPORT_PYTHON` selects its Python.

Successful `collect` commands now regenerate the report after saving the tables;
`collect --dry-run` does not. Custom `--output-root` directories are respected.
Set `REPORT_OUTPUT` to choose the HTML destination during collection. A report
failure leaves collected tables intact, returns a nonzero status and prints how
to retry report generation without collecting the same experiment again.

When copying result files into `final/` directly, run `exp.sh report` again or
leave `exp.sh report --watch` running. Watch mode rebuilds on local result changes;
stop it with Ctrl-C. Fetch/pull remote results first and reload the generated HTML
in your browser. A previously downloaded HTML remains a snapshot.

Use `report --output /path/report.html`, `report --experiments E09 E37`, or
`report --help`. Optional `--tile-results`, `--scaling-results` and `--phase-results`
add measured tile sweeps, scaling, efficiency, bandwidth and implementation phases
to the same HTML. Files named `tile_results.csv`, `scaling_results.csv` and
`phase_results.csv` in the input directory are discovered automatically; tabs
appear only when measurements exist. See
[the profiling guide](../../transformer_profiling/README.md) for the complete
installation, data schemas, formulas, and source-data caveats.

## Output layout & provenance

```
$EXP_ROOT/<id>/share/       binary + weights snapshot + run.rcS (private 9p share)
$EXP_ROOT/<id>/out_<ts>/    one gem5 run: stats.txt, gem5_profile_regions.tsv,
                            gem5_stdout.log, system.terminal, config.ini
$EXP_ROOT/<id>/manifest.tsv one row per launch: timestamp, id, params, repo
                            commit(+dirty), binary sha256, checkpoint, outdir, pid
$EXP_ROOT/_cpt/sve<b>[_c<n>]/  boot checkpoints, shared per (sve_bits, cores)
$EXP_ROOT/_stage/<id>/      notebook stage: _generator.py, _generator.log, artifacts
```

## Concurrency safety

- Artifact generation + compilation serialize under `_locks/build.lock`; the
  repo's committed headers are restored after every build (and generators run
  with `PYTHONDONTWRITEBYTECODE=1` so tracked `__pycache__` files stay intact).
- Each run has its own outdir; the disk image is opened copy-on-write; the
  guest bind-mounts the experiment's own weight snapshot over any weights
  baked into the image, so concurrent runs cannot see each other's data.
- Checkpoints are only read at restore; cache-size and model-dim changes reuse
  them (the atomic boot has no caches), while a different `cores` value gets
  its own checkpoint directory automatically.

## Scheduling notes

No Slurm on this host, so `submit` uses `nohup setsid` with a `MAX_PARALLEL`
throttle and `STAGGER` seconds between launches. gem5 MultiSim was evaluated
and not adopted: it requires a parameterless config module, while this flow
needs per-run `--restore/--script/-d`.
