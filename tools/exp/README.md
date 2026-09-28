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
./exp.sh build-libm5                   # REQUIRED: gem5's libm5.a for the guest
./exp.sh patch-gem5                    # only needed for sve_bits != 128 or cache-size overrides
```

`build-libm5` compiles `$GEM5_ROOT/util/m5` for arm64 with the same cross
toolchain the transformer is built with, producing
`build/arm64/out/libm5.a`. The guest binary links it statically so
`profile.cc` issues `m5_dump_stats()` / `m5_reset_stats()` directly.

`patch-gem5` adds `--sve-vl`, `--l1i-size`, `--l1d-size`, `--l2-size` options
to the gem5 tree's `starter_fs.py` (idempotent; original kept as `.orig`).
Without it, 128-bit runs with stock cache sizes work as-is.

## Why libm5 is not optional for measurement

The fallback is `std::system("m5 dumpstats")` at every region boundary. The
fork of `/bin/sh` and the two `exec`s happen **before** the m5 op executes, so
those cycles land in the region that is about to be dumped; the shell also
evicts L1/L2 and TLB entries, so the *next* region starts cold and pays the
misses. The overhead is roughly constant per boundary, which means it distorts
small regions (`Projection`, `non_GEMM_*`) far more than large ones — it
changes the ratios between regions, not just the totals.

`USE_LIBM5=0` in `runner.conf` restores the shell path for a machine with no
gem5 sources. **Runs built with `USE_LIBM5=1` and `USE_LIBM5=0` are not
comparable.** `build_config.tsv` records `I2CE_USE_LIBM5_FLAG` inside
`compile_flags`, plus `libm5_sha256`, so which path a run used is always
recoverable from the run itself.

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
./exp.sh status           # gem5 processes + one row per run (see States below)
./exp.sh results 39       # where the latest run's logs/stats are
./exp.sh collect 39       # newest finished run -> tables + refreshed HTML report
./exp.sh collect 39 --run 20260927_231044   # ... or a specific run
./exp.sh report           # all collected experiments -> one offline HTML
./exp.sh gc-src           # drop the worktrees --at created
./exp.sh gc-builds        # drop build directories no run points at
```

### States reported by `status`

`status` prints one row per *run*, not per experiment id: every run still in
flight, plus the newest row of each id. Several runs of one id can be live at
once — that is the point of per-build shares — and a view that showed only the
last row hid the rest. The commit column is the commit the binary was built
from, so two rows of the same id are told apart by the thing that differs.

| state | meaning |
|---|---|
| `RUNNING` | a gem5 process is alive for this output directory |
| `DONE -> collect <id>` | `stats.txt` and `gem5_profile_regions.tsv` are both written; ready to harvest (with `--run <ts>` when the id has more than one row shown) |
| `COLLECTED E<n>` | already harvested, and which table row it became |
| `SMOKE-OK` | a `smoke` run that reached its end |
| `INCOMPLETE` | no live process and no complete output — look at `gem5_stdout.log` and `system.terminal` |

`RUNNING` is decided first by a live process whose command line carries this
run's `-d` directory. Failing that, and only for a run that was never harvested,
the pid recorded at launch counts if it still belongs to the gem5 binary — on its
own that pid is not trusted, because over a ten-hour simulation the kernel can
hand the number to something else.

`COLLECTED` comes from `collected_as`, which `collect` writes into the run
directory *after* `add_experiment.py` has returned 0, holding the assigned id.
It is deliberately not keyed on `provenance.tsv`: that file is an **input** to
`add_experiment.py`, so it exists before the harvest and survives a failed one.
Runs harvested before this marker existed are still recognised, by the run
timestamp their per-run table file carries.

### Running a specific commit

`build`, `submit`, `dryrun`, `checkpoint` and `run` accept `--at <commit-ish>`:

```bash
./exp.sh submit 38 --at v0.3        # the binary comes from that commit
./exp.sh submit 38                  # ... and this one from the current checkout
./exp.sh collect 38                 # both collected with today's tooling
```

The commit is built in a `git worktree` under `$EXP_ROOT/_src/`, so your working
tree is untouched and two commits can be built side by side. The runner,
`experiments.tsv` and `add_experiment.py` always come from the checkout you
invoked — **the measured code is the variable, the measuring apparatus is the
constant.** `--at` is refused on `collect` for that reason.

### Two commits under one experiment row

Every build gets its own directory, `$EXP_ROOT/<id>/builds/<timestamp>_<sha>/`,
and `$EXP_ROOT/<id>/share` is a symlink to the newest. That is what makes this
safe:

```bash
./exp.sh submit 38 --at shaA     # builds A, launches a run that mounts A
./exp.sh submit 38 --at shaB     # builds B; A's run is untouched
```

A run's share is a live 9p filesystem: the guest executes `transformer.o` from
it, which Linux demand-pages for the whole run, and bind-mounts `weights/` out
of it. With one directory per experiment row the second build replaced both
under the first run. A launch also resolves the symlink to an absolute path, so
a later build retargeting it cannot move a running mount.

Two runs of *the same* build at once are still refused: both guests write
`gem5_profile_regions.tsv` into the shared directory and would overwrite each
other. Build again to get a fresh directory.

`./exp.sh gc-builds` deletes build directories no recorded run points at (each
run stores its own in `build_dir`), keeping the current one. Roughly 35 MB of
binary and weights per build, so it is worth running after a sweep.

The worktree is a sparse checkout of just what a build reads — roughly 60 MB
rather than the ~2 GB a full checkout of this repository costs, most of which
(`executable archive/`, `gem5-X-TiC-SAT/`, `weights/multiple_learner_outputs/`)
no build touches. `./exp.sh gc-src` removes them when done. On a git too old
for `sparse-checkout` the full tree is taken instead, with a warning.

Because `$EXP_ROOT/<id>/share` belongs to the experiment id and not to the run,
the next build of that id overwrites it. Each launch therefore snapshots
`build_config.tsv` into its own run directory, and `collect` reads the run's
copy — so a run still in flight when you build another commit is still
described by the build it actually used. Runs are also selected by name rather
than modification time, since collecting one writes into it.

A tree old enough to predate a build flag would otherwise be recorded as
something it is not, so `build` checks the binary against the configuration
before recording anything: with `USE_LIBM5=1` a binary that still contains the
`std::system("m5 ...")` string is rejected outright.

Long jobs are launched with `nohup setsid` (they survive disconnects), but run
`submit` itself inside `screen`/`tmux` when it has to regenerate artifacts or
boot a new checkpoint, so the build phase survives too.

## Harvesting results (collect)

When `status` shows `DONE`, `./exp.sh collect <id>` calls
`transformer_profiling/add_experiment.py` with the run's own recorded
parameters (`--stats … --n-learners … --codebook-size|--dense … --sve-bits …
--exp-id E<id>`, plus `--model/--dims` when the model dimensions were
overridden). Extra arguments pass through and win on conflict, e.g.:

Experiment IDs are assigned automatically (the next free one in
`manifest.tsv`), so there is nothing to remember; `--exp-id E38 --replace`
re-collects an existing one. Because IDs are automatic, collecting the same run
twice is caught by its stats file rather than its ID — pass `--force-new` if a
second row really is wanted.

Every row records how it was produced: `runner_id`, `repo_commit`,
`commit_subject`, `binary_sha256`, `compile_flags`, `overrides`, `cores` and
the three cache sizes. Two runs with identical parameters but different code
are therefore distinguishable, and the commit appears in the file name too. The
run's full `build_config.tsv` plus the gem5 command line is copied to
`final/provenance/<exp_id>.tsv`, so settings with no column of their own stay
recorded.

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
$EXP_ROOT/<id>/share/build_config.tsv  what the binary was ACTUALLY built with
                            (compile_flags, binary/libm5/codebooks sha256, commit)
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
