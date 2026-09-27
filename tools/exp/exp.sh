#!/usr/bin/env bash
#
# exp.sh — config-driven gem5 experiment runner for I2CE_tranformers.  (v2)
#
#   ./exp.sh list                     show the experiment table
#   ./exp.sh dryrun  <id>             resolve + print every step, touch nothing
#   ./exp.sh build   <id>             artifacts (notebook if needed) + binary -> share/
#   ./exp.sh checkpoint <id>          ensure a boot checkpoint for the id's (sve, cores)
#   ./exp.sh run     <id>             launch one measurement run (background)
#   ./exp.sh submit  <id...>          build+checkpoint+run several, throttled
#   ./exp.sh smoke   [id]             end-to-end pipeline test WITHOUT the 10 h sim
#   ./exp.sh status                   running gem5 jobs + last manifest rows
#   ./exp.sh results <id>             where the logs/stats of the latest run are
#   ./exp.sh collect <id> [--run <ts>] [args...]  a finished run -> the tables
#   ./exp.sh patch-gem5               add --sve-vl/--l1*-size/--l2-size to starter_fs.py
#   ./exp.sh build-libm5              build gem5's libm5.a for the aarch64 guest
#   ./exp.sh gc-src                   drop the worktrees `--at` created
#   ./exp.sh gc-builds                drop build directories no run points at
#
# Any of build/submit/dryrun/checkpoint/run also accepts `--at <commit-ish>`:
# the binary is built from a worktree of that commit, while the runner, the
# experiment table and the collector stay on the invoking checkout. That is what
# makes two runs comparable -- only the measured code changes.
#
# Parameters live in experiments.tsv, one row per experiment:
#   id  cb  n_learners  sve_bits  impl  overrides  notes
# overrides is "-" or comma-separated key=value with these keys:
#   BUILD-time (regenerate artifacts + rebuild binary):
#       seq_len d_model num_heads d_ff d_q        (BERT-mini defaults 512/256/4/1024/64)
#   SIM-time (gem5 flags only, binary unchanged):
#       l1i l1d l2   cache sizes, e.g. l1d=64KiB  (stock 48KiB/32KiB/1MiB)
#       cores        number of CPU cores (default 1)
# codebook_size / n_learners / sve / model dims are BUILD-time properties:
# they live in notebook-generated headers/weights, so each configuration gets
# its own generated artifacts and its own binary.
#
# Concurrency safety:
#   - artifact generation + compilation are serialized under one lock, and the
#     resulting binary/weights are snapshotted into $EXP_ROOT/<id>/share/
#   - each run gets its own gem5 outdir; the disk image is opened
#     copy-on-write by starter_fs.py; checkpoints are only read at restore
#   - boot checkpoints are cached per (sve_bits, cores) — cache-size and
#     model-dimension changes reuse them (atomic boot has no caches)
#   - each experiment's 9p share is private, so guest-side outputs
#     (gem5_profile_regions.tsv) cannot interleave across experiments.

set -euo pipefail

SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "$SELF_DIR/../.." && pwd)}"
CONF="$SELF_DIR/runner.conf"
[[ -f "$CONF" ]] || CONF="$SELF_DIR/runner.conf.example"
# shellcheck disable=SC1090
source "$CONF"

EXP_ROOT="${EXP_ROOT:-$HOME/i2ce/exp}"
GEM5_BIN="${GEM5_BIN:?runner.conf must set GEM5_BIN}"
GEM5_CWD="${GEM5_CWD:?runner.conf must set GEM5_CWD}"
GEM5_CFG="${GEM5_CFG:-configs/example/arm/starter_fs.py}"
KERNEL="${KERNEL:?runner.conf must set KERNEL}"
DISK="${DISK:?runner.conf must set DISK}"
RUN_CPU="${RUN_CPU:-minor}"
BOOT_CPU="${BOOT_CPU:-atomic}"
GEM5_MACHINE_ARGS="${GEM5_MACHINE_ARGS:---cpu-freq=4GHz --mem-size=2GiB --mem-type=DDR3_1600_8x8 --mem-channels=1}"
NUM_CORES="${NUM_CORES:-1}"
MAX_PARALLEL="${MAX_PARALLEL:-4}"
STAGGER="${STAGGER:-20}"
GEN_PYTHON="${GEN_PYTHON:-python3}"
# libm5: gem5's m5 ops as linked instructions instead of std::system("m5 ...").
# USE_LIBM5=0 falls back to the shell, which charges a guest fork + exec to
# every measured region -- only for a tree with no gem5 sources to build from.
GEM5_ROOT="${GEM5_ROOT:-$GEM5_CWD}"
LIBM5_A="${LIBM5_A:-$GEM5_ROOT/util/m5/build/arm64/out/libm5.a}"
M5_INC="${M5_INC:-$GEM5_ROOT/include}"
USE_LIBM5="${USE_LIBM5:-1}"
# Cache sizes gem5 uses when no override is passed. They are recorded with each
# run, so they must match the gem5 tree's configs/example/arm/devices.py; set
# them in runner.conf if that tree differs.
STOCK_L1I="${STOCK_L1I:-48KiB}"
STOCK_L1D="${STOCK_L1D:-32KiB}"
STOCK_L2="${STOCK_L2:-1MiB}"
# gem5 SysPaths (bootloader etc.) need M5_PATH; export a sane default.
M5_PATH="${M5_PATH:-$HOME/i2ce/gem5_resources}"
export M5_PATH
# The repo tracks Full_NN/generators/__pycache__; never rewrite it.
export PYTHONDONTWRITEBYTECODE=1
TABLE="$SELF_DIR/experiments.tsv"

die()  { echo "ERROR: $*" >&2; exit 1; }
note() { printf '  %-18s %s\n' "$1" "$2"; }
gem5_count() { pgrep -x "$(basename "$GEM5_BIN")" 2>/dev/null | wc -l; }

# Resolve the real gem5 pid for a given -d output directory. nohup/setsid can
# make $! the wrapper rather than gem5 itself, so match the actual gem5 process
# by its unique outdir argument and confirm its command name.
gem5_pid_for() { # $1 = outdir
  local p b; b="$(basename "$GEM5_BIN")"
  for p in $(pgrep -f -- "-d $1 " 2>/dev/null); do
    [[ "$(cat "/proc/$p/comm" 2>/dev/null)" == "$b" ]] && { echo "$p"; return 0; }
  done
  return 1
}

# Which live gem5, if any, has this directory mounted as its 9p share? The
# share is a filesystem a running guest reads from, so anything that rewrites
# one has to know whether a simulation is still looking at it.
run_using_share() { # $1 = build directory -> prints the pid of a live user
  local p b; b="$(basename "$GEM5_BIN")"
  for p in $(pgrep -f -- "--vio-9p=$1 " 2>/dev/null); do
    [[ "$(cat "/proc/$p/comm" 2>/dev/null)" == "$b" ]] && { echo "$p"; return 0; }
  done
  return 1
}

# ---------------------------------------------------------------- table
row_for() { # id -> tab row (id cb nl sve impl overrides [notes])
  awk -F'\t' -v id="$1" '!/^#/ && NF>=6 && $1==id {print; found=1} END{exit !found}' "$TABLE" \
    || die "experiment id '$1' not found in $TABLE (7 tab-separated columns required)"
}

load_row() {
  IFS=$'\t' read -r EID CB NL SVE IMPL OVR NOTES <<<"$(row_for "$1")"
  NOTES="${NOTES:-}"
  [[ "$CB"  =~ ^(2|4|8|16|32)$ ]] || die "[exp $EID] bad codebook_size '$CB'"
  [[ "$NL"  =~ ^(1|2|4)$      ]] || die "[exp $EID] bad n_learners '$NL'"
  [[ "$SVE" =~ ^(128|256|512)$ ]] || die "[exp $EID] bad sve_bits '$SVE'"
  case "$IMPL" in
    codebook_int8) : ;;
    dense_int8)
      die "[exp $EID] impl 'dense_int8' is not wired up: build_one always sets" \
          "USE_CODEBOOK_GEMM_FLAG=1 / DENSE_NO_SIMD_BASELINE_FLAG=0, so this row" \
          "would build a CODEBOOK binary and collect would then label the result" \
          "as dense. Add a real dense build path before using this impl." ;;
    *) die "[exp $EID] impl '$IMPL' not supported (codebook_int8)" ;;
  esac
  SVE_LANES=$(( SVE / 32 ))
  SVE_VL=$(( SVE / 128 ))

  # ---- overrides ----
  OV_SEQ_LEN="" OV_D_MODEL="" OV_NUM_HEADS="" OV_D_FF="" OV_D_Q=""
  OV_L1I="" OV_L1D="" OV_L2="" OV_CORES=""
  if [[ -n "$OVR" && "$OVR" != "-" ]]; then
    local kv k v
    local _kvs=()
    IFS=',' read -ra _kvs <<<"$OVR"
    for kv in ${_kvs[@]+"${_kvs[@]}"}; do
      k="${kv%%=*}"; v="${kv#*=}"
      if [[ "$kv" != *"="* || -z "$k" || -z "$v" ]]; then
        die "[exp $EID] bad override '$kv' (want key=value)"
      fi
      case "$k" in
        seq_len)   OV_SEQ_LEN="$v" ;;
        d_model)   OV_D_MODEL="$v" ;;
        num_heads) OV_NUM_HEADS="$v" ;;
        d_ff)      OV_D_FF="$v" ;;
        d_q)       OV_D_Q="$v" ;;
        l1i)       OV_L1I="$v" ;;
        l1d)       OV_L1D="$v" ;;
        l2)        OV_L2="$v" ;;
        cores)     OV_CORES="$v" ;;
        *) die "[exp $EID] unknown override key '$k' (known: seq_len d_model num_heads d_ff d_q l1i l1d l2 cores)" ;;
      esac
    done
  fi
  local d
  for d in "$OV_SEQ_LEN" "$OV_D_MODEL" "$OV_NUM_HEADS" "$OV_D_FF" "$OV_D_Q"; do
    if [[ -n "$d" && ! "$d" =~ ^[0-9]+$ ]]; then die "[exp $EID] model dims must be integers (got '$d')"; fi
  done
  for d in "$OV_L1I" "$OV_L1D" "$OV_L2"; do
    if [[ -n "$d" && ! "$d" =~ ^[0-9]+([kKmMgG]i?)?B$ ]]; then
      die "[exp $EID] bad cache size '$d' (use gem5 syntax, e.g. 32KiB, 64kB, 1MiB)"
    fi
  done
  if [[ -n "$OV_CORES" && ! "$OV_CORES" =~ ^[1-8]$ ]]; then die "[exp $EID] cores must be 1..8"; fi
  # The codebook GEMM path has no OpenMP/std::thread worker parallelism, and the
  # extractor reads single-core stat paths (system.cpu_cluster.cpus.*). Adding
  # simulated cores would change the machine without dividing the work and would
  # silently mis-extract per-core counters, so refuse until both are addressed.
  if [[ -n "$OV_CORES" && "$OV_CORES" != 1 ]]; then
    die "[exp $EID] cores=$OV_CORES refused: no worker parallelism is implemented" \
        "in the codebook path, and add_experiment.py reads single-core stat paths." \
        "Extra simulated cores would idle and the per-core counters would be wrong."
  fi

  # effective model dimensions (BERT-mini defaults from the notebook)
  D_Q_EFF="${OV_D_Q:-64}"; SEQ_EFF="${OV_SEQ_LEN:-512}"; DM_EFF="${OV_D_MODEL:-256}"
  NH_EFF="${OV_NUM_HEADS:-4}"; DFF_EFF="${OV_D_FF:-1024}"
  if (( DM_EFF % NH_EFF != 0 )); then die "[exp $EID] d_model ($DM_EFF) must be divisible by num_heads ($NH_EFF)"; fi
  for d in "$SEQ_EFF" "$DM_EFF" "$DFF_EFF" "$D_Q_EFF"; do
    if (( d % 4 != 0 )); then die "[exp $EID] seq_len/d_model/d_ff/d_q must be multiples of 4 (SA kernel dim), got $d"; fi
  done
  MODEL_IS_DEFAULT=1
  if [[ -n "$OV_SEQ_LEN$OV_D_MODEL$OV_NUM_HEADS$OV_D_FF$OV_D_Q" ]]; then MODEL_IS_DEFAULT=0; fi

  CORES="${OV_CORES:-$NUM_CORES}"
  EDIR="$EXP_ROOT/$EID"; SHARE="$EDIR/share"; STAGE="$EXP_ROOT/_stage/$EID"
  # checkpoint key: sve + cores (cache sizes and model dims do not affect the
  # atomic boot, so those share checkpoints). cores=1 keeps the legacy name.
  if [[ "$CORES" == 1 ]]; then CPT_DIR="$EXP_ROOT/_cpt/sve$SVE"
  else CPT_DIR="$EXP_ROOT/_cpt/sve${SVE}_c$CORES"; fi

  IS_DEFAULT_ARTIFACTS=0
  if [[ "$CB" == 8 && "$NL" == 4 && "$SVE" == 128 && "$MODEL_IS_DEFAULT" == 1 && "$IMPL" == codebook_int8 ]]; then
    IS_DEFAULT_ARTIFACTS=1
  fi
}

# gem5 CLI extras for this row. $1 = "boot" (no cache flags) or "run".
gem5_extra() {
  EXTRA=()
  if [[ "$SVE" != 128 ]]; then EXTRA+=("--sve-vl=$SVE_VL"); fi
  EXTRA+=("--num-cores=$CORES")
  if [[ "$1" == run ]]; then
    if [[ -n "$OV_L1I" ]]; then EXTRA+=("--l1i-size=$OV_L1I"); fi
    if [[ -n "$OV_L1D" ]]; then EXTRA+=("--l1d-size=$OV_L1D"); fi
    if [[ -n "$OV_L2"  ]]; then EXTRA+=("--l2-size=$OV_L2");  fi
  fi
}

gen_args() { # fills GEN_ARGS for gen_artifacts.py
  GEN_ARGS=( --repo "$REPO_ROOT" --stage-root "$STAGE"
             --n-learners "$NL" --codebook-size "$CB" --sve-lanes "$SVE_LANES"
             --python "$GEN_PYTHON" )
  if [[ -n "$OV_SEQ_LEN"   ]]; then GEN_ARGS+=( --seq-len   "$OV_SEQ_LEN"   ); fi
  if [[ -n "$OV_D_MODEL"   ]]; then GEN_ARGS+=( --d-model   "$OV_D_MODEL"   ); fi
  if [[ -n "$OV_NUM_HEADS" ]]; then GEN_ARGS+=( --num-heads "$OV_NUM_HEADS" ); fi
  if [[ -n "$OV_D_FF"      ]]; then GEN_ARGS+=( --d-ff      "$OV_D_FF"      ); fi
  if [[ -n "$OV_D_Q"       ]]; then GEN_ARGS+=( --d-q       "$OV_D_Q"       ); fi
}

# ------------------------------------------------------------- preflight
preflight() { # $1 = sve_bits (optional), then load_row's OV_* are consulted
  [[ -x "$GEM5_BIN" ]] || die "gem5 binary not executable: $GEM5_BIN"
  [[ -f "$GEM5_CWD/$GEM5_CFG" ]] || die "gem5 config missing: $GEM5_CWD/$GEM5_CFG"
  [[ -f "$KERNEL" ]] || die "kernel missing: $KERNEL"
  [[ -f "$DISK" ]] || die "disk image missing: $DISK"
  [[ -f "$REPO_ROOT/compile_transformer.sh" ]] || die "not a repo clone: $REPO_ROOT"
  grep -q "CowDiskImage" "$GEM5_CWD/$GEM5_CFG" \
    || die "$GEM5_CFG does not open the disk copy-on-write; concurrent runs would be unsafe"
  if [[ "${1:-}" != "" && "$1" != 128 ]]; then
    grep -q -- "--sve-vl" "$GEM5_CWD/$GEM5_CFG" \
      || die "sve_bits=$1 needs the --sve-vl option — run: ./exp.sh patch-gem5"
  fi
  if [[ -n "${OV_L1I:-}${OV_L1D:-}${OV_L2:-}" ]]; then
    grep -q -- "--l1d-size" "$GEM5_CWD/$GEM5_CFG" \
      || die "cache-size overrides need the --l1*/--l2-size options — run: ./exp.sh patch-gem5"
  fi
}

# ---------------------------------------------------------------- build
build_one() {
  load_row "$1"; preflight "$SVE"

  # One share per BUILD, not per experiment row. The share is a live 9p mount
  # for as long as a run lasts, so a single directory per row meant rebuilding
  # replaced the binary and weights a running guest was reading -- which is
  # exactly what comparing two commits under one row asks for. A build gets a
  # directory of its own, never reused, and $EDIR/share points at the latest.
  local BUILD_ID
  BUILD_ID="$(date +%Y%m%d_%H%M%S)_$(git -C "$REPO_ROOT" rev-parse --short=8 HEAD 2>/dev/null || echo nogit)"
  git -C "$REPO_ROOT" diff --quiet 2>/dev/null || BUILD_ID="${BUILD_ID}d"

  # A pre-existing plain directory is from before this change; keep it (older
  # runs fall back to its build_config.tsv) but get it out of the way first.
  if [[ -d "$EDIR/share" && ! -L "$EDIR/share" ]]; then
    local INUSE
    if INUSE="$(run_using_share "$EDIR/share")"; then
      die "[build $EID] $EDIR/share is the old shared directory and gem5 pid $INUSE is still using it. Wait for that run to finish; the next build will move it aside and switch to per-build directories."
    fi
    mkdir -p "$EDIR/builds"
    mv "$EDIR/share" "$EDIR/builds/legacy_$(date +%Y%m%d_%H%M%S)"
    echo "[build $EID] moved the old shared directory into builds/ (one directory per build from now on)"
  fi

  SHARE="$EDIR/builds/$BUILD_ID"
  mkdir -p "$SHARE" "$STAGE" "$EXP_ROOT/_locks"

  # 1. artifacts
  if [[ "$IS_DEFAULT_ARTIFACTS" == 1 ]]; then
    echo "[build $EID] cb=$CB nl=$NL sve=$SVE model=default == committed defaults; using repo artifacts"
    HDR_SRC="$REPO_ROOT/Full_NN/gemm_definitions"
    WGT_SRC="$REPO_ROOT/weights/generated_from_notebook"
  else
    echo "[build $EID] generating artifacts (notebook, headless) ..."
    gen_args
    "$GEN_PYTHON" "$SELF_DIR/lib/gen_artifacts.py" "${GEN_ARGS[@]}"
    HDR_SRC="$STAGE/Full_NN/gemm_definitions"
    WGT_SRC="$STAGE/weights/generated_from_notebook"
  fi
  [[ -f "$HDR_SRC/codebooks_def.h" ]] || die "[exp $EID] headers missing at $HDR_SRC"
  [[ -d "$WGT_SRC" ]] || die "[exp $EID] weights missing at $WGT_SRC"

  # 2. compile under the build lock (headers are swapped inside the repo).
  # BUILD_FLAGS is a single source of truth: it is both applied to the compile
  # and recorded in build_config.tsv, so the manifest cannot drift from reality.
  BUILD_FLAGS="USE_FP32_TRANSFORMER_FLAG=0 FULL_INTERLEAVED_PIPELINE_FLAG=1 SIMD_FLAG=1"
  BUILD_FLAGS="$BUILD_FLAGS RELOAD_WEIGHT_FLAG=1 USE_NOTEBOOK_GENERATED_WEIGHTS_FLAG=1"
  BUILD_FLAGS="$BUILD_FLAGS USE_CODEBOOK_GEMM_FLAG=1 ENABLE_CODEBOOK_REFERENCE_FLAG=0"
  BUILD_FLAGS="$BUILD_FLAGS ENABLE_DEBUG_PRINT_FLAG=0 PROFILE_GEMM_ONLY_FLAG=1"
  BUILD_FLAGS="$BUILD_FLAGS GEM5_PROFILE_REGIONS_FLAG=1 DENSE_NO_SIMD_BASELINE_FLAG=0"
  BUILD_FLAGS="$BUILD_FLAGS I2CE_USE_LIBM5_FLAG=$USE_LIBM5"
  # Fail before the lock, with the remedy, rather than deep inside build.log.
  if [[ "$USE_LIBM5" == 1 ]]; then
    [[ -f "$LIBM5_A" ]] || die "[build $EID] libm5.a missing at $LIBM5_A -- run '$SELF_DIR/exp.sh build-libm5' first. (USE_LIBM5=0 in runner.conf falls back to std::system(\"m5 ...\"), which forks a guest shell inside every measured region; runs built either way are NOT comparable.)"
    [[ -f "$M5_INC/gem5/m5ops.h" ]] || die "[build $EID] gem5/m5ops.h missing under $M5_INC -- is GEM5_ROOT ($GEM5_ROOT) a gem5 source tree?"
  fi
  echo "[build $EID] compiling (serialized) ..."
  (
    flock -w 1800 9 || die "could not obtain build lock"
    cd "$REPO_ROOT"
    if [[ "$IS_DEFAULT_ARTIFACTS" != 1 ]]; then
      git diff --quiet -- Full_NN/gemm_definitions \
        || die "Full_NN/gemm_definitions has local modifications; refusing to swap headers"
      cp "$HDR_SRC"/*.h Full_NN/gemm_definitions/
    fi
    set +e
    env $BUILD_FLAGS GEM5_ROOT="$GEM5_ROOT" LIBM5_A="$LIBM5_A" M5_INC="$M5_INC" \
      bash ./compile_transformer.sh > "$EDIR/build.log" 2>&1
    RC=$?
    set -e
    if [[ "$IS_DEFAULT_ARTIFACTS" != 1 ]]; then
      git checkout -- Full_NN/gemm_definitions   # restore committed headers
    fi
    [[ $RC -eq 0 && -f transformer.o ]] \
      || die "[exp $EID] build failed (see $EDIR/build.log)"
    mv transformer.o "$SHARE/transformer.o"
    # the repository commits a transformer.o at the root; restore it so the
    # clone stays pristine (harmless no-op when it is untracked)
    git checkout -- transformer.o 2>/dev/null || true
  ) 9>"$EXP_ROOT/_locks/build.lock"

  # 3. snapshot weights + provenance into the private share
  rm -rf "$SHARE/weights"
  mkdir -p "$SHARE/weights"
  cp -a "$WGT_SRC" "$SHARE/weights/generated_from_notebook"
  cp "$HDR_SRC/codebooks_def.h" "$SHARE/codebooks_def.h.provenance"
  sed -e "s|@ID@|$EID|g" -e "s|@CB@|$CB|g" -e "s|@NL@|$NL|g" \
      -e "s|@SVE@|$SVE|g" -e "s|@SHARE@|$SHARE|g" \
      "$SELF_DIR/lib/run.rcS.tmpl" > "$SHARE/run.rcS"
  chmod +x "$SHARE/run.rcS"
  # Verify the binary matches the flags we are about to record. An older tree
  # (`--at <sha>`) can predate I2CE_USE_LIBM5_FLAG and ignore it completely,
  # which would otherwise be recorded as a libm5 run and quietly compared
  # against runs that really are one.
  if [[ "$USE_LIBM5" == 1 ]]; then
    grep -qa 'm5 dumpstats' "$SHARE/transformer.o" \
      && die "[build $EID] the tree at $(git -C "$REPO_ROOT" rev-parse --short HEAD) ignored I2CE_USE_LIBM5_FLAG and built the std::system(\"m5 ...\") path. Its numbers are not comparable with libm5 runs; rebuild from a commit that supports the flag, or set USE_LIBM5=0 for the whole comparison."
  else
    grep -qa 'm5 dumpstats' "$SHARE/transformer.o" \
      || die "[build $EID] USE_LIBM5=0 but the binary has no std::system(\"m5 ...\") call; the build did not do what the configuration says."
  fi
  sha256sum "$SHARE/transformer.o" | tee "$SHARE/transformer.o.sha256"

  # Provenance: record what this binary was ACTUALLY built with, so a later
  # edit to experiments.tsv can never change how an existing run is described.
  {
    # printf, not echo: bash's echo does not expand \t, which silently produced
    # a file with literal backslash-t and made the collect guard below match
    # nothing while still reporting success.
    printf 'exp_id\t%s\n'               "$EID"
    printf 'built_at\t%s\n'             "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    printf 'impl\t%s\n'                 "$IMPL"
    printf 'codebook_size\t%s\n'        "$CB"
    printf 'n_learners\t%s\n'           "$NL"
    printf 'sve_bits\t%s\n'             "$SVE"
    printf 'sve_lanes\t%s\n'            "$SVE_LANES"
    printf 'cores\t%s\n'                "$CORES"
    printf 'overrides\t%s\n'            "$OVR"
    printf 'model_dims\td_q=%s seq_len=%s d_model=%s num_heads=%s d_ff=%s\n' \
           "$D_Q_EFF" "$SEQ_EFF" "$DM_EFF" "$NH_EFF" "$DFF_EFF"
    printf 'default_artifacts\t%s\n'    "$IS_DEFAULT_ARTIFACTS"
    # 12 chars: unambiguous, and the same width add_experiment.py normalises
    # older full-sha records to, so the column reads uniformly.
    printf 'repo_commit\t%s\n' \
           "$(git -C "$REPO_ROOT" rev-parse --short=12 HEAD)$(git -C "$REPO_ROOT" diff --quiet || echo '+dirty')"
    printf 'repo_commit_full\t%s\n'  "$(git -C "$REPO_ROOT" rev-parse HEAD)"
    # Subject only, tabs and newlines flattened: this ends up in a TSV cell.
    printf 'commit_subject\t%s\n' \
           "$(git -C "$REPO_ROOT" log -1 --format=%s | tr -d '\n' | tr '\t' ' ')"
    printf 'compile_flags\t%s\n'        "$BUILD_FLAGS"
    printf 'binary_sha256\t%s\n'        "$(cut -d' ' -f1 "$SHARE/transformer.o.sha256")"
    if [[ "$USE_LIBM5" == 1 ]]; then
      printf 'libm5_sha256\t%s\n'       "$(sha256sum "$LIBM5_A" | cut -d' ' -f1)"
    fi
    printf 'codebooks_def_sha256\t%s\n' "$(sha256sum "$HDR_SRC/codebooks_def.h" | cut -d' ' -f1)"
    printf 'gem5_bin\t%s\n'             "$GEM5_BIN"
    printf 'gem5_cfg\t%s\n'             "$GEM5_CWD/$GEM5_CFG"
    printf 'build_dir\t%s\n'            "$SHARE"
  } > "$SHARE/build_config.tsv"
  if ! git -C "$REPO_ROOT" diff --quiet; then
    git -C "$REPO_ROOT" diff > "$SHARE/repo_uncommitted.diff"
    echo "[build $EID] WARNING: repo has uncommitted changes; saved $SHARE/repo_uncommitted.diff"
  fi
  # Point $EDIR/share at this build, atomically, so `run <id>` and anything
  # else that knows only the experiment finds the newest one.
  ln -sfn "$SHARE" "$EDIR/.share.new" && mv -Tf "$EDIR/.share.new" "$EDIR/share"
  echo "[build $EID] share ready: $SHARE"
  echo "[build $EID] $EDIR/share -> $BUILD_ID"
}

# ------------------------------------------------------------ checkpoint
checkpoint_one() {
  load_row "$1"; preflight "$SVE"
  local KEY="sve$SVE cores=$CORES"
  if compgen -G "$CPT_DIR/cpt.*" >/dev/null; then
    echo "[cpt $KEY] cached: $(ls -d "$CPT_DIR"/cpt.* | head -1)"
    return 0
  fi
  mkdir -p "$CPT_DIR"
  local BOOT_RCS="$GEM5_CWD/configs/boot/hack_back_ckpt.rcS"
  [[ -f "$BOOT_RCS" ]] || BOOT_RCS="$SELF_DIR/lib/boot_ckpt.rcS"
  gem5_extra boot
  echo "[cpt $KEY] booting ($BOOT_CPU CPU) to take a checkpoint (~10 min) ..."
  ( cd "$GEM5_CWD" && nohup setsid \
    "$GEM5_BIN" -d "$CPT_DIR" --stats-file=boot_stats.txt \
      "$GEM5_CFG" --cpu="$BOOT_CPU" --kernel="$KERNEL" --disk-image="$DISK" \
      --script="$BOOT_RCS" --vio-9p="$EXP_ROOT" --checkpoint \
      $GEM5_MACHINE_ARGS ${EXTRA[@]+"${EXTRA[@]}"} \
      > "$CPT_DIR/boot_stdout.log" 2>&1 < /dev/null & echo $! > "$CPT_DIR/boot.pid" )
  sleep 1
  local RP; if RP="$(gem5_pid_for "$CPT_DIR")"; then echo "$RP" > "$CPT_DIR/boot.pid"; fi
  local BPID; BPID="$(cat "$CPT_DIR/boot.pid")"
  echo "[cpt $KEY] boot pid $BPID — detached, survives disconnects"
  local waited=0
  while ! compgen -G "$CPT_DIR/cpt.*" >/dev/null; do
    if ! kill -0 "$BPID" 2>/dev/null; then
      die "boot gem5 exited without a checkpoint (see $CPT_DIR/boot_stdout.log)"
    fi
    sleep 20; waited=$((waited+20))
    if [[ $waited -ge 3600 ]]; then
      die "no checkpoint after 60 min — boot pid $BPID still running (see $CPT_DIR/boot_stdout.log)"
    fi
  done
  # checkpoint is on disk; give the boot a moment to finish, then stop it
  local grace=0
  while kill -0 "$BPID" 2>/dev/null && [[ $grace -lt 120 ]]; do sleep 10; grace=$((grace+10)); done
  if kill -0 "$BPID" 2>/dev/null; then
    echo "[cpt $KEY] checkpoint written; stopping leftover boot pid $BPID"
    kill "$BPID" 2>/dev/null || true
  fi
  echo "[cpt $KEY] done: $(ls -d "$CPT_DIR"/cpt.* | head -1)"
}

# ----------------------------------------------------------------- run
launch_one() { # $1=id  $2=rcS  $3=outdir-prefix  $4=cpu
  load_row "$1"
  local RCS="$2" PREFIX="$3" CPU="$4"
  # Resolve the symlink now: gem5 is given an absolute path, so a later build
  # retargeting $EDIR/share cannot move this run's mount out from under it.
  local SHARE_REAL
  SHARE_REAL="$(cd "$SHARE" 2>/dev/null && pwd -P)" \
    || die "[exp $EID] no build to run — run: ./exp.sh build $EID"
  SHARE="$SHARE_REAL"
  local BUSY
  if BUSY="$(run_using_share "$SHARE")"; then
    die "[exp $EID] gem5 pid $BUSY is already running from this build ($SHARE). Both guests write gem5_profile_regions.tsv into it and would overwrite each other. Build again to get a fresh directory, or wait."
  fi
  local CPT; CPT="$(ls -d "$CPT_DIR"/cpt.* 2>/dev/null | head -1)"
  [[ -n "$CPT" ]] || die "[exp $EID] no checkpoint for sve$SVE cores=$CORES — run: ./exp.sh checkpoint $EID"
  [[ -f "$SHARE/transformer.o" ]] || die "[exp $EID] no binary — run: ./exp.sh build $EID"
  local TS OUT; TS="$(date +%Y%m%d_%H%M%S)"; OUT="$EDIR/${PREFIX}_$TS"
  mkdir -p "$OUT"
  # Snapshot the build description into the run. $SHARE is per experiment id, so
  # the next build of this id -- a different commit via --at, or just a rebuild
  # -- overwrites it, and a run still in flight would afterwards be collected
  # against a build that is not its own.
  [[ -f "$SHARE/build_config.tsv" ]] && cp "$SHARE/build_config.tsv" "$OUT/build_config.tsv"
  printf '%s\n' "$SHARE" > "$OUT/build_dir"
  gem5_extra run
  ( cd "$GEM5_CWD" && nohup setsid \
    "$GEM5_BIN" -d "$OUT" --stats-file=stats.txt --dump-config=config.ini \
      "$GEM5_CFG" --cpu="$CPU" --kernel="$KERNEL" --disk-image="$DISK" \
      --restore="$CPT" --script="$RCS" --vio-9p="$SHARE" \
      $GEM5_MACHINE_ARGS ${EXTRA[@]+"${EXTRA[@]}"} \
      > "$OUT/gem5_stdout.log" 2>&1 < /dev/null & echo $! > "$OUT/pid" )
  # $! may be the setsid/nohup wrapper; resolve the real gem5 pid by its outdir
  sleep 1
  local RP; if RP="$(gem5_pid_for "$OUT")"; then echo "$RP" > "$OUT/pid"; fi
  local PID; PID="$(cat "$OUT/pid")"
  printf '%s\t%s\tcb=%s nl=%s sve=%s c=%s ovr=%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$TS" "$EID" "$CB" "$NL" "$SVE" "$CORES" "$OVR" \
    "$(git -C "$REPO_ROOT" rev-parse --short HEAD)$(git -C "$REPO_ROOT" diff --quiet || echo '+dirty')" \
    "$(cut -d' ' -f1 "$SHARE/transformer.o.sha256")" "$CPT" "$OUT" "$PID" \
    >> "$EDIR/manifest.tsv"
  echo "[run $EID] pid=$PID  out=$OUT"
}

run_one()   { launch_one "$1" "$EXP_ROOT/$1/share/run.rcS" out "$RUN_CPU"; }

smoke_one() {
  load_row "${1:-37}"
  build_one "$EID"
  checkpoint_one "$EID"
  sed -e "s|@ID@|$EID|g" -e "s|@SHARE@|$SHARE|g" \
      "$SELF_DIR/lib/smoke.rcS.tmpl" > "$SHARE/smoke.rcS"
  launch_one "$EID" "$SHARE/smoke.rcS" smoke "$BOOT_CPU"
  local OUT; OUT="$(ls -d "$EDIR"/smoke_* | sort | tail -1)"
  echo "[smoke] waiting (checkpoint restore + guest script, a few minutes) ..."
  local i=0
  while kill -0 "$(cat "$OUT/pid")" 2>/dev/null; do
    sleep 15; i=$((i+15)); [[ $i -ge 1200 ]] && die "smoke timed out after 20 min ($OUT)"
  done
  if [[ -f "$OUT/smoke_result.txt" ]] && grep -q smoke-ok "$OUT/smoke_result.txt"; then
    echo "[smoke] PASS  ($OUT)"
  else
    echo "[smoke] FAIL — inspect $OUT/gem5_stdout.log and $OUT/system.terminal"; exit 1
  fi
}

# --------------------------------------------------------------- collect
collect_one() { # <id> [--run <ts|dir>] [extra add_experiment.py args...]
  local ID="$1"; shift
  load_row "$ID"
  # --run selects a run other than the newest; everything else is passed through
  # to add_experiment.py, where it wins on conflict.
  local WANT_RUN="" _PASS=()
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --run)   WANT_RUN="${2:?usage: collect <id> --run <timestamp|outdir>}"; shift 2 ;;
      --run=*) WANT_RUN="${1#--run=}"; shift ;;
      *)       _PASS+=("$1"); shift ;;
    esac
  done
  set -- "${_PASS[@]+"${_PASS[@]}"}"

  local OUT
  if [[ -n "$WANT_RUN" ]]; then
    if   [[ -d "$WANT_RUN" ]];            then OUT="$WANT_RUN"
    elif [[ -d "$EDIR/out_$WANT_RUN" ]];  then OUT="$EDIR/out_$WANT_RUN"
    elif [[ -d "$EDIR/$WANT_RUN" ]];      then OUT="$EDIR/$WANT_RUN"
    else die "[collect $EID] no run '$WANT_RUN'. Available: $(cd "$EDIR" 2>/dev/null && echo out_* )"
    fi
  else
    # By name, not by mtime: collect writes provenance.tsv into the run it
    # harvests, which would otherwise make the run it just collected look like
    # the newest one and get picked again. out_YYYYMMDD_HHMMSS sorts correctly.
    OUT="$(ls -d "$EDIR"/out_* 2>/dev/null | sort | tail -1)" || true
    [[ -n "${OUT:-}" ]] || die "no measurement runs recorded for exp $EID"
  fi
  # Deliberately NOT $REPO_ROOT: with `--at <sha>` that points at a checkout of
  # some older commit. The code under test is the variable; the tool that
  # records results has to be the constant, or two runs are not comparable.
  local AE="$SELF_DIR/../../transformer_profiling/add_experiment.py"
  [[ -f "$AE" ]] || die "add_experiment.py not found at $AE (needs the submission branch)"

  # A row in experiments.tsv can be edited after a run. The binary that actually
  # produced the stats recorded what it was built with, so verify the table still
  # describes that build before writing anything into the results tables.
  # The run's own snapshot is authoritative; the share copy describes the most
  # recent build of this id, which may be a different commit entirely.
  local BC="$OUT/build_config.tsv"
  if [[ ! -f "$BC" ]]; then
    BC="$SHARE/build_config.tsv"
    [[ -f "$BC" ]] && echo "[collect $EID] NOTE: this run predates per-run build_config.tsv; falling back to $SHARE/build_config.tsv, which describes the LATEST build of exp $EID and may not be this run's" >&2
  fi
  # Defaults matter: a run predating a field records "-", which is visibly
  # different from a field that was recorded empty.
  local BC_COMMIT="-" BC_SUBJECT="-" BC_BINSHA="-" BC_FLAGS="-"
  if [[ -f "$BC" ]]; then
    local k v mismatch=0 checked=0
    while IFS=$'\t' read -r k v; do
      case "$k" in
        codebook_size) checked=$((checked+1)); [[ "$v" == "$CB"   ]] || { echo "  build says codebook_size=$v, table says $CB" >&2; mismatch=1; } ;;
        n_learners) checked=$((checked+1)); [[ "$v" == "$NL"   ]] || { echo "  build says n_learners=$v, table says $NL" >&2; mismatch=1; } ;;
        sve_bits) checked=$((checked+1)); [[ "$v" == "$SVE"  ]] || { echo "  build says sve_bits=$v, table says $SVE" >&2; mismatch=1; } ;;
        impl) checked=$((checked+1)); [[ "$v" == "$IMPL" ]] || { echo "  build says impl=$v, table says $IMPL" >&2; mismatch=1; } ;;
        overrides) checked=$((checked+1)); [[ "$v" == "$OVR"  ]] || { echo "  build says overrides=$v, table says $OVR" >&2; mismatch=1; } ;;
        repo_commit)    BC_COMMIT="$v" ;;
        commit_subject) BC_SUBJECT="$v" ;;
        binary_sha256)  BC_BINSHA="$v" ;;
        compile_flags)  BC_FLAGS="$v" ;;
      esac
    done < "$BC"
    [[ $mismatch -eq 0 ]] || die "[collect $EID] experiments.tsv no longer matches the binary that produced this run (see $BC). Refusing to mislabel the results."
    # A malformed file whose keys never match would otherwise "pass" silently.
    [[ $checked -ge 5 ]] || die "[collect $EID] only $checked of 5 fields could be read from $BC; the file is malformed and the labels cannot be trusted."
    # stderr: stdout carries the TSV that --dry-run prints, and a caller
    # redirecting it to a file must not get progress lines mixed in.
    echo "[collect $EID] build_config.tsv matches the table row ($checked fields verified)" >&2
  else
    echo "[collect $EID] WARNING: no build_config.tsv (run predates provenance recording); labels come from experiments.tsv and are unverified" >&2
  fi
  [[ -s "$OUT/stats.txt" && -s "$OUT/gem5_profile_regions.tsv" ]] \
    || die "exp $EID latest run is not finished ($OUT): needs non-empty stats.txt and gem5_profile_regions.tsv"

  # Everything the runner knows about this run, in one file committed beside the
  # tables: parameters that have no column of their own stay recoverable.
  local PROV="$OUT/provenance.tsv"
  local GEM5_ARGS
  GEM5_ARGS="$(sed -n 's/^command line: //p' "$OUT/gem5_stdout.log" 2>/dev/null | head -1)"
  {
    [[ -f "$BC" ]] && cat "$BC"
    printf 'collected_at\t%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    printf 'outdir\t%s\n'       "$OUT"
    printf 'gem5_args\t%s\n'    "${GEM5_ARGS:--}"
  } > "$PROV"

  # No --exp-id: add_experiment.py assigns the next free one, so ids never have
  # to be remembered. --runner-id keeps the link back to the experiments.tsv row.
  local ARGS=( --stats "$OUT/stats.txt" --n-learners "$NL" --sve-bits "$SVE"
               --study "Runner" --runner-id "$EID"
               --repo-commit "$BC_COMMIT" --commit-subject "$BC_SUBJECT"
               --binary-sha256 "$BC_BINSHA" --compile-flags "$BC_FLAGS"
               --overrides "$OVR" --cores "$CORES"
               --l1i "${OV_L1I:-$STOCK_L1I}" --l1d "${OV_L1D:-$STOCK_L1D}"
               --l2 "${OV_L2:-$STOCK_L2}"
               --provenance "$PROV" )
  case "$IMPL" in
    dense*) ARGS+=( --dense ) ;;
    *)      ARGS+=( --codebook-size "$CB" ) ;;
  esac
  if [[ "$MODEL_IS_DEFAULT" == 1 ]]; then
    ARGS+=( --model BERT-mini )
  else
    ARGS+=( --model "custom-q${D_Q_EFF}-s${SEQ_EFF}-m${DM_EFF}-h${NH_EFF}-f${DFF_EFF}"
            --dims "$D_Q_EFF" "$SEQ_EFF" "$DM_EFF" "$NH_EFF" "$DFF_EFF" )
  fi
  echo "[collect $EID] $OUT/stats.txt -> transformer_profiling/final" >&2
  "$GEN_PYTHON" "$AE" "${ARGS[@]}" "$@"
  if [[ " $* " != *" --dry-run "* ]]; then
    echo
    echo "[collect $EID] tables updated. To publish them:"
    echo "    cd $REPO_ROOT"
    echo "    git add transformer_profiling/final"
    echo "    git commit -m \"chore(profiling): add exp $EID cb=$CB nl=$NL sve=$SVE run at $BC_COMMIT\""
    echo "    git push"
  fi
}

# --------------------------------------------------------------- others
cmd_list() {
  awk -F'\t' '!/^#/ && NF>=6 {printf "  %-4s cb=%-3s learners=%-2s sve=%-4s %-14s %-28s %s\n",$1,$2,$3,$4,$5,$6,$7}' "$TABLE"
}

cmd_dryrun() {
  load_row "$1"; preflight "$SVE"
  echo "exp $EID: codebook_size=$CB n_learners=$NL sve_bits=$SVE impl=$IMPL overrides=$OVR"
  note "artifacts" "$([[ $IS_DEFAULT_ARTIFACTS == 1 ]] && echo 'committed defaults (no notebook)' || echo "regenerate via notebook -> $STAGE")"
  if [[ "$MODEL_IS_DEFAULT" != 1 ]]; then
    note "model dims" "d_q=$D_Q_EFF seq_len=$SEQ_EFF d_model=$DM_EFF num_heads=$NH_EFF d_ff=$DFF_EFF (UNTESTED path — smoke first)"
  fi
  note "binary"     "$SHARE/transformer.o"
  note "checkpoint" "$CPT_DIR (shared per sve+cores)"
  gem5_extra run
  note "gem5 extras" "${EXTRA[*]}"
  note "run outdir" "$EDIR/out_<timestamp>/"
  note "gem5"       "$GEM5_BIN --cpu=$RUN_CPU --restore=<cpt> --script=$SHARE/run.rcS --vio-9p=$SHARE"
  note "repo"       "$REPO_ROOT @ $(git -C "$REPO_ROOT" rev-parse --short HEAD)"
  note "collect"    "./exp.sh collect $EID   (after the run finishes)"
  if [[ "$IS_DEFAULT_ARTIFACTS" != 1 ]]; then
    gen_args
    "$GEN_PYTHON" "$SELF_DIR/lib/gen_artifacts.py" "${GEN_ARGS[@]}" --dry-run
  fi
  echo "dryrun OK — nothing was modified"
}

cmd_submit() {
  [[ $# -ge 1 ]] || die "usage: exp.sh submit <id> [id...]"
  for id in "$@"; do build_one "$id"; done
  for id in "$@"; do checkpoint_one "$id"; done
  for id in "$@"; do
    while [[ "$(gem5_count)" -ge "$MAX_PARALLEL" ]]; do
      echo "  $(gem5_count) gem5 running — waiting for a slot (cap $MAX_PARALLEL)"; sleep 30
    done
    run_one "$id"; sleep "$STAGGER"
  done
  echo; echo "submitted: $*   (watch: ./exp.sh status; harvest: ./exp.sh collect <id>)"
}

cmd_status() {
  echo "== gem5 processes =="
  ps -o pid,stat,etime,pcpu,args -C "$(basename "$GEM5_BIN")" 2>/dev/null | cut -c1-140 || echo "  none"
  echo; echo "== latest manifest rows =="
  local m ts id params commit sha cpt out pid state
  for m in "$EXP_ROOT"/*/manifest.tsv; do
    [[ -f "$m" ]] || continue
    IFS=$'\t' read -r ts id params commit sha cpt out pid < <(tail -1 "$m") || continue
    if [[ -s "$out/stats.txt" && -s "$out/gem5_profile_regions.tsv" ]]; then state="DONE -> collect $id"
    elif [[ -s "$out/smoke_result.txt" ]]; then state="SMOKE-OK"
    elif kill -0 "$pid" 2>/dev/null || gem5_pid_for "$out" >/dev/null; then state="RUNNING"
    else state="INCOMPLETE"; fi
    printf '  exp %-4s %-38s pid=%-8s %-18s %s\n' "$id" "$params" "$pid" "$state" "$out"
  done
}

cmd_results() {
  load_row "$1"
  local OUT; OUT="$(ls -d "$EDIR"/out_* 2>/dev/null | sort | tail -1)" || true
  [[ -n "${OUT:-}" ]] || die "no runs recorded for exp $EID"
  echo "latest run: $OUT"
  for f in stats.txt gem5_profile_regions.tsv gem5_stdout.log system.terminal config.ini; do
    if [[ -s "$OUT/$f" ]]; then printf '  %-28s %8s bytes\n' "$f" "$(stat -c%s "$OUT/$f")"
    else printf '  %-28s MISSING/EMPTY\n' "$f"; fi
  done
  [[ -s "$OUT/gem5_profile_regions.tsv" ]] && { echo "  --- region index ---"; sed 's/^/    /' "$OUT/gem5_profile_regions.tsv"; }
}

# Build tree for `--at <sha>`: a git worktree of that commit under $EXP_ROOT,
# sharing the main clone's object store. A plain `git checkout` would disturb
# whatever the user is editing and would make two commits impossible to build
# side by side.
src_tree_for() { # $1 = commit-ish  ->  prints the worktree path
  local ref="$1" sha short dir
  sha="$(git -C "$REPO_ROOT_MAIN" rev-parse --verify "${ref}^{commit}" 2>/dev/null)" \
    || die "not a commit in $REPO_ROOT_MAIN: $ref"
  short="${sha:0:12}"
  dir="$EXP_ROOT/_src/$short"
  if [[ -e "$dir/.git" ]]; then echo "$dir"; return 0; fi
  mkdir -p "$EXP_ROOT/_src"

  # A full checkout of this repository is ~2 GB, most of it material a build
  # never reads ("executable archive", gem5-X-TiC-SAT, the multi-learner
  # outputs). Check out only what compile_transformer.sh and the generator
  # touch -- about 60 MB -- so several commits can coexist on a full disk.
  local sparse=(
    /compile_transformer.sh /transformer.cpp /transformer.h
    "/transformer_layers/**" "/accelerator/**" "/Full_NN/**"
    "/weights/generated_from_notebook/**"
  )
  if git -C "$REPO_ROOT_MAIN" sparse-checkout list >/dev/null 2>&1 \
     || git -C "$REPO_ROOT_MAIN" sparse-checkout --help >/dev/null 2>&1; then
    git -C "$REPO_ROOT_MAIN" worktree add --no-checkout --detach "$dir" "$sha" >&2 \
      || die "could not create a worktree for $sha at $dir"
    if git -C "$dir" sparse-checkout set --no-cone "${sparse[@]}" >&2 \
       && git -C "$dir" checkout >&2; then
      :
    else
      # Older git without usable sparse-checkout: take the whole tree instead
      # of failing, and say what that costs.
      echo "[--at] sparse checkout unavailable; checking out the full tree (~2 GB)" >&2
      git -C "$dir" sparse-checkout disable >/dev/null 2>&1 || true
      git -C "$dir" checkout >&2 || die "could not check out $sha at $dir"
    fi
  else
    echo "[--at] this git has no sparse-checkout; the worktree will be ~2 GB" >&2
    git -C "$REPO_ROOT_MAIN" worktree add --detach "$dir" "$sha" >&2 \
      || die "could not create a worktree for $sha at $dir"
  fi
  [[ -f "$dir/compile_transformer.sh" ]] \
    || die "worktree at $dir has no compile_transformer.sh; that commit cannot be built"
  echo "$dir"
}

# Build directories accumulate: one per build, ~35 MB of binary and weights.
# Anything a recorded run points at, and the current $EDIR/share, is kept.
cmd_gc_builds() {
  local e d real target ref removed=0 kept=0
  for e in "$EXP_ROOT"/*; do
    [[ -d "$e/builds" ]] || continue
    target=""
    [[ -L "$e/share" ]] && target="$(cd "$e/share" 2>/dev/null && pwd -P)"
    local refs=""
    for ref in "$e"/out_*/build_dir "$e"/smoke_*/build_dir; do
      [[ -f "$ref" ]] && refs="$refs$(cat "$ref")"$'\n'
    done
    for d in "$e"/builds/*; do
      [[ -d "$d" ]] || continue
      real="$(cd "$d" 2>/dev/null && pwd -P)" || continue
      if [[ "$real" == "$target" ]] || grep -qxF "$real" <<<"$refs" \
         || run_using_share "$real" >/dev/null; then
        kept=$((kept+1)); continue
      fi
      echo "[gc-builds] removing $d"
      rm -rf "$d"; removed=$((removed+1))
    done
  done
  echo "[gc-builds] removed $removed, kept $kept"
}

cmd_gc_src() {
  compgen -G "$EXP_ROOT/_src/*" >/dev/null || { echo "[gc-src] nothing to remove"; return 0; }
  local d
  for d in "$EXP_ROOT"/_src/*; do
    [[ -d "$d" ]] || continue
    echo "[gc-src] removing $d"
    git -C "$REPO_ROOT_MAIN" worktree remove --force "$d" 2>/dev/null || rm -rf "$d"
  done
  git -C "$REPO_ROOT_MAIN" worktree prune
  echo "[gc-src] done"
}

cmd_patch_gem5() {
  "$GEN_PYTHON" "$SELF_DIR/lib/apply_gem5_opts.py" "$GEM5_CWD/$GEM5_CFG"
}

# Build gem5's static m5 op library for the guest, with the SAME cross toolchain
# compile_transformer.sh will use, so archive and binary come from one compiler.
cmd_build_libm5() {
  [[ -f "$GEM5_ROOT/util/m5/SConstruct" ]] \
    || die "no $GEM5_ROOT/util/m5/SConstruct -- set GEM5_ROOT to a gem5 source tree"
  command -v scons >/dev/null 2>&1 \
    || die "scons not found; activate the environment gem5 itself was built in"

  local cxx prefix
  cxx="${A64CXX:-}"
  if [[ -z "$cxx" ]]; then
    if [[ -n "${CONDA_PREFIX:-}" && -x "$CONDA_PREFIX/bin/aarch64-conda-linux-gnu-g++" ]]; then
      cxx="$CONDA_PREFIX/bin/aarch64-conda-linux-gnu-g++"
    elif command -v aarch64-linux-gnu-g++ >/dev/null 2>&1; then
      cxx="$(command -v aarch64-linux-gnu-g++)"
    elif command -v aarch64-conda-linux-gnu-g++ >/dev/null 2>&1; then
      cxx="$(command -v aarch64-conda-linux-gnu-g++)"
    else
      die "no aarch64 C++ compiler found; set A64CXX to your cross compiler"
    fi
  fi
  prefix="${cxx%g++}"
  [[ -x "${prefix}gcc" ]] || die "expected ${prefix}gcc next to $cxx; set A64CXX"

  echo "[libm5] gem5 tree : $GEM5_ROOT"
  echo "[libm5] toolchain : ${prefix}gcc"
  ( cd "$GEM5_ROOT/util/m5" \
    && scons "arm64.CROSS_COMPILE=$prefix" build/arm64/out/libm5.a )
  [[ -f "$LIBM5_A" ]] || die "scons reported success but $LIBM5_A is missing"
  echo "[libm5] ok: $(sha256sum "$LIBM5_A")"
  echo "[libm5] binaries built from here on issue m5 ops inline; runs are NOT"
  echo "        comparable with runs built before this. Re-baseline."
}

# ------------------------------------------------------- --at <commit-ish>
# Pulled out before the subcommand dispatch so it can appear anywhere.
REPO_ROOT_MAIN="$REPO_ROOT"
AT_COMMIT=""
_ARGV=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --at)   AT_COMMIT="${2:?usage: --at <commit-ish>}"; shift 2 ;;
    --at=*) AT_COMMIT="${1#--at=}"; shift ;;
    *)      _ARGV+=("$1"); shift ;;
  esac
done
set -- "${_ARGV[@]+"${_ARGV[@]}"}"

if [[ -n "$AT_COMMIT" ]]; then
  case "${1:-}" in
    build|submit|dryrun|checkpoint|run) : ;;
    *) die "--at applies to build/submit/dryrun/checkpoint/run, not '${1:-}'. Results are always collected with the current tooling." ;;
  esac
  REPO_ROOT="$(src_tree_for "$AT_COMMIT")"
  echo "[--at] $AT_COMMIT -> building from $REPO_ROOT"
  echo "[--at] $(git -C "$REPO_ROOT" log -1 --format='%h %s')"
fi

# ---------------------------------------------------------------- main
case "${1:-}" in
  list)        cmd_list ;;
  dryrun)      cmd_dryrun "${2:?usage: exp.sh dryrun <id>}" ;;
  build)       build_one "${2:?usage: exp.sh build <id>}" ;;
  checkpoint)  checkpoint_one "${2:?usage: exp.sh checkpoint <id>}" ;;
  run)         shift; for id in "$@"; do run_one "$id"; done ;;
  submit)      shift; cmd_submit "$@" ;;
  smoke)       smoke_one "${2:-37}" ;;
  status)      cmd_status ;;
  results)     cmd_results "${2:?usage: exp.sh results <id>}" ;;
  collect)     shift; [[ $# -ge 1 ]] || die "usage: exp.sh collect <id> [add_experiment.py args...]"; collect_one "$@" ;;
  patch-gem5)  cmd_patch_gem5 ;;
  build-libm5) cmd_build_libm5 ;;
  gc-src)      cmd_gc_src ;;
  gc-builds)   cmd_gc_builds ;;
  *) sed -n '3,39p' "$0"; exit 1 ;;
esac
