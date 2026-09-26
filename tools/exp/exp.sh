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
#   ./exp.sh collect <id> [args...]   finished run -> tables + refreshed HTML
#   ./exp.sh report [args...]        all collected experiments -> one offline HTML
#   ./exp.sh patch-gem5               add --sve-vl/--l1*-size/--l2-size to starter_fs.py
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
# Reporting is read-only with respect to experiments and needs no gem5 paths,
# machine-specific runner.conf, cross compiler, or running simulator.
if [[ "${1:-}" == report ]]; then
  shift
  exec "${REPORT_PYTHON:-python3}" "$REPO_ROOT/transformer_profiling/report.py" "$@"
fi
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
    codebook_int8|dense_int8) : ;;
    *) die "[exp $EID] impl '$IMPL' not supported (codebook_int8 or dense_int8)" ;;
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

  # 2. compile under the build lock (headers are swapped inside the repo)
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
    USE_FP32_TRANSFORMER_FLAG=0 FULL_INTERLEAVED_PIPELINE_FLAG=1 SIMD_FLAG=1 \
    RELOAD_WEIGHT_FLAG=1 USE_NOTEBOOK_GENERATED_WEIGHTS_FLAG=1 \
    USE_CODEBOOK_GEMM_FLAG=1 ENABLE_CODEBOOK_REFERENCE_FLAG=0 \
    ENABLE_DEBUG_PRINT_FLAG=0 PROFILE_GEMM_ONLY_FLAG=1 \
    GEM5_PROFILE_REGIONS_FLAG=1 DENSE_NO_SIMD_BASELINE_FLAG=0 \
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
  sha256sum "$SHARE/transformer.o" | tee "$SHARE/transformer.o.sha256"
  echo "[build $EID] share ready: $SHARE"
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
  local CPT; CPT="$(ls -d "$CPT_DIR"/cpt.* 2>/dev/null | head -1)"
  [[ -n "$CPT" ]] || die "[exp $EID] no checkpoint for sve$SVE cores=$CORES — run: ./exp.sh checkpoint $EID"
  [[ -f "$SHARE/transformer.o" ]] || die "[exp $EID] no binary — run: ./exp.sh build $EID"
  local TS OUT; TS="$(date +%Y%m%d_%H%M%S)"; OUT="$EDIR/${PREFIX}_$TS"
  mkdir -p "$OUT"
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
  local OUT; OUT="$(ls -dt "$EDIR"/smoke_* | head -1)"
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
collect_one() { # <id> [extra add_experiment.py args...] — extras win on conflict
  local ID="$1"; shift
  load_row "$ID"
  local AE="$REPO_ROOT/transformer_profiling/add_experiment.py"
  [[ -f "$AE" ]] || die "add_experiment.py not found at $AE (needs the submission branch)"
  local OUT; OUT="$(ls -dt "$EDIR"/out_* 2>/dev/null | head -1)" || true
  [[ -n "${OUT:-}" ]] || die "no measurement runs recorded for exp $EID"
  [[ -s "$OUT/stats.txt" && -s "$OUT/gem5_profile_regions.tsv" ]] \
    || die "exp $EID latest run is not finished ($OUT): needs non-empty stats.txt and gem5_profile_regions.tsv"

  local ARGS=( --stats "$OUT/stats.txt" --n-learners "$NL" --sve-bits "$SVE"
               --exp-id "E$EID" --study "Runner" )
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
  echo "[collect $EID] $OUT/stats.txt -> transformer_profiling/final"
  "$GEN_PYTHON" "$AE" "${ARGS[@]}" "$@"
  if [[ " $* " != *" --dry-run "* ]]; then
    local REPORT_INPUT="$REPO_ROOT/transformer_profiling/final" ARG PREVIOUS=""
    for ARG in "$@"; do
      if [[ "$PREVIOUS" == --output-root ]]; then REPORT_INPUT="$ARG"; fi
      case "$ARG" in --output-root=*) REPORT_INPUT="${ARG#*=}" ;; esac
      PREVIOUS="$ARG"
    done
    echo "[collect $EID] refreshing HTML report"
    if ! "${REPORT_PYTHON:-$GEN_PYTHON}" "$REPO_ROOT/transformer_profiling/report.py" \
      --input "$REPORT_INPUT" \
      --output "${REPORT_OUTPUT:-$REPO_ROOT/transformer_profiling/reports/profiling_report.html}"; then
      die "tables were saved, but HTML refresh failed. Fix the reported error and rerun exp.sh report --input '$REPORT_INPUT'."
    fi
    echo
    echo "[collect $EID] tables updated. To publish them:"
    echo "    cd $REPO_ROOT"
    echo "    git add transformer_profiling/final"
    echo "    git commit -m \"chore(profiling): add E$EID cb=$CB nl=$NL sve=$SVE run\""
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
  local OUT; OUT="$(ls -dt "$EDIR"/out_* 2>/dev/null | head -1)" || true
  [[ -n "${OUT:-}" ]] || die "no runs recorded for exp $EID"
  echo "latest run: $OUT"
  for f in stats.txt gem5_profile_regions.tsv gem5_stdout.log system.terminal config.ini; do
    if [[ -s "$OUT/$f" ]]; then printf '  %-28s %8s bytes\n' "$f" "$(stat -c%s "$OUT/$f")"
    else printf '  %-28s MISSING/EMPTY\n' "$f"; fi
  done
  [[ -s "$OUT/gem5_profile_regions.tsv" ]] && { echo "  --- region index ---"; sed 's/^/    /' "$OUT/gem5_profile_regions.tsv"; }
}

cmd_patch_gem5() {
  "$GEN_PYTHON" "$SELF_DIR/lib/apply_gem5_opts.py" "$GEM5_CWD/$GEM5_CFG"
}

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
  *) sed -n '3,31p' "$0"; exit 1 ;;
esac
