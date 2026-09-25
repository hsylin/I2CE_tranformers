#!/usr/bin/env python3
"""
Headless, parameterized execution of Full_NN/generators/Transformer_generator.ipynb.

The notebook is treated as data: its code cells are extracted, the parameter
assignments in the two configuration cells are overridden, and the result is
executed with plain python from inside Full_NN/generators/. No jupyter is
required — only the notebook's own imports (numpy, torch, tqdm; CPU suffices).

Always overridden:      N_LEARNERS, CODEBOOK_SIZE, SVE_LANES, TICSAT_REPO_ROOT
Optionally overridden:  TRANSFORMER_SEQ_LEN, TRANSFORMER_D_MODEL,
                        TRANSFORMER_NUM_HEADS, TRANSFORMER_D_FF, TRANSFORMER_D_Q
(the notebook derives HEAD_HIDDEN_SIZE = D_MODEL // NUM_HEADS by itself)

Generation is deterministic (the notebook sets SEED = 12), so identical
parameters always reproduce identical artifacts.

Outputs land under --stage-root:
    <stage>/Full_NN/gemm_definitions/*.h        headers for the build
    <stage>/weights/generated_from_notebook/    runtime weight .bin files

Usage:
  gen_artifacts.py --repo <clone> --stage-root <dir> \
      --n-learners 2 --codebook-size 4 [--sve-lanes 4] \
      [--seq-len 256 --d-model 128 --num-heads 2 --d-ff 512 --d-q 64] [--dry-run]
"""
import argparse
import json
import re
import subprocess
import sys
from pathlib import Path


def override_line(line: str, name: str, value: str) -> str:
    # Replace `NAME = <expr>` at statement start, keep any trailing comment.
    m = re.match(rf"^(\s*){name}\s*=\s*[^#]*(#.*)?$", line)
    if not m:
        return line
    indent, comment = m.group(1), (m.group(2) or "")
    sep = "  " if comment else ""
    return f"{indent}{name} = {value}{sep}{comment}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, help="repository clone root")
    ap.add_argument("--stage-root", required=True,
                    help="directory the notebook writes into (becomes TICSAT_REPO_ROOT)")
    ap.add_argument("--n-learners", type=int, required=True, choices=[1, 2, 4])
    ap.add_argument("--codebook-size", type=int, required=True,
                    choices=[2, 4, 8, 16, 32])
    ap.add_argument("--sve-lanes", type=int, default=4,
                    help="SVE register width / 32 bits: 4=128b, 8=256b, 16=512b")
    ap.add_argument("--seq-len", type=int, default=None,
                    help="override TRANSFORMER_SEQ_LEN (default: notebook value 512)")
    ap.add_argument("--d-model", type=int, default=None,
                    help="override TRANSFORMER_D_MODEL (default 256)")
    ap.add_argument("--num-heads", type=int, default=None,
                    help="override TRANSFORMER_NUM_HEADS (default 4)")
    ap.add_argument("--d-ff", type=int, default=None,
                    help="override TRANSFORMER_D_FF (default 1024)")
    ap.add_argument("--d-q", type=int, default=None,
                    help="override TRANSFORMER_D_Q (default 64)")
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    repo = Path(args.repo).resolve()
    stage = Path(args.stage_root).resolve()
    nb_path = repo / "Full_NN/generators/Transformer_generator.ipynb"
    if not nb_path.is_file():
        print(f"ERROR: notebook not found: {nb_path}", file=sys.stderr)
        return 2

    nb = json.loads(nb_path.read_text())
    cells = [c for c in nb["cells"] if c["cell_type"] == "code"]

    values = {
        "N_LEARNERS": str(args.n_learners),
        "CODEBOOK_SIZE": str(args.codebook_size),
        "SVE_LANES": str(args.sve_lanes),
        "TICSAT_REPO_ROOT": repr(str(stage)),
    }
    dims = {
        "TRANSFORMER_SEQ_LEN": args.seq_len,
        "TRANSFORMER_D_MODEL": args.d_model,
        "TRANSFORMER_NUM_HEADS": args.num_heads,
        "TRANSFORMER_D_FF": args.d_ff,
        "TRANSFORMER_D_Q": args.d_q,
    }
    for name, val in dims.items():
        if val is not None:
            values[name] = str(val)
    applied = {k: 0 for k in values}

    pieces = []
    for c in cells:
        lines = "".join(c["source"]).splitlines()
        out = []
        for ln in lines:
            for name, val in values.items():
                new = override_line(ln, name, val)
                if new != ln:
                    applied[name] += 1
                    ln = new
                    break
            out.append(ln)
        pieces.append("\n".join(out))

    missing = [k for k, n in applied.items() if n == 0]
    if missing:
        print(f"ERROR: could not find assignment(s) to override: {missing} "
              f"— notebook layout changed; refusing to run.", file=sys.stderr)
        return 3

    script = ("# auto-generated from Transformer_generator.ipynb — do not edit\n"
              + "\n\n# ----- next cell -----\n".join(pieces) + "\n")

    dim_note = " ".join(f"{k.replace('TRANSFORMER_', '')}={v}"
                        for k, v in dims.items() if v is not None)
    print(f"[gen] parameters: N_LEARNERS={args.n_learners} "
          f"CODEBOOK_SIZE={args.codebook_size} SVE_LANES={args.sve_lanes}"
          + (f" {dim_note}" if dim_note else ""))
    if args.dry_run:
        print(f"[gen] dry run — all {sum(applied.values())} overrides resolved; "
              f"nothing written")
        return 0

    stage.mkdir(parents=True, exist_ok=True)
    gen_py = stage / "_generator.py"
    gen_py.write_text(script)
    print(f"[gen] staged generator: {gen_py}")

    # The notebook imports helper modules that live next to it.
    res = subprocess.run(
        [args.python, str(gen_py)],
        cwd=repo / "Full_NN/generators",
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    (stage / "_generator.log").write_text(res.stdout)
    tail = "\n".join(res.stdout.splitlines()[-12:])
    print(tail)
    if res.returncode != 0:
        print(f"ERROR: generator exited {res.returncode} "
              f"(full log: {stage/'_generator.log'})", file=sys.stderr)
        return 4

    hdr = stage / "Full_NN/gemm_definitions/codebooks_def.h"
    if not hdr.is_file():
        print(f"ERROR: expected header missing after generation: {hdr}",
              file=sys.stderr)
        return 5
    # sanity: the generated defines must match what we asked for
    text = hdr.read_text()
    for macro, want in (("N_LEARNERS", args.n_learners),
                        ("CB_SIZE", args.codebook_size)):
        m = re.search(rf"#define\s+{macro}\s+(\d+)", text)
        got = int(m.group(1)) if m else None
        if got != want:
            print(f"ERROR: generated {macro}={got}, expected {want}",
                  file=sys.stderr)
            return 6
    print(f"[gen] OK: {hdr} has N_LEARNERS={args.n_learners} "
          f"CB_SIZE={args.codebook_size}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
