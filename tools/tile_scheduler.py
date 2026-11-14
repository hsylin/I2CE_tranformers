#!/usr/bin/env python3
"""Compute the frozen v9 hierarchical tile policy, without fitting or lookup.

Required: --l1 SIZE --l2 SIZE --cb 4|8|16 --learners 1|2|4.
Default output: s1 o1 k1 s2 o2 k2, with no status text on stderr.
Use --json for calculations and limitations, or --verbose for status on stderr.
Values are capacity proposals; they do not establish backend support or speed.
"""
import argparse
import hashlib
import json
import re
import sys

FORMULA_SOURCE_SHA256 = (
    "2cba8531851ad94c7926a2cfd7cbb92e03f049f3d1b251df8602057199084db4"
)
# v9 names the validation study; the frozen implementation calls itself v8.
VERSION = "false-hierarchical-joint-half-v8"
AXES = ("s1", "o1", "k1", "s2", "o2", "k2")
M = 512
# (input features K, output features O, calls per Transformer block)
LAYERS = ((256, 64, 12), (256, 256, 1), (256, 1024, 1), (1024, 256, 1))
BASELINE = (32, 32, 256, 256, 32, 256)
O1 = O2 = 32
K_LEVELS = (256, 128, 64, 32, 16, 8, 4, 2, 1)


def ceildiv(numerator, denominator):
    """Divide nonnegative counts with an integer ceiling."""
    return (numerator + denominator - 1) // denominator


def line_round(byte_count):
    """Round a nominal byte window up to a 64-byte cache line."""
    return 64 * ceildiv(byte_count, 64)


def key(tiles):
    """Return tiles in the executable's six-parameter order."""
    return tuple(tiles[name] for name in AXES)


def profile(row):
    """Return the exact cache/cardinality identity used only for annotation."""
    return tuple(row[name] for name in ("l1_bytes", "l2_bytes", "entries", "learners"))


def capacity(text):
    """Parse integer bytes or binary KiB/MiB CLI capacities."""
    value = text.strip().lower()
    for suffix, factor in (("mib", 1048576), ("kib", 1024), ("b", 1)):
        if value.endswith(suffix):
            number = value[: -len(suffix)]
            if number.isascii() and number.isdecimal():
                return int(number) * factor
            raise argparse.ArgumentTypeError("Use integer bytes or integer KiB/MiB.")
    if value.isascii() and value.isdecimal():
        return int(value)
    raise argparse.ArgumentTypeError("Use integer bytes or integer KiB/MiB.")


def validate(l1, l2, entries, learners):
    """Check the historical workload's supported cache and cardinality domain."""
    if any(
        type(value) is not int or value <= 0 for value in (l1, l2, entries, learners)
    ):
        raise ValueError("All inputs must be positive integers.")
    if l1 > l2 or entries not in (4, 8, 16) or learners not in (1, 2, 4):
        raise ValueError("Require L1<=L2, CB4/8/16 and learners1/2/4.")
    if l1 % 128 or l2 % 1024:
        raise ValueError("Require complete 64-byte cache sets: L1 2-way, L2 16-way.")


def requirements(entries, learners, tiles):
    """Retained X/W/C bytes plus the frozen, nominal line-rounded H2 windows."""
    s1, o1, k1, s2, o2, k2 = key(tiles)
    bits = entries.bit_length() - 1
    overhead = {
        "shared_CI_window": line_round(ceildiv(bits * o2 * k2, 8)),
        "source_codebook": line_round(4 * learners * entries),
        "bias_window": line_round(4 * learners * o2),
    }
    h2 = sum(overhead.values())
    layers = []
    for features, outputs, count in LAYERS:
        padded_k = k2 * ceildiv(features, k2)
        arena = learners * (s2 * padded_k + o2 * k2 + 4 * s2 * o2)
        layers.append(
            {
                "K": features,
                "O": outputs,
                "multiplicity": count,
                "Kpad": padded_k,
                "arena_bytes": arena,
                "R2_bytes": arena + h2,
            }
        )
    return {
        "R1_bytes": learners * (s1 * k1 + o1 * k1 + 4 * s1 * o1),
        "R1_is_full_inner_all_learners": True,
        "R2_bytes": max(layer["R2_bytes"] for layer in layers),
        "H2_bytes": h2,
        "H2_components": overhead,
        "layers": layers,
    }


def capacity_attempt(l1, l2, entries, learners, depth):
    """Compute four-row feasibility and row bounds at one joint K depth."""
    minimum_tiles = dict(zip(AXES, (4, O1, depth, 4, O2, depth)))
    minimum = requirements(entries, learners, minimum_tiles)
    padded_k = max(depth * ceildiv(features, depth) for features, _, _ in LAYERS)
    inner_bound = (l1 - learners * O1 * depth) // (learners * (depth + 4 * O1))
    outer_bound = (l2 - learners * O2 * depth - minimum["H2_bytes"]) // (
        learners * (padded_k + 4 * O2)
    )
    failed = []
    if inner_bound < 4:
        failed.append("L1")
    if outer_bound < 4:
        failed.append("L2")
    return {
        "k1": depth,
        "k2": depth,
        "Kpad_max": padded_k,
        "H2_bytes": minimum["H2_bytes"],
        "L1_s1_bound": inner_bound,
        "L2_s2_bound": outer_bound,
        "feasible": not failed,
        "failure_constraints": failed,
        "minimum_4row_requirements": minimum,
    }


def largest_power_of_two(bound):
    """Return the largest power of two no greater than a positive bound."""
    return 1 << (bound.bit_length() - 1)


def select(l1, l2, entries, learners):
    """Apply v9's first-feasible K depth and balanced row policy, without evidence."""
    validate(l1, l2, entries, learners)
    result = {
        "version": VERSION,
        "formula_source": {
            "validation": "v9",
            "internal_version": VERSION,
            "sha256": FORMULA_SOURCE_SHA256,
        },
        "inputs": {
            "l1_bytes": l1,
            "l2_bytes": l2,
            "entries": entries,
            "learners": learners,
        },
        "policy": {
            "o1": O1,
            "o2": O2,
            "joint_k_levels": K_LEVELS,
            "fallback_trigger": "No four-row inner/outer pair fits both capacity targets",
        },
        "assumptions": {
            "writeback_clean": False,
            "sequence_rows": M,
            "layers_K_O_count": LAYERS,
            "packed_X": "All padded input features for the current outer row tile",
            "X_W_bytes_per_element": 1,
            "C_codebook_bias_bytes_per_element": 4,
            "SVE_bits": 128,
            "microkernel_rows": 4,
            "microkernel_outputs": 16,
            "line_bytes": 64,
            "L1_ways": 2,
            "L2_ways": 16,
            "eta1": 1,
            "eta2": 1,
        },
        "measurement_lookup_used_for_selection": False,
        "fitted_coefficients": 0,
        "absolute_runtime_prediction": False,
        "limitations": [
            "Workload-specific historical M512 policy; not arbitrary model dimensions.",
            "Capacity fit does not guarantee cache residency, numerical correctness or faster runtime.",
            "H2 is a nominal line-rounded index/codebook/bias allowance, not an address trace.",
            "Original X streaming and final Y writes are not retained-buffer targets.",
            "K below 16 is a mathematical proposal; backend support must be checked separately.",
            "Historical timing does not measure a newly compiled kernel or prove hardware generalization.",
        ],
        "attempts": [],
    }
    selected_attempt = None
    # Preserve the first feasible level; never replace it with a measured winner.
    for depth in K_LEVELS:
        attempt = capacity_attempt(l1, l2, entries, learners, depth)
        result["attempts"].append(attempt)
        if attempt["feasible"]:
            selected_attempt = attempt
            break

    if selected_attempt is None:
        result.update(
            status="NO_CAPACITY_FIT",
            tiles=None,
            fallback_used=True,
            reason="No joint K level from 256 through 1 fits four rows.",
            runtime_validation={"status": "NO_SELECTION", "records": []},
        )
        return result

    depth = selected_attempt["k1"]
    inner_bound = selected_attempt["L1_s1_bound"]
    outer_bound = selected_attempt["L2_s2_bound"]
    s1_from_l1 = largest_power_of_two(min(M, inner_bound))
    s1 = largest_power_of_two(min(M, inner_bound, outer_bound))
    # Balance row blocks without increasing the minimum number of W decode passes.
    maximum_s2 = s1 * (min(M, outer_bound) // s1)
    row_blocks = ceildiv(M, maximum_s2)
    s2 = s1 * ceildiv(M, row_blocks * s1)
    tiles = dict(zip(AXES, (s1, O1, depth, s2, O2, depth)))
    result.update(
        status="CAPACITY_PROPOSAL",
        tiles=tiles,
        output_order=AXES,
        fallback_used=depth != K_LEVELS[0],
        capacity_row_bounds={
            name: selected_attempt[name]
            for name in ("L1_s1_bound", "L2_s2_bound", "Kpad_max", "H2_bytes")
        },
        requirements=requirements(entries, learners, tiles),
        steps={
            "s1_from_L1_only": s1_from_l1,
            "s1_reduced_for_L2_feasibility": s1 != s1_from_l1,
            "s2_max_aligned": maximum_s2,
            "minimum_outer_row_blocks": row_blocks,
            "balanced_s2": s2,
            "actual_row_block_sizes": [min(s2, M - start) for start in range(0, M, s2)],
            "W_decode_passes_before": ceildiv(M, maximum_s2),
            "W_decode_passes_after": ceildiv(M, s2),
        },
        runtime_validation={"status": "UNMEASURED_SELECTION", "records": []},
    )
    return result


def annotate(result, rows):
    """Attach exact-coordinate historical evidence after selection, grouped by binary."""
    p = profile(result["inputs"])
    matches = [r for r in rows if profile(r) == p]
    if any(r.get("policy") is not False for r in matches):
        raise ValueError("False evidence only.")
    records = []
    identities = set()
    for r in matches:
        if not result["tiles"] or key(r["tiles"]) != key(result["tiles"]):
            continue
        binary = r.get("binary_sha256")
        if not isinstance(binary, str) or re.fullmatch(r"[0-9a-f]{64}", binary) is None:
            raise ValueError("Evidence requires a binary_sha256 identity.")
        identity = (binary, key(r["tiles"]))
        if identity in identities:
            raise ValueError("Duplicate exact coordinates for one binary.")
        identities.add(identity)
        bases = [
            b
            for b in matches
            if key(b["tiles"]) == BASELINE and b.get("binary_sha256") == binary
        ]
        if len(bases) > 1:
            raise ValueError("Duplicate same-binary baselines.")
        ticks = r["BLOCK_ticks"]
        if type(ticks) is not int or ticks <= 0:
            raise ValueError("BLOCK_ticks must be a positive integer.")
        base = bases[0] if bases else None
        if base and (type(base["BLOCK_ticks"]) is not int or base["BLOCK_ticks"] <= 0):
            raise ValueError("Baseline BLOCK_ticks must be a positive integer.")
        records.append(
            {
                "measurement_id": r["id"],
                "binary_sha256": binary,
                "BLOCK_ticks": ticks,
                "baseline_BLOCK_ticks": base["BLOCK_ticks"] if base else None,
                "baseline_runtime_reduction_pct": (
                    100 * (1 - ticks / base["BLOCK_ticks"]) if base else None
                ),
            }
        )
    result["runtime_validation"] = {
        "status": (
            "NO_SELECTION"
            if result["tiles"] is None
            else "HISTORICAL_EXACT" if records else "UNMEASURED_SELECTION"
        ),
        "records": records,
        "applies_to_new_binary": False,
        "note": "Only exact coordinates and same-binary baselines; historical timing does not validate a new binary.",
    }
    return result


def backend_status(result, capability=None):
    """Check a proposal against a supplied backend declaration, not timings."""
    if not result["tiles"]:
        return {"status": "NO_SELECTION"}
    if capability is None:
        return {
            "status": "UNKNOWN",
            "reason": "No backend capability supplied; output is a formula candidate.",
        }
    if capability.get("schema") != "i2ce.runtime-tiles.v1":
        raise ValueError("Unsupported backend capability schema.")
    t = result["tiles"]
    reasons = []
    for name, value in t.items():
        rule = capability["tiles"][name]
        outside_range = not rule.get("min", 1) <= value <= rule.get("max", value)
        outside_values = "values" in rule and value not in rule["values"]
        not_power_two = rule.get("power_of_two", False) and value & (value - 1) != 0
        if outside_range or outside_values or not_power_two:
            reasons.append(name)
    for inner, outer in (("s1", "s2"), ("o1", "o2"), ("k1", "k2")):
        if t[outer] % t[inner]:
            reasons.append(outer + " must be divisible by " + inner)
    if min(t["k1"], t["k2"]) < 128 and t["k1"] != t["k2"]:
        reasons.append("Compact K depths require k1=k2")
    inputs = result["inputs"]
    if (
        inputs["learners"] not in capability["learners"]
        or inputs["entries"] not in capability["codebook_entries"]
    ):
        reasons.append("learner/codebook combination")
    return {
        "status": "UNSUPPORTED" if reasons else "DECLARED_SUPPORTED",
        "backend": capability["backend"],
        "reasons": reasons,
        "numerically_verified_here": False,
    }


def main():
    """Print the proposal and distinguish capacity, backend and evidence status."""
    p = argparse.ArgumentParser(
        description=__doc__,
        epilog=(
            "Formula source: frozen v9 validation scheduler, internal version "
            + VERSION
            + ". Fixed M=512 and listed layers; H2 includes line-rounded shared indices, "
            "codebooks and bias. Capacity fit is not a cache-hit or speed guarantee. "
            "Example: python3 tools/tile_scheduler.py --l1 32KiB --l2 1MiB --cb 8 --learners 4"
        ),
    )
    p.add_argument("--l1", "--l1d-size", required=True, type=capacity)
    p.add_argument("--l2", "--l2-size", required=True, type=capacity)
    p.add_argument("--cb", "--codebook-size", required=True, type=int)
    p.add_argument("--learners", required=True, type=int)
    output = p.add_mutually_exclusive_group()
    output.add_argument(
        "--json",
        action="store_true",
        help="Show capacities, selection steps, assumptions and status.",
    )
    output.add_argument(
        "--emit-args",
        action="store_true",
        help="Emit named executable options instead of six values.",
    )
    p.add_argument(
        "--capabilities",
        help="Optional backend capability JSON; reject an unsupported selected tile.",
    )
    p.add_argument(
        "--measurements",
        help="Optional historical observation JSON; never used for selection.",
    )
    p.add_argument(
        "--verbose",
        action="store_true",
        help="Write backend and measurement status to stderr.",
    )
    a = p.parse_args()
    try:
        r = select(a.l1, a.l2, a.cb, a.learners)
        capability = None
        if a.capabilities:
            with open(a.capabilities, "rb") as f:
                raw = f.read()
            capability = json.loads(raw)
            r["capability_sha256"] = hashlib.sha256(raw).hexdigest()
        r["backend_support"] = backend_status(r, capability)
        if a.measurements:
            with open(a.measurements, "rb") as f:
                raw = f.read()
            rows = json.loads(raw)
            if isinstance(rows, dict):
                rows = rows["observations"]
            annotate(r, rows)
            r["evidence_sha256"] = hashlib.sha256(raw).hexdigest()
        invalid = r["tiles"] is None or r["backend_support"]["status"] == "UNSUPPORTED"
        if a.json:
            print(json.dumps(r, indent=2, allow_nan=False))
        elif not invalid:
            print(
                " ".join(
                    (
                        ("--%s %d" % (name, r["tiles"][name]))
                        if a.emit_args
                        else str(r["tiles"][name])
                    )
                    for name in AXES
                )
            )
            if a.verbose:
                print(
                    r["backend_support"]["status"]
                    + "; "
                    + r["runtime_validation"]["status"],
                    file=sys.stderr,
                )
        else:
            print(
                json.dumps(
                    {"status": r["status"], "backend_support": r["backend_support"]}
                ),
                file=sys.stderr,
            )
        return 2 if invalid else 0
    except (ValueError, OSError, KeyError, TypeError) as e:
        p.error(str(e))


if __name__ == "__main__":
    sys.exit(main())
