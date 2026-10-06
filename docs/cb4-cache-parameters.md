# CB4 cache parameters: E09 packing-v2 baseline

`submission` maintains E09's complete two-level staging design. Measured source
remains immutable at `exp/cb4-cache/e09-v2` (`fbc2d5fb6d94d67034c160108b408c5be85e05df`).
The current correctness fixes and parameter interface are **not a new performance result**.
[E01–E09 results](../transformer_profiling/hsylin/README.md) continue to identify the original measured commits.

## Scope and defaults

CB4, two learners, shared 2-bit indices, SVE128, single-output two-learner SDOT,
and one executing core. No MR2/NR4, automatic selector or multicore extension.

| Level | Sequence rows S | Output columns O | Reduction K | X/W row stride (bytes) |
|---|---:|---:|---:|---:|
| Inner | 16 | 32 | 128 | 144 |
| Outer | 128 | 128 | 512 | 528 |

`Full_NN/inc/gemm_cb4_cache_config.h` defines six `I2CE_CACHE_{S,O,K}{1,2}` macros.
They are independent of generated `TILE_L1_SIZE` / `TILE_L2_SIZE`, which must remain 1.
Strides remain K+16; changing padding is a separate experiment, not an exposed option.
External X is `int8 [S][K][2]`, Y is `int8 [S][O][2]`; scratch X/W/C use two learner planes.

```text
for s2, o2:
    initialize C2 from bias
    for k2:
        pack X into X2; decode shared indices into learner-specific W2
        for s1, o1:
            C2 -> C1
            for k1:
                X2 -> X1; W2 -> W1
                compute one (s,o), accumulating both learners into C1
            C1 -> C2
    low8(C2) -> Y
```

X1 and C1 are actively used. S1 still controls a real staging boundary. The later
direct-X2/direct-C2, W1 traversal, interleaved-C2 and vector-epilogue experiments
are not incorporated.

## Legal configurations

- All six dimensions are positive. S/O tiles are multiples of 4; K tiles are multiples of 16.
- Inner dimensions must not exceed their corresponding outer dimensions; exact divisibility is unnecessary.
- Outer S/O are at most 65535, outer K at most 65520. K1 bounds keep a signed-byte partial dot product within int32.
- Fixed K+16 strides cover every full 16-byte tail store/load. Inactive lanes are zeroed.
- Every arena field starts at a 64-byte boundary. Total allocation must not exceed INT32_MAX bytes, a conservative addressing limit, **not a cache-fit criterion**.
- Invalid geometry fails compilation. Explicit overrides also reject incompatible generated CB/learner/index/legacy-tile/SVE configurations.
- Runtime dimensions retain the existing pre-narrowing, bounds, alignment and alias guards. Allocation failure and unsupported runtime SVE widths can use the existing fallback; a timed cache experiment must prove it exercised the cache path.

With strides D1=K1+16, D2=K2+16, allocated bytes are:

```text
2(S1+O1)D1 + 8 S1 O1 + 2(S2+O2)D2 + 8 S2 O2
```

Defaults: X1 4608 B, C1 4096 B, W1 9216 B, X2 135168 B, C2 131072 B,
W2 135168 B; total **419328 B**. Field offsets are
0, 4608, 8704, 17920, 153088, 284160. Compile-time assertions preserve these defaults.
Actual working sets, reuse distance and interface traffic must be considered
separately. Changing a tile can also move later arena fields; record this rather
than attributing every timing change solely to capacity.

## Use the interface

For a manual build with the correct generated CB4/shared-I2 headers and data:

```bash
A64CXX=/absolute/path/to/aarch64-linux-gnu-g++ \
SIMD_FLAG=1 USE_CODEBOOK_GEMM_FLAG=1 FULL_INTERLEAVED_PIPELINE_FLAG=1 \
I2CE_CACHE_K1=64 I2CE_CACHE_O1=16 bash compile_transformer.sh
```

Unset all six environment variables for defaults. Build logs print explicit overrides.
The build still needs the model's other ordinary flags/assets and correct weight path.

For a **future** experiment row, the runner accepts `tile_s1`, `tile_o1`, `tile_k1`,
`tile_s2`, `tile_o2`, `tile_k2` in the overrides column. For example:

```text
tile_k1=64,tile_o1=16
```

The runner translates these into compile definitions and records them in
`build_config.tsv` (`overrides` and `compile_flags`). It clears ambient tile variables,
rejects explicit tile overrides for old sources that ignore them, and retains
source/binary identity. No new experiment rows or simulations were created here.

Existing `l1d=64KiB,l2=2MiB` overrides remain **hardware** parameters. They do not
change these software tiles. Change one category at a time initially. Verify the
actual run `config.ini`, machine/checkpoint compatibility, generated inputs,
compiler, ROI and dispatch before comparing. Associativity, arbitrary SVE widths
and multicore support are not added by this interface.

## Manual checks

No new CI service is required. With a matching AArch64 compiler and SVE QEMU:

```bash
export A64CXX=/absolute/path/to/aarch64-linux-gnu-g++
export I2CE_QEMU=/absolute/path/to/qemu-aarch64
bash tests/run_cb4_c32_wrap_test.sh
bash tests/run_cb4_cache_test.sh
bash tests/run_cb4_cache_parameters_test.sh
python3 -O tests/cb4_cache_parameters_config_test.py
python3 -O tests/cb4_publication_test.py
python3 -O transformer_profiling/hsylin/reproduction/verify_results.py
```

The parameter test checks default, smaller and non-dividing inner/outer tiles,
actual consumer dispatch, all 32 accumulator bits before low8 conversion,
poisoned tails, guarded bounds, alias/allocation fallback and invalid geometry.
An intentional child SIGSEGV verifies guard-page enforcement; the parent must report PASS.
These are correctness checks, not performance repetitions. Whole-pipeline checks
also need the preserved generated inputs; public external-asset limitations are
listed in the [reproduction notes](../transformer_profiling/hsylin/reproduction/README.md).

## Validation and next step

The manual integration record is [here](cb4-cache-integration-validation.md).
C32 now uses explicit modulo-2^32 addition. Flat scratch arrays preserve E09's
bytes, field offsets and staging, but changed compiler register allocation and
address calculations. Default parameterization adds no further cache-function
machine-code change in the checked build. This does **not** establish identical
whole-model runtime to measured E09.

Before the first new tile comparison, obtain one default-configuration performance
reference for the corrected source in a new namespace, then test a small justified
candidate set. Keep the one-run-per-configuration policy. Declare intended source,
tile or hardware differences explicitly; other provenance must agree. The archived
E01–E09 verifier is deliberately strict and is not an arbitrary-hardware comparator.
CI, Releases/DOI, automatic tuning and additional fixed-configuration optimizations
are deferred.
