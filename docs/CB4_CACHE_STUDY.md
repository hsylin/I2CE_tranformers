# Fixed single-output CB4 cache study

This source is E08: L1/L2 compressed-index blocking, derived from E06 and the
measured E05 non-tiling baseline. It targets two learners, CB4, shared 2-bit indices and
SVE128. Other configurations decline the new entry and retain the E05 caller's
fallback. The production caller already checks the two-learner shared-index
contract. This is not a general tiled GEMM interface.

External layout is X[S][K][2], I[O][ceil(K/16)] uint32, CB[4][2], bias[O][2],
Y[S][O][2]. Each index selects both learner weights; learner outputs stay
separate. The cache entry accepts size_t dimensions, validates before narrowing,
and rejects output overlap with inputs, indices, codebook or bias without writes.
The caller rejects dimensions outside its inherited uint16/uint32 offset domain
before constructing its descriptor. This does not repair other legacy APIs.

Fixed S1/O1/K1 = 16/32/128 and S2/O2/K2 = 128/128/512. Loop order is
s2, o2, k2, s1, o1, k1, s, o, vector-K.
One arena per GEMM holds X1[2][16][144] (4,608 B), C1[2][16][32] uint32
(4,096 B), I1[32][12] uint32 (1,536 B): 10,240 B actually allocated,
with a 64-byte aligned base and fields at 64-byte aligned offsets.
No multi-output register block: each (s,o) has two SDOT vector accumulators.
X1 is deinterleaved once per tile; I1 is copied once, but each sequence row
decodes it again. Zero activation padding makes the final SDOT vector safe.
Bias initializes C1 once, modulo-2^32 partial sums accumulate across all K1,
and only the final result is narrowed to its low byte. Signed partial sums
are bounded by 2,097,152; the two learners are never summed together.

The conservative L1 budget remains 15,808 B: arena 10,240 + bias 256 + setup
1,024 + alignment/separation reserve 192 + source streaming reserve 4,096.
Setup includes two 256-byte codebook scratch arrays. The 192-byte alignment
reserve is headroom, not an additional arena allocation. This is below the
24 KiB target budget in a 32 KiB, 2-way L1D, but it does not guarantee residency
or eliminate set conflicts. The outer arena adds X2[2][128][528] (135,168 B), C2[2][128][128] uint32
(131,072 B), I2[128][36] uint32 (18,432 B). Total allocation is 294,912 B.
C2 receives bias once before all K2 blocks. Each C1 is loaded from C2, updated
over K1 subpanels, and written back before the next K2 block. Y is written only
after the entire K dimension. X2 and I2 are packed once per outer window and
copied into the inner panels; compressed decoding still occurs per output.
The conservative L2 budget is 510,336 B (498.375 KiB), including the actual
inner/outer arena, 384 B alignment reserve, 131,072 B raw X window, 16,384 B
raw I window, 32,768 B final Y window, 1,024 B bias, 1,024 B setup, and 32,768 B
other/cache-line/streaming reserve. It is below 768 KiB of the 1 MiB L2.

Allocation failure returns before any output write; the caller then executes
E05. Test-only allocator injection follows the same status-check branch, and
test counters prove the cache path and single-output partials are exercised.
They are absent from production binaries. `tests/run_cb4_cache_test.sh` uses
QEMU for scalar-oracle tails, wrap/bias, guard pages, invalid dimensions,
alias rejection, allocation failure and SVE/compile-time dispatch checks.
The deliberate guard-page probe child must receive SIGSEGV; it is not a
kernel crash. Full-pipeline validation separately compares 84 tensors to E03.

There is no tuning interface or environment-selected tile size. Keep measured
source commits and result provenance immutable; report actual simulated ROI
and phase-level counters after the single formal run.

Design references are the approved plan's layered packing/cache model, adapted
to shared compressed indices and per-learner weights. Capacity is only one
constraint; associativity, TLB reach, packing traffic and instruction overhead
must be assessed from measured results. See Goto & van de Geijn (2008), Low
et al. (2016), Smith et al. (2014), and BLIS `docs/Multithreading.md`; this
experiment uses one simulated core and does not introduce GEMM threading.
