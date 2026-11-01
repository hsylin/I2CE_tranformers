# Reconstructed implementation milestones

This sequence starts at `048f573d1e1eb8c1d562aeedb8ab9989dc36e914`.
Two preceding fixes reject incomplete required input files and protect scalar
Softmax normalization when its shifted denominator would be zero.
The shared Roofline collector, report and stage tools precede E01.

| Milestone | Cumulative change |
|---|---|
| E01 | Scalar Codebook baseline and common compiler policy. |
| E02 | Codebook SDOT with packed A/W panels and safe tails. |
| E03 | Byte activations, shared indices and paired learner packing. |
| E04 | Inner blocking with fixed Codebook Kacc 256. |
| E05 | Outer blocking. |
| E06 | Packed A retention across all reduction panels. |
| E07 | Complete weight-panel decoding. |
| E08 | Learner-first traversal and bounded C32 tile conversion. |
| E09 | Attention SDOT. |
| E10 | Blocking of Attention QK and SV products. |

E08 combines traversal and output-storage changes. Conversion remains outside
the SDOT microkernel. Final-K fusion, Query16 streaming and vector Softmax are
absent. All translation units use `-O2`; only `Full_NN/src/gemm_SVE.c` adds
`-fno-schedule-insns`. A selected microkernel inspection does not establish
that the whole binary is free of spills.

Each E01–E10 tag contains its correctly named source manifest and preparation
tool. Use the tool from the selected tag or a later sequence checkout:

```sh
python tools/exp/prepare_stage.py --stage E09 --output /tmp/i2ce-e09
python tools/exp/prepare_stage.py --stage E10 --output /tmp/i2ce-e10
```

The output directory must be new. The default revision is the public stage
tag; `--revision` selects another revision with a matching stage manifest.
Source files, canonical workload inputs and build-tool hashes are checked.
Canonical generated headers and weights are copied only into the new output.

To build, set `A64CXX` to the registered AArch64 compiler, `LIBM5_A` to its
matching arm64 libm5 archive, and `M5_INC` to the matching gem5 include directory,
then add `--build`. The manifest records the required tool hashes and flags.
The command records the resulting binary hash; it does not execute the binary,
run component tests, or launch a simulator. A successful build alone is not a
correctness or performance observation. From E03, execute a prepared workload
from its source directory so `weights/compressed_indices` resolves correctly.

These repaired sources are not measured. Their binary hash is unknown until
built, and their production validation remains pending. Each manifest keeps
the previous measured source/binary under `reference_measurement`; previous
component proofs are nested under `reference_validation`. They do not certify
the repaired binary. Source hashes identify the new code independently.

[Saved reports](../reports/README.md) retain observations of the earlier
registered binaries. In particular, 68.611685 ms is the historical E10 BLOCK
observation, not a measurement of the repaired E10. No timing is inferred from
matching algorithm names. The L1/L2 references remain independent measured,
pattern-specific attainable rates; see [the method](roofline-method.md).
