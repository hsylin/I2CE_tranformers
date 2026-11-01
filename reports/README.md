# Cache A experiment reports

Open [profiling.html](profiling.html) locally for Overview, Codebook Roofline
and Complete BLOCK Roofline. [roofline.html](roofline.html) contains the two
Roofline views. Both files are self-contained and work offline.

The published stages are E01–E10. E09 is Attention SDOT; E10 adds Attention
blocking. Names come from `transformer_profiling/sequence.json` in both
generators. All plotted values come from the validated saved measurements.

These plots describe the registered binaries from before the input and
Softmax repairs. The repaired checkpoints are not measured. In particular,
the E10 BLOCK value of 68.611685 ms remains attached to its original measured
source and binary, not to a newly reconstructed commit. The exported HTML
provenance retains those identities. Candidate source manifests separate
these historical references from new sources and pending validation.

To regenerate from the preserved measurement package, set `MEASURED_RESULTS`
to its root directory and run from the repository root:

```sh
python transformer_profiling/report.py \
  --input "$MEASURED_RESULTS/reports/profiling.tsv" \
  --roofline "$MEASURED_RESULTS/received-completion/roofline.json" \
  --output reports/profiling.html --single-run-study --compact-labels
python transformer_profiling/roofline.py \
  --input "$MEASURED_RESULTS/received-completion/roofline.json" \
  --output reports/roofline.html
```

The generators validate saved evidence; they do not run simulations.
