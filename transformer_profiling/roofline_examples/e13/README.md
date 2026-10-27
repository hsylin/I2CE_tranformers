# Preserved E13 Roofline observation

This bundle contains excerpts of a completed, separately instrumented E13 run.
It is not a new measurement and is not the historical TSV experiment named E13.
The accepted E13 source and binary were unchanged. The cache policy is the
original L2 `mostly_excl`, with L1D `writeback_clean=false`.

- Source: `f601c78b482494ff33030d5a84a0cad966df30ae`.
- Binary SHA256: `5d52441dc570a1a1b2db66bdb80287290c3c2c5318fd82fb3bbff059337b9766`.
- gem5: Jerry0209/gem5 `ed752e78a2ddd5587c1954e2cda51f94cbbe554e`.
- `stats.txt`: selected native counter lines, unchanged, from all 23 dumps.
  `provenance.json` maps every retained line to the complete original file and
  records the complete original stats/config SHA256 values.
- `config.json`: hardware objects and CPU fields used by the reference model,
  with retained values unchanged. Software/image/input paths are omitted.
  This excerpt is an analysis input, **not a bootable gem5 configuration**.
- `regions.tsv`: original region index. Dump 22 (zero-based 21) is the cumulative
  whole-block ROI; dump 23 includes cleanup and is excluded.
- `receipt.json`: extracted successful completion, configuration-gate and guest
  binary checks. Its original evidence hashes are in `provenance.json`.
- `roofline.json`: hash-bound inputs, algorithm dimensions, ROI selection,
  limitations, and explicit unavailable entries for E01–E12.

The complete raw evidence remains preserved outside the source repository.
Hashes identify those originals; this compact bundle does not contain enough
information to independently repeat every original runtime identity check.
It does contain the native counters/configuration needed to recompute the plot.
There is one observation, no uncertainty estimate, and no observed production
tensor-value validation. Monitor and control ROI timing differed by +0.024765%.
Earlier custom/official calibration and the policy-C comparison are separate
studies; this bundle does not use their measured rates as theoretical ceilings.
