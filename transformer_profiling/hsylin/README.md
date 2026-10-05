# CB4 cache study results

The legacy E01–E11 dataset has been preserved verbatim in [the legacy archive](../archives/legacy-hsylin/). The archive manifest maps every file to its checksum and external raw-run package. No legacy raw simulation data was deleted.

New results use two separate namespaces with fixed action IDs:

- `cb4-baseline-optimization`: E01–E05, one performance run per action.
- `cb4-hierarchical-cache-tiling`: E06–E09, one performance run per action, after E05 passes the non-tiling gate.

Runner configuration ID 38 is shared by all actions; it is not a result ID. Use the study controller to collect into these subdirectories. Do not allocate new result IDs in this root.
