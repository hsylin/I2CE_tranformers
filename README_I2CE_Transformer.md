> E01-E13 implementation demo: see [the reconstruction workflow](transformer_profiling/experiment-runner.md).
> Existing archives and measurements below retain their historical meaning.

# I2CE Transformer README

This document explains the Transformer experiment implementation in this repository. It describes the required files, the role of each major component,
the integer and FP32 Transformer execution paths, the fully interleaved multi-learner pipeline, and the compile-time macros used for experiments.

The implementation described here is located in the TiC-SAT project root:

```bash
/home/thu/TiC-SAT
```

The final executable is generated as:

```bash
/home/thu/TiC-SAT/transformer.o
```

Note: the compile script in this repository is named:

```bash
compile_transformer.sh
```

with a lowercase `c`.

## Documentation Guide

Please read the semester project report first.  
The report explains the motivation, methodology, experimental setup, and main results of the project.

After reading the report, please use the following manuals depending on your purpose:

| Document | Purpose |
| --- | --- |
| `USER_MANUAL.md` | Explains how to use the code, including the end-to-end workflow, notebook generation, compilation, QEMU/gem5 execution, correctness checking, and experiment reproduction. |
| `DEVELOPER_MANUAL.md` | Provides a more detailed explanation of the implementation, including code structure, generated files, Transformer execution paths, function-call maps, GEMM/SVE kernels, and notes for future development. |

Recommended reading order:

```text
Semester project report
  -> README_I2CE_Transformer.md
  -> USER_MANUAL.md
  -> DEVELOPER_MANUAL.md
```
---

## Project Repositories

The final submission version is integrated into the TiC-SAT repository. As requested by the supervisors, the necessary run scripts, Jupyter notebooks, generated headers, and notebook-based generator outputs for evaluation are included in TiC-SAT. Please check this repository first when evaluating or reproducing the experiments.

Additional development files are available in the data-generator, gem5 and TiC-SAT repositories if needed.

All archived experiment executables are stored in: `/executable archive`.

All the experiments results are stored in: `/transformer_profiling`.

A video of inference runtime is stored as `inference_runtime.mov`.

The required Python/Conda environment is specified in `environment.yml` and `requirements.txt`.

For final submission and evaluation, the generated `/weights` directory and `Full_NN/gemm_definitions` directory are included in the repository. 
During normal development, these generated files are not intended to be tracked by Git. 
Future developers should regenerate them from the notebook when changing model dimensions, learner count, codebook size, or execution mode.

| Repository                                                   | Role                                                         |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| [TiC-SAT Transformer framework and project integration](https://github.com/Jerry0209/TiC-SAT) | Main repository for the final implementation, integration, compilation, and evaluation scripts. The final version is integrated in the `submission*` branch. |
| [Data generator implementation](https://github.com/Jerry0209/NN_layers-Jerry) | Contains the notebook and Python-side generator development used to create Transformer weights, codebooks, packed indices, and reference outputs. This repository was forked from a private repository, so access permission may be required. |
| [Modified gem5 full-system simulation environment](https://github.com/Jerry0209/gem5) | Contains the modified gem5 setup used for full-system simulation, SVE configuration, VirtIO 9P shared-folder support, checkpoints, and experiment launch scripts. |

---

## Background

This project builds on the TiC-SAT Transformer framework from the Embedded Systems Laboratory (ESL) at EPFL. TiC-SAT provides a C/C++ Transformer inference framework and gem5-based full-system simulation environment for evaluating Transformer acceleration.

This semester project extends the original framework toward SIMD acceleration of Transformer-based ensembles. The implementation adds support for
codebook-compressed Transformer GEMM layers, shared-index multi-learner execution, ARM SVE kernels, int8 and FP32 execution paths, and fully interleaved multi-learner Transformer inference.

The data-generation flow is inspired by the I2-CE methodology, where multiple learners share the same packed index stream while using learner-specific
codebooks. This structure reduces memory overhead and exposes data-level parallelism that can be exploited by SIMD execution.

---

## Acknowledgements

This work is based on previous research and code from the Embedded Systems Laboratory at EPFL.

The original TiC-SAT framework was developed by Alireza Amirshahi, Joshua Klein, Giovanni Ansaloni, and David Atienza.

The I2-CE methodology and codebase were developed by Stefano Albini and collaborators. The `NN_layers-Jerry` repository used in this project was forked from the private I2-CE/NN-layers development repository.

I would like to thank Stefano Albini, Dr. Giovanni Ansaloni, and Prof. David Atienza for their supervision, guidance, and support throughout this semester project.

