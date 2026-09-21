# User Manual

## Document Purpose and Reading Order

Please read the semester project report first before using this manual.  
The report explains the motivation, methodology, experimental setup, and main results of the project.

After reading the report, this **User Manual** explains how to use the code in practice.  
It focuses on the end-to-end workflow, including how to generate Transformer weights, compile the implementation, run the executable, verify correctness, and reproduce the gem5 profiling experiments.

For more detailed implementation explanations, internal function-call maps, and future development notes, please refer to `DEVELOPER_MANUAL.md`.

## Table of Contents

1. [End-to-End Usage Workflow](#1-end-to-end-usage-workflow)  
   1.1 [Generate Transformer Weights from the Notebook](#11-generate-transformer-weights-from-the-notebook)  
   1.2 [Configure the C/C++ Transformer Code](#12-configure-the-cc-transformer-code)  
   1.3 [Compile the C/C++ Implementation](#13-compile-the-cc-implementation)  
   1.4 [Run the Executable](#14-run-the-executable)  
   1.5 [Verify C/C++ Outputs Against Python Reference Outputs](#15-verify-cc-outputs-against-python-reference-outputs)  
   1.6 [Important Consistency Checklist](#16-important-consistency-checklist)
2. [How to Use the Compile Script](#2-how-to-use-compile-script)  
   2.1 [Experiment Flags and Macros](#21-experiment-flags-and-macros)  
   2.2 [Recommended Experiment Workflow](#22-recommended-experiment-workflow)  
   2.3 [Compile Command Examples](#23-compile-command-examples)
3. [How to Run `transformer.o` in QEMU](#3-how-to-run-transformero-in-qemu)
4. [How to Run `transformer.o` in gem5](#4-how-to-run-transformero-in-gem5)



## 1. End-to-End Usage Workflow

This section gives a complete overview of how to use this Transformer implementation, starting from weight generation in the Python notebook and ending with C/C++ execution and output verification.

The general workflow is:

```text
Transformer_generator.ipynb
    -> generate dense weights, codebook weights, input tensors, and Python reference outputs
    -> compile the C/C++ Transformer implementation
    -> run transformer.o in QEMU
    -> generate C/C++ layer-wise outputs
    -> compare C/C++ outputs with Python reference outputs
    -> run transformer.o in gem5 to get profiling statistics
```

---

### 1.1 Generate Transformer weights from the notebook

Open and run:

```text
Full_NN/generators/Transformer_generator.ipynb
```

The notebook is responsible for generating:

- notebook-generated dense `.bin` weights,
- codebook/index `.h` files,
- the generated codebook registry,
- the input tensor,
- Python reference layer-wise outputs.

---

#### Step 1: Set the Transformer model size

At the beginning of the notebook, select the Transformer dimensions.  
For example, the BERT-mini configuration used in this project is:

```python
# BERT-mini
TRANSFORMER_D_Q = 64
TRANSFORMER_SEQ_LEN = 512
TRANSFORMER_D_MODEL = 256
TRANSFORMER_NUM_HEADS = 4
TRANSFORMER_D_FF = 1024
```

These values must later match the definitions in `transformer.h`.

---

#### Step 2: Set the TiC-SAT export path

Set the export path to the root directory of the TiC-SAT repository:

```python
TICSAT_REPO_ROOT = "/home/thu/TiC-SAT"
```

The notebook writes generated files directly into this repository.

---

#### Step 3: Set the TiC-SAT systolic-array size

Set the TiC-SAT systolic-array size used by the generated configuration:

```python
TICSAT_SA_SIZE = 4
```

This value should match the `SA_SIZE` used when compiling TiC-SAT.

---

#### Step 4: Set the I2-CE/codebook hyperparameters

Set the codebook and ensemble parameters:

```python
N_LEARNERS = 4

CODEBOOK_SIZE = 8
# CODEBOOK_SIZE = 4, 8, and 16 have been tested for the single-learner case.
# The generated codebook results match the default dense reference results.

USE_CODEBOOKS = True
# Always set to True in this project.

SVE_LANES = 4
# Number of 32-bit lanes in the assumed SVE vector register. Change it according to the SVE vector register size.

SAME_SEQ = True
# True  -> shared-index / same-sequence implementation. 
# False -> independent per-learner index sequence implementation.
# Always set to True in this project.

USE_FP32_TRANSFORMER = False
# False -> generate and use the int8 Transformer path.
# True  -> additionally generate data for the FP32 Transformer path.
```

For the main shared-index I2-CE implementation, use:

```python
SAME_SEQ = True
USE_CODEBOOKS = True
```

---

#### Step 5: Run the entire notebook

After setting the model size, export path, and hyperparameters, run the whole
notebook.

The generated dense `.bin` weights and inputs should appear under:

```text
weights/
├── generated_from_notebook/
│   ├── learner0/
│   ├── learner1/
│   ├── learner2/
│   └── learner3/
└── H-1_L-1.bin
```

Here, `H-1_L-1.bin` is the generated input tensor.

The generated codebook `.h` files should appear under:

```text
Full_NN/
└── gemm_definitions/
    ├── codebooks_def.h
    ├── gemm_data.h
    ├── gemm_header_0.h
    ├── gemm_header_1.h
    ├── ...
    ├── gemm_header_38.h
    ├── generated_codebook_registry.h
    └── input_matrix.h
```

The Python reference layer-wise outputs should appear under:

```text
weights/
└── multiple_learner_outputs/
    └── python/
```

These Python outputs are later used to check the correctness of the C/C++
execution.

---

### 1.2 Configure the C/C++ Transformer code

After generating the files from the notebook, update the C/C++ side so that it
uses the same dimensions and paths.

---

#### Step 1: Update `transformer.h`

Open:

```text
transformer.h
```

Set the Transformer dimensions to the same values used in the notebook.

For BERT-mini:

```c
// BERT-mini
#define D_Q 64
#define D_SEQ 512
#define D_MODEL 256
#define NUM_HEAD 4
#define D_FF 1024
```

These definitions must match:

```python
TRANSFORMER_D_Q
TRANSFORMER_SEQ_LEN
TRANSFORMER_D_MODEL
TRANSFORMER_NUM_HEADS
TRANSFORMER_D_FF
```

in `Transformer_generator.ipynb`.

---

#### Step 2: Update the weight path in `transformer.cpp`

Open:

```text
transformer.cpp
```

Make sure the weight directory points to your TiC-SAT repository path.

For example:

```cpp
// Prefer the host-side project path and fall back to the 9p mount in gem5.
std::string dir_name = "/home/thu/TiC-SAT/weights";
if (!std::filesystem::exists(dir_name)) {
    dir_name = "/mnt/weights";
}
std::string notebook_weights_dir = dir_name + "/generated_from_notebook";
```

If your TiC-SAT repository is located somewhere else, replace:

```text
/home/thu/TiC-SAT
```

with your own repository path.

The `/mnt/weights` fallback is useful when running inside gem5 with a mounted
weights directory.

---

### 1.3 Compile the C/C++ implementation

After the notebook-generated files and C/C++ dimensions are consistent, compile
the Transformer executable from the TiC-SAT root directory:

```bash
cd /home/thu/TiC-SAT
source ./compile_transformer.sh
```

For actual experiments, use one of the compile commands listed in the compile
command section below, such as:

- dense no-SIMD baseline,
- int8 codebook GEMM with SVE,
- FP32 codebook Transformer with SVE.

After successful compilation, the executable is generated at:

```text
transformer.o
```

in the root directory of TiC-SAT.

---

### 1.4 Run the executable

Run the generated executable in QEMU or gem5 according to the normal TiC-SAT
workflow.

The executable to run is:

```text
transformer.o
```

During execution, the C/C++ implementation can dump layer-wise outputs for each
learner.

The C/C++ layer-wise outputs should appear under:

```text
weights/
└── multiple_learner_outputs/
    └── c/
```

After both Python and C/C++ outputs are generated, the output directory should
look like:

```text
weights/
└── multiple_learner_outputs/
    ├── c/
    └── python/
```

---

### 1.5 Verify C/C++ outputs against Python reference outputs

After running the C/C++ executable, go back to:

```text
Full_NN/generators/Transformer_generator.ipynb
```

Then either:

1. rerun the whole notebook, or
2. rerun only the final comparison/checking cell.

The comparison cell checks the C/C++ layer-wise outputs in:

```text
weights/multiple_learner_outputs/c/
```

against the Python reference outputs in:

```text
weights/multiple_learner_outputs/python/
```

If the implementation is correct, the C/C++ outputs should match the Python
reference outputs within the expected tolerance.

Below is an expected output of notebook: 

* int8


```text
input shape: (512, 256)

=============== LEARNER 0 ===============

Exported Python reference outputs to: /home/thu/TiC-SAT/weights/multiple_learner_outputs/python/learner0

multihead_out shape: (512, 256)

condense_out shape: (512, 256)

after_attn_addnorm shape: (512, 256)

ff0_out shape: (512, 1024)

ff1_out shape: (512, 256)

final_out shape: (512, 256)

===== C vs Python comparison | learner 0 =====
q_h0: max_abs_diff=0, mismatches=0/32768
k_h0: max_abs_diff=0, mismatches=0/32768
v_h0: max_abs_diff=0, mismatches=0/32768
head_out_h0: max_abs_diff=0, mismatches=0/32768
q_h1: max_abs_diff=0, mismatches=0/32768
k_h1: max_abs_diff=0, mismatches=0/32768
v_h1: max_abs_diff=0, mismatches=0/32768
head_out_h1: max_abs_diff=0, mismatches=0/32768
q_h2: max_abs_diff=0, mismatches=0/32768
k_h2: max_abs_diff=0, mismatches=0/32768
v_h2: max_abs_diff=0, mismatches=0/32768
head_out_h2: max_abs_diff=0, mismatches=0/32768
q_h3: max_abs_diff=0, mismatches=0/32768
k_h3: max_abs_diff=0, mismatches=0/32768
v_h3: max_abs_diff=0, mismatches=0/32768
head_out_h3: max_abs_diff=0, mismatches=0/32768
multihead_out: max_abs_diff=0, mismatches=0/131072
condense_out: max_abs_diff=0, mismatches=0/131072
after_attn_addnorm: max_abs_diff=0, mismatches=0/131072
ff0_out: max_abs_diff=0, mismatches=0/524288
ff1_out: max_abs_diff=0, mismatches=0/131072
final_out: max_abs_diff=0, mismatches=0/131072

```
* fp32

```text
softmax_qk_h1: max_abs_diff=6.98492e-09, mismatches=0/262144 (atol=0.001, rtol=0.0001)
softmax_v_pre_post_h1: max_abs_diff=1.86265e-07, mismatches=0/32768 (atol=0.001, rtol=0.0001)
head_out_h1: max_abs_diff=1.86265e-07, mismatches=0/32768 (atol=0.001, rtol=0.0001)
head_out_post_h1: max_abs_diff=1.86265e-07, mismatches=0/32768 (atol=0.001, rtol=0.0001)
q_h2: max_abs_diff=1.43051e-06, mismatches=0/32768 (atol=0.001, rtol=0.0001)
k_h2: max_abs_diff=1.54972e-06, mismatches=0/32768 (atol=0.001, rtol=0.0001)
v_h2: max_abs_diff=1.07288e-06, mismatches=0/32768 (atol=0.001, rtol=0.0001)
softmax_qk_h2: max_abs_diff=4.19095e-09, mismatches=0/262144 (atol=0.001, rtol=0.0001)
softmax_v_pre_post_h2: max_abs_diff=1.15484e-07, mismatches=0/32768 (atol=0.001, rtol=0.0001)
head_out_h2: max_abs_diff=1.15484e-07, mismatches=0/32768 (atol=0.001, rtol=0.0001)
head_out_post_h2: max_abs_diff=1.15484e-07, mismatches=0/32768 (atol=0.001, rtol=0.0001)
q_h3: max_abs_diff=1.19209e-06, mismatches=0/32768 (atol=0.001, rtol=0.0001)
k_h3: max_abs_diff=1.43051e-06, mismatches=0/32768 (atol=0.001, rtol=0.0001)
v_h3: max_abs_diff=1.54972e-06, mismatches=0/32768 (atol=0.001, rtol=0.0001)
softmax_qk_h3: max_abs_diff=4.65661e-09, mismatches=0/262144 (atol=0.001, rtol=0.0001)
softmax_v_pre_post_h3: max_abs_diff=1.04308e-07, mismatches=0/32768 (atol=0.001, rtol=0.0001)
head_out_h3: max_abs_diff=1.04308e-07, mismatches=0/32768 (atol=0.001, rtol=0.0001)
head_out_post_h3: max_abs_diff=1.04308e-07, mismatches=0/32768 (atol=0.001, rtol=0.0001)
multihead_out: max_abs_diff=1.86265e-07, mismatches=0/131072 (atol=0.001, rtol=0.0001)
condense_out: max_abs_diff=1.19209e-07, mismatches=0/131072 (atol=0.001, rtol=0.0001)
after_attn_addnorm: max_abs_diff=9.53674e-07, mismatches=0/131072 (atol=0.001, rtol=0.0001)
ff0_out: max_abs_diff=2.20537e-06, mismatches=0/524288 (atol=0.001, rtol=0.0001)
ff1_out: max_abs_diff=1.2368e-05, mismatches=0/131072 (atol=0.001, rtol=0.0001)
final_out: max_abs_diff=3.51667e-06, mismatches=0/131072 (atol=0.001, rtol=0.0001)
```



Instead of using python notebook to compare the results, C code Transformer also has a reference mode (enabled by the macros `ENABLE_CODEBOOK_REFERENCE_FLAG` and `ENABLE_DEBUG_PRINT_FLAG`).  If `transformer.o` with above MACRO enabled runs in the QEMU, it's expected to see the outputs below. 

<p align="center">
  <img src="./README_figures/c_transformer_ouput.png" width="800">
</p>



---

### 1.6 Important consistency checklist

Before compiling and running, make sure the following settings are consistent:

| Item                   | Location                           | Must match                                 |
| ---------------------- | ---------------------------------- | ------------------------------------------ |
| Transformer dimensions | `Transformer_generator.ipynb`      | `transformer.h`                            |
| Number of learners     | `Transformer_generator.ipynb`      | `Full_NN/gemm_definitions/codebooks_def.h` |
| Codebook size          | `Transformer_generator.ipynb`      | generated `codebooks_def.h` and registry   |
| Same-sequence mode     | `SAME_SEQ` in notebook             | selected interleaved GEMM path             |
| FP32/int8 mode         | `USE_FP32_TRANSFORMER` in notebook | `USE_FP32_TRANSFORMER_FLAG` during compile |
| Repository path        | `TICSAT_REPO_ROOT` in notebook     | weight path in `transformer.cpp`           |
| TiC-SAT SA size        | `TICSAT_SA_SIZE` in notebook       | `SA_SIZE` during compile                   |

If any of these settings are changed, regenerate the notebook outputs before
compiling and running the C/C++ implementation again.





## 2. How to use compile script

Compile from the project root:

```bash
cd /home/thu/TiC-SAT
source ./compile_transformer.sh
```

The script finds an AArch64 compiler in this order:

1. USER CUSTOM compiler path
2. `aarch64-linux-gnu-g++`
3. `aarch64-conda-linux-gnu-g++`

The core compile command includes:

```text
transformer.cpp
transformer_layers/*.cc
Full_NN/src/gemm_exec.c
Full_NN/src/gemm_SVE.c          only when SIMD_FLAG=1
accelerator/smm_gem.cpp
accelerator/systolic_m2m.cc
```

Include paths:

```text
-I.
-IFull_NN/inc
-IFull_NN/gemm_definitions
```

Always-defined compile macros (Legacy from TiC-SAT, do not need to change them, to see what are they you could also refer to the legacy README.md from TiC-SAT):

| Macro | Meaning |
| --- | --- |
| `SA_SIZE=4` | Systolic-array size used by accelerator helpers. (This should match the TICSAT_SA_SIZE = 4 inside notebook)|
| `DEVELOP` | Enables development build behavior in the accelerator code. This flag should remain enabled for this project. |
| `CORE_NUM=1` | Number of cores requested by this build configuration. |

The output binary is:

```text
transformer.o
```

### 2.1 Experiment Flags and Macros

`compile_transformer.sh` accepts environment variables ending in `_FLAG`. When a
flag is `1`, the script adds the corresponding `-D...` macro.

| Environment flag | Default | Macro | Meaning |
| --- | ---: | --- | --- |
| `RELOAD_WEIGHT_FLAG` | `1` | `RELOAD_WEIGHT` | Enables runtime loading/generation of weights instead of relying only on static/default state. If this is disabled, the weights will be generated during the runtime, otherwise the program will load the weights generated before |
| `USE_NOTEBOOK_GENERATED_WEIGHTS_FLAG` | `1` | `USE_NOTEBOOK_GENERATED_WEIGHTS` | Uses `.bin` weights and input generated by `Transformer_generator.ipynb`. Effective only when `RELOAD_WEIGHT` is enabled. |
| `USE_CODEBOOK_GEMM_FLAG` | `0` | `USE_CODEBOOK_GEMM` | Uses generated `CodebookDense` layers (codebook GEMM) instead of the default dense layer path. With this flag enabled, the program execute codebooked GEMM instead of default dense GEMM. Effective only when `RELOAD_WEIGHT` is enabled. |
| `ENABLE_CODEBOOK_REFERENCE_FLAG` | `0` | `ENABLE_CODEBOOK_REFERENCE` | Builds a dense reference path beside codebook GEMM and compares outputs. Useful for debugging correctness, not profiling. This enables the computation of the dense Transformer and print the comparison results. |
| `ENABLE_DEBUG_PRINT_FLAG` | `0` | `ENABLE_DEBUG_PRINT` | Enables debug prints and previews. Automatically disabled by profiling modes. |
| `PROFILE_GEMM_ONLY_FLAG` | `0` | `PROFILE_GEMM_ONLY` | Disables reference/debug overhead so profiling focuses on codebook GEMM. |
| `GEM5_PROFILE_REGIONS_FLAG` | `0` | `GEM5_PROFILE_REGIONS` | Emits named gem5 stats checkpoints for high-level Transformer regions. If this is enabled, the profiling of the program will be divided into 6 parts: `after_mha`, `after_projection`, `after_attn_addnorm`, `after_ff1`, `after_ff2`, `final_total`, so we can see the statistics of different layers. |
| `FULL_INTERLEAVED_PIPELINE_FLAG` | `0` | `FULL_INTERLEAVED_PIPELINE` | Keeps all learner activations interleaved across the whole grouped Transformer block (Our project is focused on fully interleaved path, so it's always enabled). |
| `USE_FP32_TRANSFORMER_FLAG` | `0` | `USE_FP32_TRANSFORMER` | Runs the FP32 Transformer path instead of the integer path. Requires codebook GEMM. |
| `DENSE_NO_SIMD_BASELINE_FLAG` | `0` | `DENSE_NO_SIMD_BASELINE` | Runs the default dense Transformer as a no-SIMD sequential learner baseline. This flag cannot be combined with SIMD, codebook GEMM, or the fully interleaved pipeline. It is mainly used to build fair multi-learner dense baselines. Instead of loading one learner's weights, computing it, and then repeating the same process for the next learner, this mode first loads the weights for all learners and then executes the learners sequentially. This avoids mixing weight-loading overhead into the profiled computation region, making the multi-learner dense baseline more reasonable for comparison. |
| `SIMD_FLAG` | `0` | `SIMD` | Enables ARM SVE compilation. When this flag is enabled, the compile script adds `-march=armv8-a+sve` and compiles `Full_NN/src/gemm_SVE.c`. Together with `USE_CODEBOOK_GEMM_FLAG=1` and `FULL_INTERLEAVED_PIPELINE_FLAG=1`, this enables the shared-index codebook SIMD implementation. |



Important sanity checks:

| Rule | Why |
| --- | --- |
| `ENABLE_CODEBOOK_REFERENCE` requires `USE_CODEBOOK_GEMM`. | Reference comparison only makes sense against codebook execution. |
| `DENSE_NO_SIMD_BASELINE` requires `SIMD_FLAG=0`. | This baseline is intentionally scalar/sequential. |
| `DENSE_NO_SIMD_BASELINE` requires `USE_CODEBOOK_GEMM_FLAG=0`. | It benchmarks dense execution, not codebook execution. |
| `DENSE_NO_SIMD_BASELINE` cannot combine with `FULL_INTERLEAVED_PIPELINE`. | Full interleaving is a grouped codebook pipeline. |
| `FULL_INTERLEAVED_PIPELINE` requires `USE_CODEBOOK_GEMM`. | Full interleaving relies on codebook interleaved kernels. |
| Integer `FULL_INTERLEAVED_PIPELINE` requires `SIMD_FLAG=1`. | The int8 full pipeline is intended for SVE interleaved execution. |
| `USE_FP32_TRANSFORMER` requires `USE_CODEBOOK_GEMM`. | The FP32 path uses the generated codebook registry. |
| `USE_FP32_TRANSFORMER` cannot combine with `ENABLE_CODEBOOK_REFERENCE`. | The dense reference checker is for the integer codebook path. |

### 2.2 Recommended experiment workflow

For each experiment, the recommended workflow uses two compilation stages: one for correctness checking and one for clean profiling.

The experiments commands can also be found in `simulation_takeaway.txt`.

#### 2.2.1. Correctness-checking stage

First, compile the program in debug/correctness-checking mode:

```bash
RELOAD_WEIGHT_FLAG=1 \
USE_NOTEBOOK_GENERATED_WEIGHTS_FLAG=1 \
ENABLE_CODEBOOK_REFERENCE_FLAG=1 \
ENABLE_DEBUG_PRINT_FLAG=1 \
source ./compile_transformer.sh
```

In this stage, the program loads the weights generated by `Transformer_generator.ipynb` and enables debug output and reference checking.
This is useful when running in QEMU, because the C/C++ outputs can be compared against the Python reference outputs or the dense Transformer reference path.

This stage is mainly used to confirm that the selected configuration is functionally correct before running expensive gem5 profiling.

#### 2.2.2. Profiling stage

After correctness has been verified, recompile the program in profiling mode:

```bash
PROFILE_GEMM_ONLY_FLAG=1 \
GEM5_PROFILE_REGIONS_FLAG=1 \
ENABLE_CODEBOOK_REFERENCE_FLAG=0 \
ENABLE_DEBUG_PRINT_FLAG=0 \
source ./compile_transformer.sh
```

In this stage, reference comparison and debug printing are disabled to avoid unnecessary overhead. `PROFILE_GEMM_ONLY_FLAG=1` keeps the measured region clean, while `GEM5_PROFILE_REGIONS_FLAG=1` enables gem5 region checkpoints for layer-level performance analysis.

Above commands are only used to show the workflow, not representing the full commands.



### 2.3 Compile Command Examples

Here are the commands used for experiments, it's highly recommended to use the commands below.

#### 2.3.1 DENSE  baseline (default dense Transformer, originally implemented by TiC-SAT)

This mode uses the old/default dense Transformer path with notebook-generated `.bin` weights. SVE and codebook GEMM are disabled. Dense mode is inherited from TiC-SAT, so it's not necessary to verify the correctness of itself. Therefore, dense mode will not generate any layer-wise outputs and is not used to compare with python outputs. 

**Debug mode**

```bash
DENSE_NO_SIMD_BASELINE_FLAG=0 \
USE_FP32_TRANSFORMER_FLAG=0 \
FULL_INTERLEAVED_PIPELINE_FLAG=0 \
SIMD_FLAG=0 \
RELOAD_WEIGHT_FLAG=1 \
USE_NOTEBOOK_GENERATED_WEIGHTS_FLAG=1 \
USE_CODEBOOK_GEMM_FLAG=0 \
ENABLE_CODEBOOK_REFERENCE_FLAG=0 \
ENABLE_DEBUG_PRINT_FLAG=1 \
PROFILE_GEMM_ONLY_FLAG=0 \
GEM5_PROFILE_REGIONS_FLAG=0 \
source ./compile_transformer.sh
```

**Profiling mode**

```bash
DENSE_NO_SIMD_BASELINE_FLAG=0 \
USE_FP32_TRANSFORMER_FLAG=0 \
FULL_INTERLEAVED_PIPELINE_FLAG=0 \
SIMD_FLAG=0 \
RELOAD_WEIGHT_FLAG=1 \
USE_NOTEBOOK_GENERATED_WEIGHTS_FLAG=1 \
USE_CODEBOOK_GEMM_FLAG=0 \
ENABLE_CODEBOOK_REFERENCE_FLAG=0 \
ENABLE_DEBUG_PRINT_FLAG=0 \
PROFILE_GEMM_ONLY_FLAG=1 \
GEM5_PROFILE_REGIONS_FLAG=1 \
source ./compile_transformer.sh
```



#### 2.3.2 DENSE no-SIMD baseline (E01–E06)

This mode runs the default dense Transformer implementation without SVE.
It is used as the sequential dense baseline. 

Compared to the `DENSE` mode, this loads all learners' dense weights first and then executes the learners sequentially, which avoids including repeated weight-loading overhead in the profiled computation region.

**Debug mode**

```bash
DENSE_NO_SIMD_BASELINE_FLAG=1 \
USE_FP32_TRANSFORMER_FLAG=0 \
FULL_INTERLEAVED_PIPELINE_FLAG=0 \
SIMD_FLAG=0 \
RELOAD_WEIGHT_FLAG=1 \
USE_NOTEBOOK_GENERATED_WEIGHTS_FLAG=1 \
USE_CODEBOOK_GEMM_FLAG=0 \
ENABLE_CODEBOOK_REFERENCE_FLAG=0 \
ENABLE_DEBUG_PRINT_FLAG=1 \
PROFILE_GEMM_ONLY_FLAG=0 \
GEM5_PROFILE_REGIONS_FLAG=0 \
source ./compile_transformer.sh
```

**Profiling mode**

```bash
DENSE_NO_SIMD_BASELINE_FLAG=1 \
USE_FP32_TRANSFORMER_FLAG=0 \
FULL_INTERLEAVED_PIPELINE_FLAG=0 \
SIMD_FLAG=0 \
RELOAD_WEIGHT_FLAG=1 \
USE_NOTEBOOK_GENERATED_WEIGHTS_FLAG=1 \
USE_CODEBOOK_GEMM_FLAG=0 \
ENABLE_CODEBOOK_REFERENCE_FLAG=0 \
ENABLE_DEBUG_PRINT_FLAG=0 \
PROFILE_GEMM_ONLY_FLAG=1 \
GEM5_PROFILE_REGIONS_FLAG=1 \
source ./compile_transformer.sh
```

---

#### 2.3.3 Int8 codebook GEMM with SVE (E07 - E36)

This mode uses generated codebook GEMM layers and SVE kernels. 

This is the goal of the current project: codebook GEMM Transformer with SIMD. The conducted experiments mainly built on this mode.

For `N_LEARNERS=1`, `FULL_INTERLEAVED_PIPELINE_FLAG=1` has no practical effect.
For `N_LEARNERS=2` or `N_LEARNERS=4`, it enables the fully interleaved
multi-learner pipeline. In the experiments, `FULL_INTERLEAVED_PIPELINE_FLAG=1` is used as it's the optimized version.

**Int8 codebook debug / correctness-check mode**

```bash
DENSE_NO_SIMD_BASELINE_FLAG=0 \
USE_FP32_TRANSFORMER_FLAG=0 \
FULL_INTERLEAVED_PIPELINE_FLAG=1 \
SIMD_FLAG=1 \
RELOAD_WEIGHT_FLAG=1 \
USE_NOTEBOOK_GENERATED_WEIGHTS_FLAG=1 \
USE_CODEBOOK_GEMM_FLAG=1 \
ENABLE_CODEBOOK_REFERENCE_FLAG=1 \
ENABLE_DEBUG_PRINT_FLAG=1 \
PROFILE_GEMM_ONLY_FLAG=0 \
GEM5_PROFILE_REGIONS_FLAG=0 \
source ./compile_transformer.sh
```

**Int8 codebook profiling mode**

```bash
DENSE_NO_SIMD_BASELINE_FLAG=0 \
USE_FP32_TRANSFORMER_FLAG=0 \
FULL_INTERLEAVED_PIPELINE_FLAG=1 \
SIMD_FLAG=1 \
RELOAD_WEIGHT_FLAG=1 \
USE_NOTEBOOK_GENERATED_WEIGHTS_FLAG=1 \
USE_CODEBOOK_GEMM_FLAG=1 \
ENABLE_CODEBOOK_REFERENCE_FLAG=0 \
ENABLE_DEBUG_PRINT_FLAG=0 \
PROFILE_GEMM_ONLY_FLAG=1 \
GEM5_PROFILE_REGIONS_FLAG=1 \
source ./compile_transformer.sh
```

---

#### 2.3.4 FP32 codebook Transformer with SVE

This mode runs the FP32 Transformer path. It still uses the generated codebook registry, so `USE_CODEBOOK_GEMM_FLAG=1` is required.

The floating-point implementation was mainly included for functional completeness and future experimentation. It was used only to
verify that the codebook execution flow produces numerically correct results in floating-point format. No gem5 profiling or performance comparison was conducted for this implementation.

After verification, the FP32 implementation was confirmed to match the notebook reference outputs.



*Note: the FP32 path is provided mainly for functional completeness and future work.* 
*The main profiling results reported in the semester project report are based on the int8 codebook GEMM with SVE path, not the FP32 path.*

**FP32 debug mode**

```bash
DENSE_NO_SIMD_BASELINE_FLAG=0 \
USE_FP32_TRANSFORMER_FLAG=1 \
FULL_INTERLEAVED_PIPELINE_FLAG=1 \
SIMD_FLAG=1 \
RELOAD_WEIGHT_FLAG=1 \
USE_NOTEBOOK_GENERATED_WEIGHTS_FLAG=1 \
USE_CODEBOOK_GEMM_FLAG=1 \
ENABLE_CODEBOOK_REFERENCE_FLAG=0 \
ENABLE_DEBUG_PRINT_FLAG=1 \
PROFILE_GEMM_ONLY_FLAG=0 \
GEM5_PROFILE_REGIONS_FLAG=0 \
source ./compile_transformer.sh
```

**FP32 profiling mode**

```bash
DENSE_NO_SIMD_BASELINE_FLAG=0 \
USE_FP32_TRANSFORMER_FLAG=1 \
FULL_INTERLEAVED_PIPELINE_FLAG=1 \
SIMD_FLAG=1 \
RELOAD_WEIGHT_FLAG=1 \
USE_NOTEBOOK_GENERATED_WEIGHTS_FLAG=1 \
USE_CODEBOOK_GEMM_FLAG=1 \
ENABLE_CODEBOOK_REFERENCE_FLAG=0 \
ENABLE_DEBUG_PRINT_FLAG=0 \
PROFILE_GEMM_ONLY_FLAG=1 \
GEM5_PROFILE_REGIONS_FLAG=1 \
source ./compile_transformer.sh
```

---

#### Important notes

`ENABLE_CODEBOOK_REFERENCE_FLAG=1` is useful for correctness checking, but it should be disabled during profiling because it adds dense reference-comparison overhead.

`GEM5_PROFILE_REGIONS_FLAG=1` should usually be used together with `PROFILE_GEMM_ONLY_FLAG=1` and `ENABLE_DEBUG_PRINT_FLAG=0`.





## 3. How to run `transformer.o` in QEMU



After compiling `transformer.o`, QEMU SVE vector length can be selected with `sve-default-vector-length`.

Remember to change `SVE_LANE` in the notebook accordingly (`SVE_LANE` = sve-default-vector-length in bits / 32).

Examples from the compile script comments (adjust with your current QEMU path):

```bash
/home/thu/opt/qemu-sve/bin/qemu-aarch64 -cpu max,sve=on,sve-default-vector-length=16 -L "$SYSROOT" ./transformer.o

# 128-bit SVE
-cpu max,sve=on,sve-default-vector-length=16

# 256-bit SVE
-cpu max,sve=on,sve-default-vector-length=32

# 512-bit SVE
-cpu max,sve=on,sve-default-vector-length=64
```



## 4. How to Run `transformer.o` in gem5

This section explains how to run the compiled Transformer executable `transformer.o` in gem5 full-system mode.

The gem5 launch script used in this project is:

```text
gem5/launch_simulation_session.sh
```

This script should be placed in the root directory of the gem5 simulator, for example:

```text
/home/thu/gem5/launch_simulation_session.sh
```

The script starts gem5 in a detached `screen` session, creates a timestamped output directory, and writes gem5 logs and statistics files automatically.

---

### 1. Prepare the gem5 launch script

From the root directory of gem5, run:

```bash
bash launch_simulation_session.sh
```

The script creates directories such as:

```text
output/run_YYYYMMDD_HHMMSS/
logs/
```

and starts a screen session such as:

```text
gem5_run_YYYYMMDD_HHMMSS
```

To inspect the gem5 log:

```bash
tail -f logs/gem5_YYYYMMDD_HHMMSS.log
```

To attach to the running `screen` session:

```bash
screen -r gem5_run_YYYYMMDD_HHMMSS
```

---

### 2. Shared folder through VirtIO 9P

The script uses VirtIO 9P to share the host TiC-SAT repository with the guest
Linux system inside gem5. 

This allows generated input files, binaries, and experiment outputs to be accessed inside the guest system without rebuilding the disk image for every experiment.

Please adjust the path on your machine accordingly.

In the launch script, the shared folder is specified by:

```bash
--vio-9p=/home/thu/TiC-SAT
```

This means the host-side TiC-SAT repository is exposed to the simulated Linux
system.

Inside the gem5 guest system, mount the shared folder with:

```bash
mkdir -p /mnt
mount -t 9p -o trans=virtio,version=9p2000.L,aname=/home/thu/TiC-SAT gem5 /mnt
```

After mounting, the TiC-SAT repository should be visible at:

```text
/mnt
```

Then the executable can be run from the mounted repository:

```bash
cd /mnt
./transformer.o
```

If the executable does not have permission to run, use:

```bash
chmod +x transformer.o
./transformer.o
```

#### Implementation note: gem5 shared-folder mechanism

The command-line argument added to the gem5 configuration `starter_fs.py` is:

```python
parser.add_argument(
    "--vio-9p",
    type=str,
    default=None,
    help="Path to the host directory to share via VirtIO 9P",
)
```

The PCI device list is then constructed explicitly. The default VirtIO block device is first added for the boot disk. If the `--vio-9p` option is provided, a VirtIO 9P device is appended to the same PCI device list:

```python
my_pci_devices = [
    PciVirtIO(vio=VirtIOBlock(image=create_cow_image(args.disk_image)))
]

if args.vio_9p:
    vio_9p_device = VirtIO9PDiod()
    vio_9p_device.root = args.vio_9p
    vio_9p_device.queueSize = 128

    sock_path = os.path.abspath(
        os.path.join(m5.options.outdir, "9p.sock")
    )

    if os.path.exists(sock_path):
        os.remove(sock_path)

    vio_9p_device.socketPath = sock_path
    my_pci_devices.append(PciVirtIO(vio=vio_9p_device))

system.pci_devices = my_pci_devices
```

In this project, the related gem5 configuration can be checked in either the standalone `gem5` repository or the consolidated copy under `TiC-SAT/gem5`:

```text
gem5/configs/example/arm/starter_fs.py
```

or, inside the consolidated `TiC-SAT` repository:

```text
TiC-SAT/gem5/starter_fs.py
```

Search for `vio_9p` in `starter_fs.py` to find the shared-folder support.

---

### 3. Connect to the gem5 guest terminal

After gem5 starts, connect to the simulated Linux terminal with either:

```bash
./util/term/gem5term localhost 3456
```

Inside the guest terminal, mount the 9P shared folder and run `transformer.o`.

---

### 4. SVE vector-length configuration

The SVE vector length is configured in the gem5 ARM system configuration.

In `ArmSystem.py`, the SVE vector length parameter is defined as:

```python
sve_vl = Param.SveVectorLength(
    1, "SVE vector length in quadwords (128-bit)"
)
```

The value is measured in 128-bit quadwords:

| `sve_vl` value | SVE vector length |
| -------------: | ----------------: |
|            `1` |           128 bit |
|            `2` |           256 bit |
|            `4` |           512 bit |
|            `8` |          1024 bit |
|           `16` |          2048 bit |

In `configs/example/arm/starter_fs.py`, set the SVE vector length after the
system is created:

```python
root = Root(full_system=True)
root.system = create(args)

# Change SVE vector length.
root.system.sve_vl = 2
```

For example:

```python
root.system.sve_vl = 1   # 128-bit SVE
root.system.sve_vl = 2   # 256-bit SVE
root.system.sve_vl = 4   # 512-bit SVE
```

The SVE vector length used in gem5 should be consistent with the experiment configuration generated by the notebook, especially the `SVE_LANES` setting.

For FP32 lanes:

```text
SVE_LANES = SVE vector length / 32 bits
```

For example:

| SVE vector length | FP32 lanes |
| ----------------: | ---------: |
|           128 bit |    4 lanes |
|           256 bit |    8 lanes |
|           512 bit |   16 lanes |
|          1024 bit |   32 lanes |
|          2048 bit |   64 lanes |

---

### 5. Kernel requirement for 9P and SVE

To use both the shared 9P folder and SVE execution, the Linux kernel used by gem5 must support:

```text
VirtIO 9P
ARM SVE
```

In the launch script, the kernel is selected by:

```bash
--kernel=$HOME/kernel-build/linux/vmlinux
```

The kernel file is large, so it is not included in the GitHub repository.
Place the locally built kernel in the expected path, or update the `--kernel` argument to point to the correct `vmlinux` file.

If you have any questions, please feel free to contact me.

For example:

```bash
--kernel=$HOME/kernel-build/linux/vmlinux
```

or:

```bash
--kernel=/path/to/your/vmlinux
```

After modifying gem5 VirtIO 9P support files or related gem5 source files, rebuild gem5 before launching the simulation again.

---

### 6. Example gem5 launch command

The current launch script uses a command of this form:

```bash
./build/ARM/gem5.fast \
    -d "${run_dir}" \
    --stats-file="${stats_filename}" \
    --dump-config="${config_filename}" \
    configs/example/arm/starter_fs.py \
    --kernel=$HOME/kernel-build/linux/vmlinux \
    --disk-image=../gem5_resources/arm64-ubuntu-20220727.img \
    --interactive-terminal \
    --vio-9p=/home/thu/TiC-SAT \
    --restore=/home/thu/gem5/output/run_YYYYMMDD_HHMMSS/cpt.xxxxxxxxx \
    --cpu=minor
```

Important arguments:

| Argument                             | Meaning                                            |
| ------------------------------------ | -------------------------------------------------- |
| `-d "${run_dir}"`                    | Output directory for this gem5 run.                |
| `--stats-file="${stats_filename}"`   | gem5 statistics output file.                       |
| `--dump-config="${config_filename}"` | Dumps the gem5 system configuration.               |
| `--kernel=.../vmlinux`               | Linux kernel with SVE and 9P support.              |
| `--disk-image=...`                   | Guest Linux disk image.                            |
| `--interactive-terminal`             | Enables interactive terminal access.               |
| `--vio-9p=/home/thu/TiC-SAT`         | Shares the host TiC-SAT repository with the guest. |
| `--restore=.../cpt.xxxxx`            | Restores from an existing checkpoint.              |
| `--cpu=minor`                        | Runs the benchmark on gem5 `MinorCPU`.             |

The restored checkpoint should be compatible with the selected kernel and SVE
configuration.

---

### 7. Run `transformer.o` inside gem5

After connecting to the gem5 terminal, run:

```bash
mkdir -p /mnt
mount -t 9p -o trans=virtio,version=9p2000.L,aname=/home/thu/TiC-SAT gem5 /mnt
cd /mnt
./transformer.o
```

The C/C++ layer-wise outputs will be written to:

```text
weights/multiple_learner_outputs/c/
```

The gem5 statistics will be written to the timestamped output directory created by the launch script, for example:

```text
output/run_YYYYMMDD_HHMMSS/stats_YYYYMMDD_HHMMSS.txt
```

When `GEM5_PROFILE_REGIONS_FLAG=1` is enabled during compilation, the program emits gem5 profiling checkpoints for different Transformer regions, which can be used for layer-level performance analysis.

### 8. Add the run to the profiling tables

gem5 does not number experiments. E01-E36 in `transformer_profiling/final/`
were converted from their stats files after the runs, and a new run is added
the same way with `transformer_profiling/add_experiment.py`. The script writes
the seven-row TSV (six `interval_delta` rows plus `final_total`), adds the
experiment to `manifest.tsv` and `final_all_experiments.tsv`, and picks the next
free ID (E37, E38, ...) from `manifest.tsv`.

Run it from the TiC-SAT root after the program has finished in gem5:

```bash
python3 transformer_profiling/add_experiment.py \
    --stats /home/thu/gem5/output/run_YYYYMMDD_HHMMSS/stats_YYYYMMDD_HHMMSS.txt \
    --study "Learner scaling" --model BERT-mini \
    --n-learners 4 --codebook-size 8 --sve-bits 256
```

For a dense no-SIMD baseline, replace `--codebook-size 8` with `--dense`. Use
the values the run was built and simulated with: `N_LEARNERS` and
`CODEBOOK_SIZE` from the notebook, and the `sve_vl` set in `starter_fs.py`
(1 = 128, 2 = 256, 4 = 512 bits). Add `--dry-run` to print the rows without
writing anything.

| Option | Use |
| --- | --- |
| `--exp-id E40` | Choose the ID yourself instead of the next free one. |
| `--replace` | Overwrite an ID that already exists. |
| `--start-block N` | The stats file holds more than one program run; the run to add starts at dump block `N`. |
| `--model NAME --dims D_Q D_SEQ D_MODEL NUM_HEAD D_FF` | A model other than `BERT-mini` or `BERT-base`. |
| `--gem5-timestamp run_YYYYMMDD_HHMMSS` | The stats path does not contain the run timestamp. |
| `--recorded-stats-path PATH` | Record a different path in the `stats_file` column, e.g. the server path of a copied file. |
| `--scale-first-learner` | Unfinished dense multi-learner run: first learner times `N_LEARNERS`, as for BERT-base E05/E06. |

A profiling run writes six dump blocks (6 per learner for a dense
multi-learner baseline). A run ended with `m5 exit` has one more block, which
the script ignores. If gem5 prints a malformed `simSeconds`, the script uses
`simTicks / simFreq` and prints a warning. Rerun
`python3 tests/profiling_add_experiment_test.py` after changing the script.





## 5. Practical Experiment Checklist

Before compiling:

1. Confirm the active dimensions in `transformer.h`.
2. Regenerate notebook artifacts if model dimensions, learner count, codebook size, same-sequence mode, or FP32 support changed.
3. Confirm `Full_NN/gemm_definitions/generated_codebook_registry.h` contains all required layer names:

```text
q_h0/k_h0/v_h0 ... q_hN/k_hN/v_hN
condense
ff0
ff1
```

4. Confirm `Full_NN/gemm_definitions/codebooks_def.h` has the intended `N_LEARNERS`.
5. For full int8 interleaving, use `N_LEARNERS=2` or `N_LEARNERS=4`,
   `USE_CODEBOOK_GEMM_FLAG=1`, and `SIMD_FLAG=1`.
6. For FP32 Transformer, use `USE_FP32_TRANSFORMER_FLAG=1`,
   `USE_CODEBOOK_GEMM_FLAG=1`, and `ENABLE_CODEBOOK_REFERENCE_FLAG=0`.
7. For performance profiling, keep `ENABLE_DEBUG_PRINT_FLAG=0` and
   `ENABLE_CODEBOOK_REFERENCE_FLAG=0`.

After the gem5 run:

8. Add the stats file to the profiling tables with
   `transformer_profiling/add_experiment.py` (Section 4.8).



## 6. Troubleshooting

### Missing generated registry layer

If `LayerFactory` or `FloatCodebookDense` reports that a layer is missing, rerun
`Full_NN/generators/Transformer_generator.ipynb` and confirm that the generated
registry includes all Q/K/V heads plus `condense`, `ff0`, and `ff1`.

### Shape mismatch

If a generated layer shape does not match the C++ Transformer dimensions, check:

```text
transformer.h
Full_NN/gemm_definitions/input_matrix.h
Full_NN/gemm_definitions/generated_codebook_registry.h
Full_NN/gemm_definitions/codebooks_def.h
```

The generator and C++ code must agree on `D_SEQ`, `D_MODEL`, `D_Q`, `NUM_HEAD`,
and `D_FF`.



## Appendix: Overall File Structure

```text
TiC-SAT/
├── compile_transformer.sh
├── README_I2CE_Transformer.md
├── transformer.cpp
├── transformer.h
├── transformer.o
├── USER_MANUAL.md
├── README_figures/
│   └── simd_nested_loop_2D_refined_names_interleaved_inputs_v2.png..
├── simulation_takeaway.txt
├── Full_NN/  # 81 files
│   ├── gemm_definitions/
│   │   ├── codebooks_def.h
│   │   ├── gemm_data.h
│   │   ├── gemm_header_0.h ... gemm_header_38.h
│   │   ├── generated_codebook_registry.h
│   │   └── input_matrix.h
│   ├── generators/
│   │   ├── Transformer_GEMM_generator.ipynb
│   │   ├── Transformer_generator.ipynb
│   │   ├── codebooks_defs_generator.py
│   │   ├── dense_layer_generator.py
│   │   ├── gemm_layer_generator.py
│   │   ├── input_image_gen.py
│   │   ├── transformer_debug_utils.py
│   │   ├── templates/
│   │   └── __pycache__/
│   ├── inc/
│   │   ├── SVE_implementations.h
│   │   ├── conv_def.h
│   │   ├── conv_exec.h
│   │   ├── dense_SVE.h
│   │   ├── dense_exec.h
│   │   ├── gemm_SVE.h
│   │   └── gemm_exec.h
│   └── src/
│       ├── conv_SVE.c
│       ├── conv_def.c
│       ├── conv_exec.c
│       ├── dense_SVE.c
│       ├── dense_SVE_f16.c
│       ├── dense_exec.c
│       └── gemm_exec.c
├── gem5/
│   ├── launch_simulation_session.sh
│   └── vmlinux (This kernel is not visible in the repository as it's too large)
│
├── transformer_layers/  # 43 files
│   ├── addNorm.cc/.h
│   ├── codebookDense.cc/.h
│   ├── debuggerFunctions.cc/.h
│   ├── dense.cc/.h
│   ├── floatAddNorm.cc/.h
│   ├── floatCodebookDense.cc/.h
│   ├── floatCommon.h
│   ├── floatDump.cc/.h
│   ├── floatSelfAttention.cc/.h
│   ├── floatSoftmax.cc/.h
│   ├── floatTransformerBlock.cc/.h
│   ├── interleavedPipeline.cc/.h
│   ├── layerFactory.cc/.h
│   ├── linearLayer.h
│   ├── profile.cc/.h
│   ├── registry_adapter.h
│   ├── run_mode_config.h
│   ├── selfattention.cc/.h
│   ├── softmax.cc/.h
│   ├── transformerBlock.cc/.h
│   ├── transformerBlockInterleavedHelpers.cc/.h
│   ├── transformerFloat.cc/.h
│   ├── transpose.cc/.h
│   └── util.h
└── weights/  # 1071 files
    ├── H-1_L-1.bin, H-1_L0.bin ... H11_L2.bin
    ├── condense_out_after_addnorm.txt
    ├── condense_out_pre_addnorm.txt
    ├── ffn0_output.bin/.txt
    ├── ffn1_output_after_addnorm.txt
    ├── ffn1_output_pre_addnorm.bin/.txt
    ├── output.bin
    ├── generated_from_notebook/
    │   ├── learner0/
    │   ├── learner1/
    │   ├── learner2/
    │   └── learner3/
    ├── multiple_learner_outputs/
    │   ├── c/
    │   └── python/
    └── temporary_storage/
```

