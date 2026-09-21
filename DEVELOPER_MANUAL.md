# Developer Manual

## Document Purpose and Reading Order

Please read the semester project report first before reading this manual.  
The report provides the high-level background, design motivation, evaluation methodology, and experimental conclusions.

After that, please read `USER_MANUAL.md` if your goal is to compile, run, verify, or reproduce the experiments.

This **Developer Manual** is intended for future developers who need to understand, maintain, or extend the implementation.  
It gives a more detailed explanation of the code structure, generated files, Transformer execution paths, function-call maps, interleaved multi-learner pipeline, GEMM/SVE kernels, and FP32 implementation.

## Table of Contents

1. [Generator](#1-generator)  
   1.1 [Generator File Structure](#11-generator-file-structure)  
   1.2 [Generated Transformer Layer Names](#12-generated-transformer-layer-names)  
   1.3 [Generated Codebook GEMM Files](#13-generated-codebook-gemm-files)  
   1.4 [Generated I2-CE Hyperparameter Configuration](#14-generated-i2-ce-hyperparameters-configuration)  
   1.5 [Generated Dense Transformer Weights](#15-generated-dense-transformer-weights)  
   1.6 [Python Reference Outputs](#16-python-reference-outputs)  
   1.7 [Common Generated Naming Convention](#17-common-generated-naming-convention)

2. [Top-Level Transformer Code](#2-top-level-transformer-code)

3. [Integer Transformer Layer Code](#3-integer-transformer-layer-code)  
   3.1 [Overview of Each Layer](#31-overview-of-each-layer)  
   3.2 [Layer Creation](#32-layer-creation)  
   3.3 [Function Call Maps](#33-function-call-maps)  
   3.4 [Summary of Used Functions](#34-summary-of-used-functions)

4. [GEMM and SVE Code](#4-gemm-and-sve-code)

5. [Floating-Point Transformer Code](#5-floating-point-transformer-code)

   

---

## 1. Generator

The generator files are located under:

```text
Full_NN/generators/
```

The main notebook is:

```text
Full_NN/generators/Transformer_generator.ipynb
```

This notebook is used to generate the data required by the C/C++ Transformer
implementation, including:

- Transformer input tensors,
- dense `.bin` weights,
- codebook values,
- packed codebook indices,
- generated C/C++ `.h` headers,
- Python reference outputs for correctness checking.

The generated outputs are later consumed by the integer Transformer path, the
FP32 Transformer path, and the codebook GEMM implementation.

---

### 1.1 Generator file structure

```text
Full_NN/
└── generators/
    ├── Transformer_generator.ipynb
    ├── Transformer_GEMM_generator.ipynb
    ├── codebooks_defs_generator.py
    ├── dense_layer_generator.py
    ├── gemm_layer_generator.py
    ├── input_image_gen.py
    └── transformer_debug_utils.py
```

Important generator files:

| File                                                  | Function                                                     |
| ----------------------------------------------------- | ------------------------------------------------------------ |
| `Full_NN/generators/Transformer_generator.ipynb`      | Main notebook used to generate Transformer inputs, dense weights, codebook definitions, packed indices, generated C/C++ `.h` headers, `.bin` weight files, and Python reference outputs. |
| `Full_NN/generators/gemm_layer_generator.py`          | Helper used by the notebook to generate GEMM layer headers, compact codebook/index data, and TiC-SAT-compatible weight files. |
| `Full_NN/generators/codebooks_defs_generator.py`      | Generates global codebook configuration, such as learner count, codebook size, bits per codebook index, SVE packing constants, same-sequence mode, and FP32 mode flag. |
| `Full_NN/generators/dense_layer_generator.py`         | Helper used to generate dense layer data for the default dense Transformer path. |
| `Full_NN/generators/transformer_debug_utils.py`       | Python-side reference/debug utilities for comparing notebook-generated tensors with C/C++ output dumps. |
| `Full_NN/generators/Transformer_GEMM_generator.ipynb` | Earlier or auxiliary notebook for GEMM-level generation and testing. |
| `Full_NN/generators/input_image_gen.py`               | Legacy/helper input generation utility inherited from the original TiC-SAT/NN-layer flow. |

---

### 1.2 Generated Transformer layer names

The notebook generates Transformer layer names that match the C++ layer factory.
These names are used by the generated codebook registry and later looked up by
`LayerFactory`, `CodebookDense`, and the FP32 implementation.

Typical generated layer names are:

```text
q_h0, k_h0, v_h0
q_h1, k_h1, v_h1
...
condense
ff0
ff1
```

The naming convention is:

| Logical layer                | Generated layer name |
| ---------------------------- | -------------------- |
| Query projection of head `h` | `q_h<h>`             |
| Key projection of head `h`   | `k_h<h>`             |
| Value projection of head `h` | `v_h<h>`             |
| Multi-head output projection | `condense`           |
| First feed-forward layer     | `ff0`                |
| Second feed-forward layer    | `ff1`                |

---

### 1.3 Generated codebook GEMM files

The generated codebook GEMM files are stored under:

```text
Full_NN/gemm_definitions/
```

These files are used when `USE_CODEBOOK_GEMM_FLAG=1`.

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

Important generated codebook files:

| Generated file or directory                              | Function                                                     |
| -------------------------------------------------------- | ------------------------------------------------------------ |
| `Full_NN/gemm_definitions/codebooks_def.h`               | Global generated configuration, including `N_LEARNERS`, `N_SVE_LANES`, codebook size, bits per codebook index, same-sequence mode, and FP32 mode flag. |
| `Full_NN/gemm_definitions/input_matrix.h`                | Generated FP32 input matrix used by the FP32 Transformer path. |
| `Full_NN/gemm_definitions/gemm_header_*.h`               | Layer-specific generated compact GEMM data, including packed indices, codebooks, and bias arrays. |
| `Full_NN/gemm_definitions/generated_codebook_registry.h` | Registry that maps layer names such as `q_h0`, `k_h0`, `v_h0`, `condense`, `ff0`, and `ff1` to their generated codebook/index/bias arrays. This registry is used by `LayerFactory`, `CodebookDense`, and the FP32 implementation. |
| `Full_NN/gemm_definitions/gemm_data.h`                   | Generated GEMM data header used by the compact GEMM implementation. |

---

### 1.4 Generated I2-CE HYPERPARAMETERS Configuration

The notebook also provides `Full_NN/gemm_definitions/codebooks_def.h` to controls generated experiment metadata. Those are inherited or calculated from the hyperparameters which are set in the notebook.

Typical fields:

| Macro           | Meaning                                                      |
| --------------- | ------------------------------------------------------------ |
| `N_LEARNERS`    | Number of learner models generated.                          |
| `N_SVE_LANES`   | Number of FP32 SVE lanes assumed by generated constants.     |
| `N_SVE_BYTE`    | Number of int8 values corresponding to the SVE vector length setting. |
| `CB_SIZE`       | Codebook size.                                               |
| `BITS_PER_CB`   | Number of bits per packed codebook index.                    |
| `IDXS_PER_WORD` | Number of packed codebook indexes in one 32-bit word.        |
| `IDX_MASK`      | Mask used to extract one packed codebook index.              |
| `SAME_SEQ`      | Whether the generated layers use the same weight-index stream across learners. |
| `USE_F32`       | Generated flag indicating FP32 generated data support.       |

For codebook experiments, regenerate the notebook outputs after changing model dimensions, learner count, codebook size, or same-sequence/different-sequence settings.

---

### 1.5 Generated dense Transformer weights

The default dense Transformer weights are stored under:

```text
weights/
```

These files are used by the default dense Transformer path, the dense no-SIMD
baseline, and some debug/reference modes.

```text
weights/
├── generated_from_notebook/
│   ├── learner0/
│   ├── learner1/
│   ├── learner2/
│   └── learner3/
└── H-1_L-1.bin
```

Important dense generated files:

| Generated file or directory                            | Function                                                     |
| ------------------------------------------------------ | ------------------------------------------------------------ |
| `weights/generated_from_notebook/learner*/H*.bin`      | Notebook-generated dense binary weights for each learner. These are used when `USE_NOTEBOOK_GENERATED_WEIGHTS_FLAG=1`. |
| `weights/generated_from_notebook/learner*/H-1_L-1.bin` | Notebook-generated input tensor for each learner, when stored inside learner-specific directories. |
| `weights/H-1_L-1.bin`                                  | Legacy/shared generated input file path used by some reload paths. |

---

### 1.6 Python reference outputs

The notebook also generates Python reference outputs for layer-wise correctness
checking.

These outputs are stored under:

```text
weights/
└── multiple_learner_outputs/
    └── python/
```

After running the C/C++ executable, the C/C++ outputs are dumped under:

```text
weights/
└── multiple_learner_outputs/
    └── c/
```

The final comparison is therefore performed between:

```text
weights/multiple_learner_outputs/python/
weights/multiple_learner_outputs/c/
```

This is useful for checking whether the C/C++ Transformer implementation matches
the Python reference implementation.

---

### 1.7 Common generated naming convention

The notebook-generated file names follow the original TiC-SAT layer naming
style.

| Logical tensor/layer  | Typical generated file or layer name |
| --------------------- | ------------------------------------ |
| Input tensor          | `H-1_L-1.bin`                        |
| Query head `h`        | `q_h<h>` and usually `H<h>_L0.bin`   |
| Key head `h`          | `k_h<h>` and usually `H<h>_L1.bin`   |
| Value head `h`        | `v_h<h>` and usually `H<h>_L2.bin`   |
| Multi-head projection | `condense` and usually `H-1_L0.bin`  |
| Feed-forward 0        | `ff0` and usually `H-1_L1.bin`       |
| Feed-forward 1        | `ff1` and usually `H-1_L2.bin`       |

The exact dimensions are controlled by `Transformer_generator.ipynb`. They must
match both:

```text
transformer.h
```

and the generated registry:

```text
Full_NN/gemm_definitions/generated_codebook_registry.h
```

If the Transformer dimensions, learner count, codebook size, same-sequence mode,
or FP32/int8 mode are changed, the notebook outputs should be regenerated before
compiling the C/C++ implementation again.



---

## 2. Top-Level Transformer Code

The top-level Transformer executable is built from:

```text
transformer.cpp
transformer.h
compile_transformer.sh
```

```text
TiC-SAT/
├── transformer.cpp
├── transformer.h
└── compile_transformer.sh
```

Important files:

| File                     | Role                                                         |
| ------------------------ | ------------------------------------------------------------ |
| `transformer.cpp`        | Main program entry point. It selects the execution mode, loads generated inputs/weights, creates Transformer blocks, and runs single-learner or multi-learner inference. |
| `transformer.h`          | Defines Transformer dimensions such as `D_Q`, `D_SEQ`, `D_MODEL`, `NUM_HEAD`, and `D_FF`. These values must match the notebook settings. |
| `compile_transformer.sh` | Compiles the C/C++ Transformer implementation and produces `transformer.o`. |



The top-level execution is controlled by `transformer.cpp`. This file is responsible for selecting the Transformer execution path, loading the required weights and inputs, creating Transformer blocks, and starting the actual computation.

If the FP32 Transformer macro is enabled, the program directly dispatches to the FP32 Transformer implementation:

```cpp
#if CFG_USE_FP32_TRANSFORMER
    TransformerFloat::run(
        learner_count,
        dump_outputs_enabled ? dir_name + "/multiple_learner_outputs/c" : std::string());
    return;
#endif
```

In this case, execution continues in:

```text
transformer_layers/transformerFloat.cc
```

The FP32 path is used when:

```text
USE_FP32_TRANSFORMER_FLAG=1
```

---

For the integer Transformer path, `transformer.cpp` supports several execution
modes:

| Mode                                        | Description                                                  |
| ------------------------------------------- | ------------------------------------------------------------ |
| Default dense Transformer                   | Runs the original dense Transformer implementation.          |
| Dense no-SIMD baseline                      | Runs multiple dense learners sequentially without SIMD, mainly for baseline profiling. |
| Codebook GEMM, single learner               | Runs one learner using generated codebook GEMM layers.       |
| Codebook GEMM, multiple learners            | Runs multiple codebooked learners using grouped execution.   |
| Fully interleaved codebook SIMD             | Keeps learner activations interleaved across the full Transformer block and uses SVE kernels. |
| Non-fully interleaved grouped codebook GEMM | Uses grouped/interleaved GEMM kernels only around selected dense layers, but does not keep the whole block interleaved. |
| Dense reference / correctness checking      | Loads dense reference weights and compares outputs against the codebook implementation when reference checking is enabled. |



The exact execution mode is selected by the compile-time flags described in Section 2.3, **Compile Command Examples**, of `USER_MANUAL.md`, especially:

```text
USE_FP32_TRANSFORMER_FLAG
DENSE_NO_SIMD_BASELINE_FLAG
USE_CODEBOOK_GEMM_FLAG
FULL_INTERLEAVED_PIPELINE_FLAG
SIMD_FLAG
ENABLE_CODEBOOK_REFERENCE_FLAG
```

---

After loading the required data and initializing the Transformer blocks,
`transformer.cpp` starts computation by calling one of the following entry
points:

| Function                        | Used when                                                    |
| ------------------------------- | ------------------------------------------------------------ |
| `compute(...)`                  | Single-learner execution or normal non-grouped execution.    |
| `computeWithoutStatsReset(...)` | Computation without resetting gem5 statistics around the block. |
| `computeWithStatsReset(...)`    | Computation with gem5 statistics reset/dump around the block. |
| `computeGroup2(...)`            | Grouped execution for 2 learners.                            |
| `computeGroup4(...)`            | Grouped execution for 4 learners.                            |

At a high level, the dispatch flow is:

```text
main()
  -> test()
      -> load input and weights
      -> determine learner count
      -> if CFG_USE_FP32_TRANSFORMER:
             TransformerFloat::run(...)
             return

      -> create integer TransformerBlock objects

      -> if learner_count == 1:
             compute(...)
         else if learner_count == 2:
             computeGroup2(...)
         else if learner_count == 4:
             computeGroup4(...)
```

Therefore, `transformer.cpp` mainly acts as the top-level controller. The real Transformer layer computation is implemented inside `transformer_layers/`, while the compact GEMM and SVE kernels are implemented in `Full_NN/src/`.





---

## 3. Integer Transformer Layer Code

### 3.1 Overview of each layer

The integer Transformer implementation is located under:

```text
transformer_layers/
```

Main file structure:

```text
transformer_layers/
├── transformerBlock.cc/.h
├── selfattention.cc/.h
├── codebookDense.cc/.h
├── layerFactory.cc/.h
├── linearLayer.h
├── dense.cc/.h
├── addNorm.cc/.h
├── softmax.cc/.h
├── interleavedPipeline.cc/.h
├── transformerBlockInterleavedHelpers.cc/.h
├── debuggerFunctions.cc/.h
├── transpose.cc/.h
├── profile.cc/.h
├── registry_adapter.h
└── run_mode_config.h
```

Important files:

| File                                       | Role                                                         |
| ------------------------------------------ | ------------------------------------------------------------ |
| `transformerBlock.cc/.h`                   | Implements the integer Transformer block, including attention, projection, feed-forward layers, AddNorm, and grouped learner execution. |
| `selfattention.cc/.h`                      | Implements integer self-attention, including Q/K/V projection, attention score computation, softmax, and softmax-times-V. |
| `codebookDense.cc/.h`                      | Implements integer codebook-backed dense layers and dispatches compact GEMM kernels. |
| `layerFactory.cc/.h`                       | Creates either `CodebookDense` or fallback `Dense` layers depending on compile-time macros and generated registry availability. |
| `linearLayer.h`                            | Defines the common layer interface used by dense and codebook dense layers. |
| `dense.cc/.h`                              | Implements the default dense integer layer used as fallback or reference. |
| `addNorm.cc/.h`                            | Implements integer residual addition and layer normalization. |
| `softmax.cc/.h`                            | Implements integer/LUT-based softmax.                        |
| `interleavedPipeline.cc/.h`                | Contains helper functions for the fully interleaved int8 multi-learner pipeline. |
| `transformerBlockInterleavedHelpers.cc/.h` | Contains helper functions and validation utilities for fully interleaved Transformer execution. |
| `debuggerFunctions.cc/.h`                  | Contains debug printing, tensor dumping, comparison, and weight utility functions. |
| `transpose.cc/.h`                          | Contains transpose helpers used in attention computation.    |

Important fully interleaved integer functions:

| Function                                  | File                     | Role                                              |
| ----------------------------------------- | ------------------------ | ------------------------------------------------- |
| `AddNormalize::computeInterleaved2Learners(...)` | `addNorm.cc`             | AddNorm for 2 interleaved learners.               |
| `AddNormalize::computeInterleaved4Learners(...)` | `addNorm.cc`             | AddNorm for 4 interleaved learners.               |
| `Softmax::computeInterleaved2Learners(...)`      | `softmax.cc`             | Softmax for 2 interleaved learners.               |
| `Softmax::computeInterleaved4Learners(...)`      | `softmax.cc`             | Softmax for 4 interleaved learners.               |
| `matmulInterleaved2LearnersToInt8(...)`          | `interleavedPipeline.cc` | Interleaved int8 attention matmul for 2 learners. |
| `matmulInterleaved4LearnersToInt8(...)`          | `interleavedPipeline.cc` | Interleaved int8 attention matmul for 4 learners. |

### 3.2 Layer Creation

All integer Transformer dense/projection layers are created through
`LayerFactory`:

```text
TransformerBlock constructor
  -> LayerFactory::create("condense", ...)
  -> LayerFactory::create("ff0", ...)
  -> LayerFactory::create("ff1", ...)

SingleHeadSelfAttn constructor
  -> LayerFactory::create("q_hX", ...)
  -> LayerFactory::create("k_hX", ...)
  -> LayerFactory::create("v_hX", ...)
```

`LayerFactory::create(...)` behavior:

```text
if CFG_USE_CODEBOOK_GEMM:
    look up layer name in generated_codebook_registry.h
    if found and shape matches:
        return CodebookDense
    if profile/codebook-only mode:
        throw error if missing
    otherwise:
        fall back to Dense
else:
    return Dense
```

If `CFG_USE_CODEBOOK_REFERENCE=1`, the codebook path also creates dense reference layers for comparison.



### 3.3 Function Call Maps

This section summarizes the main function-call and data-flow paths used by the Transformer implementation. Debug printing, output dumping, and profiling details are omitted here so that the main computation flow is easier to follow.

All execution modes compute the same Transformer block structure:

```text
input
  -> for each attention head:
       Q = input * Wq
       K = input * Wk
       V = input * Wv
       scores = Q * K^T
       probs = softmax(scores)
       head_out = probs * V
  -> concatenate all heads
  -> output projection / condense
  -> AddNorm(input, condense_out)
  -> FF0
  -> FF1
  -> AddNorm(after_attn_addnorm, FF1_out)
  -> output
```

The common top-level entry point is:

```text
transformer.cpp
  -> main()
      -> test()
```

Inside `test()`, the program loads the required inputs and weights, determines
the number of learners, creates Transformer blocks, and dispatches to the
selected execution path.



All function-call maps described in this section correspond to the run modes and compile commands listed in Section 2.3, **Compile Command Examples**, of `USER_MANUAL.md`.

---

#### 3.3.1 DENSE baseline  
Default dense Transformer, originally implemented by TiC-SAT.

Typical flags:

```text
USE_CODEBOOK_GEMM_FLAG=0
USE_FP32_TRANSFORMER_FLAG=0
DENSE_NO_SIMD_BASELINE_FLAG=0
SIMD_FLAG=0
```

This mode runs the default dense integer Transformer path for one learner. Since
`USE_CODEBOOK_GEMM_FLAG=0`, every linear layer created by `LayerFactory` becomes
a normal `Dense` layer.

High-level call flow:

```text
main()
  -> test()
      -> getTransformerLearnerCount()
      -> buildTransformerBlockForLearner()
      -> TransformerBlock::compute()
          -> TransformerBlock::computeBody()
```

Inside `TransformerBlock::computeBody()`:

```text
for each attention head:
    SingleHeadSelfAttn::compute()
        -> Dense::compute(q_hN)
        -> Dense::compute(k_hN)
        -> Dense::compute(v_hN)
        -> Transpose::transpose(K)
        -> smmComputeRWMA(Q, K^T)
        -> Softmax::compute()
        -> smmComputeRWMA(softmax, V)
        -> Softmax::post_softmax()

Transpose::multihead_transpose()

Dense::compute(condense)
AddNormalize::compute(input, condense_out)

Dense::compute(ff0)
Dense::compute(ff1)

AddNormalize::compute(after_attn_addnorm, ff1_out)
```

The data flow is:

```text
input
  -> dense Q/K/V projection
  -> dense attention computation
  -> dense output projection
  -> AddNorm
  -> dense feed-forward layers
  -> AddNorm
  -> output
```

This path does not use `CodebookDense`, interleaved learner layouts, or SVE
compact GEMM kernels. This is the default dense Transformer calls of TiC-SAT



**<u>The complete function call map is:</u>**

Used when learner count is `1`, or when the default dense path is selected:

```text
TransformerBlock::compute(...)
  -> TransformerBlock::computeWithStatsLabel("single_transformer_block")
      -> TransformerBlock::computeBody(...)
```

Inside `computeBody(...)`:

```text
for each attention head:
    SingleHeadSelfAttn::compute(...)
        -> query_layer_->compute(...)
             -> Dense::compute(...) or CodebookDense::compute(...)
                 -> CodebookDense::runCompactGemm(...)
                     -> gemm_exec_compact_int(...)
                     -> or gemm_exec_compact_int_sve(...) when SIMD
        -> key_layer_->compute(...)
        -> value_layer_->compute(...)
        -> Transpose::transpose(...) or Transpose::transpose_rearranged(...)
        -> attention Q*K matmul:
             -> smmComputeRWMA(...) or simdComputeRWMA(...)
             -> or smmComputeBWMA(...) or simdComputeBWMA(...) for BWMA
        -> Softmax::compute(...)
             -> or Softmax::computeRearranged(...) for BWMA
        -> attention softmax*V matmul:
             -> smmComputeRWMA(...) or simdComputeRWMA(...)
             -> or smmComputeBWMA(...) or simdComputeBWMA(...) for BWMA
        -> Softmax::post_softmax(...)

Transpose::multihead_transpose(...) if needed

condense->compute(...)
    -> Dense::compute(...) or CodebookDense::compute(...)

AddNormalize::compute(...)
    -> or AddNormalize::computeRearranged(...) for BWMA

feedForward0->compute(...)
feedForward1->compute(...)

AddNormalize::compute(...)
```





---

#### 3.3.2 DENSE no-SIMD baseline  
Dense sequential multi-learner baseline, used for E01--E06.

Typical flags:

```text
DENSE_NO_SIMD_BASELINE_FLAG=1
USE_CODEBOOK_GEMM_FLAG=0
USE_FP32_TRANSFORMER_FLAG=0
SIMD_FLAG=0
FULL_INTERLEAVED_PIPELINE_FLAG=0
```

This mode is used to construct fair dense baselines for multiple learners. The mathematical computation is the same as the default dense Transformer path, but the execution organization is different.

Instead of loading and computing one learner at a time, this mode first creates and loads all dense learner blocks. Then it executes the learners sequentially.
This avoids including repeated weight-loading overhead inside the profiled
computation region.

High-level call flow:

```text
main()
  -> test()
      -> getTransformerLearnerCount() = N_LEARNERS
      -> build all DenseNoSimdTransformerBlock objects

      for learner = 0 to N_LEARNERS - 1:
          TransformerBlock::computeWithoutStatsReset()
              -> TransformerBlock::computeBody()
```

The data flow is:

```text
same input
  -> learner 0 dense Transformer block
  -> output 0

same input
  -> learner 1 dense Transformer block
  -> output 1

...

same input
  -> learner N-1 dense Transformer block
  -> output N-1
```

Important characteristics:

```text
No CodebookDense
No SVE
No interleaved layout
No grouped execution
Learners are executed sequentially
```

This mode provides the dense no-SIMD multi-learner baseline used for comparison
against the codebook SIMD implementation.



<u>The function-call map of each learner is exactly the same as the one shown in Section 3.3.1.</u>

---

#### 3.3.3 Int8 codebook GEMM with SVE  
Notice: Some functions like `gemm_exec_*` and `sve_gemm_row_compact_*`  in below code refers to the codebooked SVE GEMM, which is discussed in Section 4, **GEMM and SVE Code**, of `DEVELOPER_MANUAL.md`. Please read them together.

Main int8 codebook SIMD implementation, used for E07--E36.

Typical flags:

```text
USE_CODEBOOK_GEMM_FLAG=1
USE_FP32_TRANSFORMER_FLAG=0
SIMD_FLAG=1
```

When `USE_CODEBOOK_GEMM_FLAG=1`, `LayerFactory` tries to create registry-backed
`CodebookDense` layers for:

```text
q_hN
k_hN
v_hN
condense
ff0
ff1
```

These layers use the generated codebook registry from:

```text
Full_NN/gemm_definitions/generated_codebook_registry.h
```

---

##### 3.3.3.1 Single-learner int8 codebook path

For `N_LEARNERS=1`, the call flow is:

```text
main()
  -> test()
      -> TransformerBlock::compute()
          -> TransformerBlock::computeBody()
```

Each codebooked linear layer follows:

```text
CodebookDense::compute()
  -> CodebookDense::runCompactGemm()
      -> gemm_exec_compact_int_sve()
          -> SVE compact int8 GEMM row kernel
```

The logical data flow is:

```text
input
  -> CodebookDense Q/K/V projection
  -> attention score computation
  -> Softmax
  -> attention output computation
  -> CodebookDense condense
  -> AddNorm
  -> CodebookDense ff0
  -> CodebookDense ff1
  -> AddNorm
  -> output
```

In this mode, codebook GEMM is enabled, but there is only one learner, so multi-learner interleaving does not provide an additional benefit.



---

##### 3.3.3.2 Multi-learner int8 fully interleaved path

For `N_LEARNERS=2` or `N_LEARNERS=4` with:

```text
FULL_INTERLEAVED_PIPELINE_FLAG=1
USE_CODEBOOK_GEMM_FLAG=1
SIMD_FLAG=1
learner_count is 2 or 4
USE_FP32_TRANSFORMER_FLAG=0
```

the program uses the fully interleaved multi-learner integer path.

Top-level call flow:

```text
main()
  -> test()
      -> TransformerBlock::computeGroup2()
          -> TransformerBlock::computeGroup2FullInterleaved()

      or

      -> TransformerBlock::computeGroup4()
          -> TransformerBlock::computeGroup4FullInterleaved()
```

At the beginning of the block, per-learner packed inputs are converted into an
interleaved int8 layout :

```text
int8_t buffer[(seq * feature + col) * learner_count + learner]
```

Since this project is iso-indexed, the inputs for different learners are the same.

```text
separate learner inputs (the same for iso-indexed)
  -> interleavePackedLearners2/4()
  -> [seq][feature][learner] int8 layout
```

The fully interleaved 2-learner or 4-learner data flow is:

```text
for each attention head:
    SingleHeadSelfAttn::computeInterleaved2Learners/4D()
        -> interleaved CodebookDense Q
        -> interleaved CodebookDense K
        -> interleaved CodebookDense V
        -> interleaved Q * K^T
        -> Softmax::computeInterleaved2Learners/4D()
        -> interleaved softmax * V
        -> copyHeadToMultiheadInterleaved2Learners/4D()

interleaved CodebookDense condense
AddNormalize::computeInterleaved2Learners/4D()

interleaved CodebookDense ff0
interleaved CodebookDense ff1

AddNormalize::computeInterleaved2Learners/4D()

packInterleavedLearners2/4()
  -> per-learner outputs
```

The key interleaved GEMM calls are:

```text
CodebookDense::computeInterleaved2LearnersToInt8()
  -> gemm_exec_compact_int_sve_interleaved_2Learners_same_seq()
      -> sve_gemm_row_compact_int8_interleaved_2Learners_same_seq()

CodebookDense::computeInterleaved4LearnersToInt8()
  -> gemm_exec_compact_int_sve_interleaved_4Learners_same_seq()
      -> sve_gemm_row_compact_int8_interleaved_4Learners_same_seq()
```



The different-sequence interleaved path is outside the main scope of this project. In the current implementation, only the 4-learner different-sequence case has been completed:

```text
gemm_exec_compact_int_sve_interleaved_4Learners_diff_seq()
  -> sve_gemm_row_compact_int8_interleaved_4Learners_diff_seq()
```

This path supports learners with different packed index streams. It has been implemented and checked for functional correctness, but it was not included in the profiling experiments. Therefore, the reported performance results focus on the same-sequence shared-index implementation.



**<u>The complete function call map for 2-learner (similar for 4-learner) is:</u>**

```text
TransformerBlock::computeGroup2FullInterleaved(...)
  -> interleavePackedLearners2(...)

  for each head:
      SingleHeadSelfAttn::computeInterleaved2Learners(...)
          -> computeCodebookDenseInterleaved2Learners("q_hX", ...)
              -> CodebookDense::computeInterleaved2LearnersToInt8(...)
                  -> gemm_exec_compact_int_sve_interleaved_2Learners_same_seq(...)
          -> computeCodebookDenseInterleaved2Learners("k_hX", ...)
          -> computeCodebookDenseInterleaved2Learners("v_hX", ...)
          -> matmulInterleaved2LearnersToInt8(...)        // Q * K
              -> sve_gemm_dense_int8_interleaved_2Learners(...) when SIMD
          -> Softmax::computeInterleaved2Learners(...)
          -> transposeInterleavedRowsToCols2(...)  // V layout for matmul
          -> matmulInterleaved2LearnersToInt8(...)        // softmax * V
          -> Softmax::post_softmax_interleaved2D(...)
      -> copyHeadToMultiheadInterleaved2Learners(...)

  -> computeCodebookDenseInterleaved2Learners("condense", ...)
      -> CodebookDense::computeInterleaved2LearnersToInt8(...)

  -> AddNormalize::computeInterleaved2Learners(...)

  -> computeCodebookDenseInterleaved2Learners("ff0", ...)
  -> computeCodebookDenseInterleaved2Learners("ff1", ...)

  -> AddNormalize::computeInterleaved2Learners(...)
  -> packInterleavedLearners2(...)
```



Important characteristics:

```text
Uses CodebookDense
Uses generated packed indices and codebooks
Uses SVE compact GEMM kernels
Uses interleaved learner layout
Keeps learners interleaved across the full Transformer block
```

This is the main shared-index codebook SIMD implementation evaluated in the
int8 experiments.



---

##### 3.3.3.3 Multi-learner int8 grouped but not fully interleaved path

For `N_LEARNERS=2` or `N_LEARNERS=4` with:

```text
FULL_INTERLEAVED_PIPELINE_FLAG=0
```

the program may still use grouped codebook GEMM around selected dense layers, but it does not keep the entire Transformer block interleaved.

In this implementation, only the codebooked GEMM layers use the interleaved multi-learner layout. Other Transformer stages, such as attention score computation, softmax, and AddNorm, still use the inherited implementations from the original dense Transformer path.

Therefore, although the output of a grouped codebooked GEMM layer is produced in an interleaved format, the following non-interleaved layers cannot consume this layout directly. Before entering these inherited layers, the interleaved output must be converted back to the normal learner-by-learner format. These stages are then executed separately for each learner.

This design is functionally correct and allows grouped codebooked GEMM to be used without rewriting the whole Transformer pipeline. However, it introduces extra layout-conversion overhead and sequential per-learner execution overhead. These costs are one of the motivations for the fully interleaved pipeline, where
CodebookDense, attention matmul, softmax, AddNorm, and feed-forward stages all operate on the same interleaved layout.

The call flow is:

```text
TransformerBlock::computeGroup2()
  -> TransformerBlock::computeGroupImpl<2>()

TransformerBlock::computeGroup4()
  -> TransformerBlock::computeGroupImpl<4>()
```

In this path:

```text
Q/K/V CodebookDense may use grouped interleaved GEMM
attention matmul is still performed per learner in for-loop
Softmax is still performed per learner in for-loop
AddNorm is still performed per learner in for-loop
FFN layers may use grouped interleaved GEMM
outputs remain separate per learner between stages
```

This mode is useful for comparison, but the main optimized path in this project
is the fully interleaved pipeline.



**<u>The complete function call map for 2/4 learners but not fully interleaved mode is:</u>**

Entry:

```text
TransformerBlock::computeGroup2(...)
  -> TransformerBlock::computeGroupImpl<2>(...)

TransformerBlock::computeGroup4(...)
  -> TransformerBlock::computeGroupImpl<4>(...)
```

Attention:

```text
for each head:
    SingleHeadSelfAttn::computeGroup2(...)
        -> SingleHeadSelfAttn::computeGroupImpl<2>(...)

    or

    SingleHeadSelfAttn::computeGroup4(...)
        -> SingleHeadSelfAttn::computeGroupImpl<4>(...)
```

For Q/K/V layers, grouped non-full interleaving tries to use grouped
CodebookDense kernels:

```text
tryComputeGroupedCodebookDense2(...)
  -> CodebookDense::computeInterleaved2LearnersSameSeq(...)
      -> gemm_exec_compact_int_interleaved_2Learners_same_seq(...)
      -> or gemm_exec_compact_int_sve_interleaved_2Learners_same_seq(...)

tryComputeGroupedCodebookDense4(...)
  -> CodebookDense::computeInterleaved4Learners(...)
      -> gemm_exec_compact_int_interleaved_4Learners_same_seq(...)
      -> or SVE versions when SIMD
```

If grouped CodebookDense is unsupported for a layer, the code falls back to
per-learner `layer->compute(...)`.

After Q/K/V projection, attention itself is still per learner:

```text
for each learner:
    Transpose::transpose(...)
    smmComputeRWMA(...) or simdComputeRWMA(...)
    Softmax::compute(...)
    smmComputeRWMA(...) or simdComputeRWMA(...)
    Softmax::post_softmax(...)
```

After attention heads:

```text
for each learner:
    copy/transpose multi-head output

condense:
    tryComputeGroupedCodebookDense2/4(...)
    else per-learner condense->compute(...)

for each learner:
    AddNormalize::compute(...)

ff0:
    tryComputeGroupedCodebookDense2/4(...)
    else per-learner feedForward0->compute(...)

ff1:
    tryComputeGroupedCodebookDense2/4(...)
    else per-learner feedForward1->compute(...)

for each learner:
    AddNormalize::compute(...)
```



---

#### 3.3.4 FP32 codebook Transformer with SVE

For the floating point implementations, it will be discussed in  Section 5. **Floating-Point Transformer Code**, of `DEVELOPER_MANUAL.md`



### 3.4 Summary of used functions



#### 3.4.1 Quick Comparison

| Area                      | Not fully interleaved                                        | Fully interleaved                                            |
| ------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| Main activation layout    | Separate per-learner buffers. Integer path uses packed `uint32_t` int8 values. FP32 path uses one `float` matrix per learner. | One shared `[seq or row][feature or col][learner]` buffer. Integer path uses `int8_t`; FP32 path uses `float`. |
| When interleaving happens | Only around selected grouped CodebookDense calls, if supported. | Once at the beginning of the block, then maintained through attention, projection, AddNorm, FFN, and final output. |
| Q/K/V projection          | May use `CodebookDense::computeInterleaved2LearnersSameSeq` or `computeInterleaved4Learners`, then returns to per-learner packed buffers. | Uses `CodebookDense::computeInterleaved2LearnersToInt8` or `computeInterleaved4LearnersToInt8` in int8, or `FloatCodebookDense::computeInterleaved` in FP32. |
| Attention QK matmul       | Per learner: `smmComputeRWMA`, `simdComputeRWMA`, or BWMA variants. | Integer: `matmulInterleaved2LearnersToInt8` or `matmulInterleaved4LearnersToInt8`. FP32: `matmulInterleavedTransposedRhs`. |
| Softmax                   | Integer: `Softmax::compute` or `computeRearranged`. FP32: `FloatSoftmax::compute`. | Integer: `Softmax::computeInterleaved2Learners` or `computeInterleaved4Learners`. FP32: `FloatSoftmax::computeInterleaved`. |
| Attention output matmul   | Per learner: `smmComputeRWMA`, `simdComputeRWMA`, or BWMA variants. | Integer: `matmulInterleaved2LearnersToInt8` or `matmulInterleaved4LearnersToInt8`. FP32: `matmulInterleavedRows`. |
| AddNorm                   | Integer: `AddNormalize::compute` or `computeRearranged`. FP32: `FloatAddNormalize::compute`. | Integer: `AddNormalize::computeInterleaved2Learners` or `computeInterleaved4Learners`. FP32: `FloatAddNormalize::computeInterleaved`. |
| Projection and FFN        | May use grouped CodebookDense, but stores outputs per learner between stages. | Projection and FFN consume and produce interleaved buffers.  |
| Final conversion          | Not needed; outputs are already per learner.                 | Integer calls `packInterleavedLearners2/4`. FP32 calls `deinterleaveLearnerMatrices`. |

#### 3.4.2 Functions Used Only by Integer Fully Interleaved Mode

These functions are part of the full int8 interleaved pipeline and are not used
by the normal grouped non-full path:

```text
TransformerBlock::computeGroup2FullInterleaved
TransformerBlock::computeGroup4FullInterleaved
SingleHeadSelfAttn::computeInterleaved2Learners
SingleHeadSelfAttn::computeInterleaved4Learners
computeCodebookDenseInterleaved2Learners
computeCodebookDenseInterleaved4Learners
CodebookDense::computeInterleaved2LearnersToInt8
CodebookDense::computeInterleaved4LearnersToInt8
interleavePackedLearners2
interleavePackedLearners4
packInterleavedLearners2
packInterleavedLearners4
transposeInterleavedRowsToCols2
transposeInterleavedRowsToCols4
matmulInterleaved2LearnersToInt8
matmulInterleaved4LearnersToInt8
copyHeadToMultiheadInterleaved2Learners
copyHeadToMultiheadInterleaved4Learners
Softmax::computeInterleaved2Learners
Softmax::computeInterleaved4Learners
Softmax::post_softmax_interleaved2D
Softmax::post_softmax_interleaved4D
AddNormalize::computeInterleaved2Learners
AddNormalize::computeInterleaved4Learners
```

#### 3.4.3 Functions Used by Integer Non-Full Interleaved Mode

These functions are used when grouped learners are enabled but the full
interleaved pipeline is off:

```text
TransformerBlock::computeGroup2
TransformerBlock::computeGroup4
TransformerBlock::computeGroupImpl<2>
TransformerBlock::computeGroupImpl<4>
SingleHeadSelfAttn::computeGroup2
SingleHeadSelfAttn::computeGroup4
SingleHeadSelfAttn::computeGroupImpl<2>
SingleHeadSelfAttn::computeGroupImpl<4>
tryComputeGroupedCodebookDense2
tryComputeGroupedCodebookDense4
CodebookDense::computeInterleaved2LearnersSameSeq
CodebookDense::computeInterleaved4Learners
gemm_exec_compact_int_interleaved_2Learners_same_seq
gemm_exec_compact_int_interleaved_4Learners_same_seq
gemm_exec_compact_int_sve_interleaved_2Learners_same_seq
gemm_exec_compact_int_sve_interleaved_4Learners_same_seq
Softmax::compute
AddNormalize::compute
```

The grouped non-full path may use interleaved GEMM kernels, but it does not use interleaved softmax, interleaved AddNorm, or interleaved attention matmul.



---

## 4. GEMM and SVE Code

The core GEMM implementation is located under:

```text
Full_NN/inc/
Full_NN/src/
```

Main file structure:

```text
Full_NN/
├── inc/
│   ├── gemm_exec.h
│   ├── gemm_SVE.h
│   ├── dense_exec.h
│   ├── dense_SVE.h
│   ├── conv_exec.h
│   ├── conv_def.h
│   └── SVE_implementations.h
└── src/
    ├── gemm_exec.c
    ├── gemm_SVE.c
    ├── dense_exec.c
    ├── dense_SVE.c
    ├── dense_SVE_f16.c
    ├── conv_exec.c
    ├── conv_SVE.c
    └── conv_def.c
```

Important files:

| File                      | Role                                                         |
| ------------------------- | ------------------------------------------------------------ |
| `Full_NN/inc/gemm_exec.h` | Declares GEMM wrapper functions and the shared `gemm_t` descriptor. |
| `Full_NN/src/gemm_exec.c` | Implements scalar compact GEMM kernels and SVE wrapper functions. |
| `Full_NN/inc/gemm_SVE.h`  | Declares low-level SVE GEMM row kernels.                     |
| `Full_NN/src/gemm_SVE.c`  | Implements low-level SVE row kernels for compact GEMM and interleaved attention matmul. |

The GEMM code is split into two levels:

| Level           | Example functions                                        | Role                                                         |
| --------------- | -------------------------------------------------------- | ------------------------------------------------------------ |
| GEMM wrappers   | `gemm_exec_compact_int_sve_interleaved_2Learners_same_seq(...)` | Check fallback conditions, prepare temporary buffers, select tiles, and call row kernels. |
| SVE row kernels | `sve_gemm_row_compact_int8_interleaved_2Learners_same_seq(...)` | Perform the actual vectorized dot-product using SVE intrinsics. |



### 4.1 GEMM wrappers

#### **Scalar/non-SIMD wrapper functions:**

* These functions are used during the early stage development
* Verify the correctness of interleaved layout
* As fallback of defensive programming: 
  * If the codebook is too large, e.g. bits_per_cb > 8, current SIMD version can't execute it, then it will fallback to scalar version

| Function                                         | Role                                                         |
| ------------------------------------------------ | ------------------------------------------------------------ |
| `gemm_exec_noCB`                                 | FP32  1-learner dense GEMM without codebook compression.     |
| `gemm_exec_compact`                              | FP32 1-learner compact/codebook GEMM.                        |
| `gemm_exec_noCB_int`                             | Integer 1-learner dense GEMM without codebook compression.   |
| `gemm_exec_compact_int`                          | Integer 1-learner compact/codebook GEMM.                     |
| `gemm_exec_compact_fp32_interleaved_2Learners_same_seq` | Scalar FP32 2-learner interleaved compact GEMM for same-sequence weights. |
| `gemm_exec_compact_fp32_interleaved_4Learners_same_seq` | Scalar FP32 4-learner interleaved compact GEMM for same-sequence weights. |
| `gemm_exec_compact_fp32_interleaved_4Learners_diff_seq` | Scalar FP32 4-learner interleaved compact GEMM for different per-learner index streams. (Developed but not used in this project) |
| `gemm_exec_compact_int_interleaved_2Learners_same_seq`  | Scalar int8 2-learner interleaved compact GEMM for same-sequence weights. |
| `gemm_exec_compact_int_interleaved_4Learners_same_seq`  | Scalar int8 4-learner interleaved compact GEMM for same-sequence weights. |
| `gemm_exec_compact_int_interleaved_4Learners_diff_seq`  | Scalar int8 4-learner interleaved compact GEMM for different per-learner index streams. (Developed but not used in this project) |

#### **SIMD wrapper functions:**

| Function                                             | Role                                                         |
| ---------------------------------------------------- | ------------------------------------------------------------ |
| `gemm_exec_compact_sve`                              | SVE FP32 1-learner compact GEMM wrapper.                     |
| `gemm_exec_compact_int_sve`                          | SVE int8 1-learner compact GEMM wrapper.                     |
| `gemm_exec_compact_sve_fp32_interleaved_2Learners_same_seq` | SVE FP32 2-learner interleaved compact GEMM wrapper for same-sequence weights. |
| `gemm_exec_compact_sve_fp32_interleaved_4Learners_same_seq` | SVE FP32 4-learner interleaved compact GEMM wrapper for same-sequence weights. |
| `gemm_exec_compact_sve_fp32_interleaved_4Learners_diff_seq` | SVE FP32 4-learner interleaved compact GEMM wrapper for different per-learner index streams (Developed but not used in this project). |
| `gemm_exec_compact_int_sve_interleaved_2Learners_same_seq`  | SVE int8 2-learner interleaved compact GEMM wrapper for same-sequence weights. |
| `gemm_exec_compact_int_sve_interleaved_4Learners_same_seq`  | SVE int8 4-learner interleaved compact GEMM wrapper for same-sequence weights. |
| `gemm_exec_compact_int_sve_interleaved_4Learners_diff_seq`  | SVE int8 4-learner interleaved compact GEMM wrapper for different per-learner index streams (Developed but not used in this project). |



##### Pseudocode: `gemm_exec_compact_int_sve_interleaved_2Learners_same_seq`

This function is the SVE wrapper for int8 compact GEMM with two same-sequence interleaved learners. It first handles unsupported or degenerate cases, then prepares int32 temporary buffers, and finally dispatches the computation to the low-level SVE row kernel.

In the experiments, tiling is effectively disabled by setting `TILE_SIZE` to 1.  Therefore, the current implementation can be treated as operating on the full matrix.  Completing and evaluating the tiled version is left as future work.

```text
function gemm_exec_compact_int_sve_interleaved_2Learners_same_seq(...):

    if seq_len == 0 or output_size == 0:
        return

    if bits_per_cb == 0:
        fill output with bias or zero
        return

    if input_size == 0 or n_words_row == 0:
        fill output with bias or zero
        return

    if bits_per_cb > 8:
        call scalar 2D same-sequence fallback
        return

    codebook_size = 1 << bits_per_cb

    if codebook does not fit in SVE registers:
        call scalar 2D same-sequence fallback
        return

    expand interleaved int8 codebook to int32:
        codebook_i32_interleaved[codebook_index][learner]

    allocate int32 temporary input buffer

    if allocation fails:
        call scalar 2D same-sequence fallback
        return

    sign-extend interleaved int8 input to int32

    choose sequence tile size
    choose packed-K-word tile size

    for each output column:
        select shared packed index row
        select two interleaved bias values

        for each sequence tile:
            processed_k = 0

            for each packed K-word tile:
                compute valid K tile size

                call sve_gemm_row_compact_int8_interleaved_2Learners_same_seq(
                    packed index tile,
                    int32 interleaved input tile,
                    int32 interleaved codebook,
                    interleaved output tile,
                    bias values,
                    add-bias flag,
                    accumulate flag
                )

                processed_k += K tile size

    free temporary input buffer

```

---

### 4.2 SVE row kernels

`Full_NN/src/gemm_SVE.c` contains the low-level SVE row kernels and dense
interleaved attention kernels. Important functions:

#### **SVE row-kernel functions (input-matrix multiplication):**

| Function                                            | Role                                                         |
| --------------------------------------------------- | ------------------------------------------------------------ |
| `sve_gemm_row_compact_int8`                         | SVE row kernel for single-learner int8 compact GEMM.         |
| `sve_gemm_row_compact_fp32`                         | SVE row kernel for single-learner FP32 compact GEMM.         |
| `sve_gemm_row_compact_fp32_interleaved_2Learners_same_seq` | SVE row kernel for FP32 2D same-sequence interleaving.       |
| `sve_gemm_row_compact_fp32_interleaved_4Learners_same_seq` | SVE row kernel for FP32 4D same-sequence interleaving.       |
| `sve_gemm_row_compact_fp32_interleaved_4Learners_diff_seq` | SVE row kernel for FP32 4D different-sequence interleaving. (Developed but not used in this project) |
| `sve_gemm_row_compact_int8_interleaved_2Learners_same_seq` | SVE row kernel for int8 2D same-sequence interleaving.       |
| `sve_gemm_row_compact_int8_interleaved_4Learners_same_seq` | SVE row kernel for int8 4D same-sequence interleaving.       |
| `sve_gemm_row_compact_int8_interleaved_4Learners_diff_seq` | SVE row kernel for int8 4D different-sequence interleaving. (Developed but not used in this project) |
| `sve_gemm_dense_int8_interleaved_2Learners`                | SVE dense int8 attention matmul for 2 interleaved learners.  |
| `sve_gemm_dense_int8_interleaved_4Learners`                | SVE dense int8 attention matmul for 4 interleaved learners.  |


#### Row-kernel execution idea

The SVE row kernels compute one output-feature row of the weight matrix against
multiple sequence rows of the input activation matrix. For compact GEMM, the
weight row is not stored as explicit weights. Instead, it is stored as packed
codebook indices.

For example, in the 2-learner same-sequence interleaved int8 kernel
`sve_gemm_row_compact_int8_interleaved_2Learners_same_seq`, the packed index row is
shared by both learners. The kernel repeatedly unpacks indices, looks up the
corresponding codebook values for learner 0 and learner 1, multiplies them with
the interleaved input activations, accumulates partial sums, and writes two
interleaved output values.

So, the figure below also shows this: a row-kernel produce one interleaved output column (two red columns at the output).

##### Pseudocode: `sve_gemm_row_compact_int8_interleaved_2Learners_same_seq`

```text
for each sequence row in the sequence tile:

    acc_learner0 = 0
    acc_learner1 = 0

    for each packed index word in the current K tile:

        load packed index word

        for each index stored inside the word:

            unpack shared codebook index

            weight_learner0 = codebook[codebook_index][learner0]
            weight_learner1 = codebook[codebook_index][learner1]

            input_learner0 = input[sequence row][input feature][learner0]
            input_learner1 = input[sequence row][input feature][learner1]

            acc_learner0 += input_learner0 * weight_learner0
            acc_learner1 += input_learner1 * weight_learner1

    if this is the first K tile:
        add bias to both learners
    else:
        accumulate into the existing output values

    output[sequence row][output feature][learner0] = acc_learner0
    output[sequence row][output feature][learner1] = acc_learner1
```

<p align="center">
  <img src="./README_figures/simd_nested_loop_2D_refined_names_interleaved_inputs_v2.png" width="650">
</p>



<p align="center">
  <b>Figure:</b> Row-kernel loop structure for compact interleaved GEMM.
  For a fixed output-feature index, the kernel reuses one packed weight-index
  row and iterates over sequence rows and input features to generate the
  corresponding output values.
</p>




### 4.3 SIMD GEMM for Runtime Attention Matrices

In the fully interleaved integer mode, the runtime attention matrix multiplications are also executed on the interleaved multi-learner layout.
This includes the two dense attention GEMMs:

```text
Q * K^T
softmax(Q * K^T) * V
```

These operations are different from codebooked GEMM layers. The matrices `Q`, `K`, `V`, and the softmax output are runtime activation matrices, so there are no packed codebook indices or codebook lookups. Instead, the implementation performs dense int8 matrix multiplication directly on interleaved learner data.

<p align="center">
  <img src="./README_figures/Q_K_GEMM.png" width="650">
</p>


<p align="center">
  <b>Figure:</b> Interleaved runtime attention GEMM. In the fully interleaved
  pipeline, the attention score matrix is computed as <code>Q * K^T</code> on
  interleaved learner data.
</p>

| Function                                  | File                                        | Role                                                         |
| ----------------------------------------- | ------------------------------------------- | ------------------------------------------------------------ |
| `matmulInterleaved2LearnersToInt8(...)`          | `transformer_layers/interleavedPipeline.cc` | Interleaved int8 runtime attention matmul for 2 learners. Used for both `Q * K^T` and `softmax * V` in the 2-learner fully interleaved path. |
| `matmulInterleaved4LearnersToInt8(...)`          | `transformer_layers/interleavedPipeline.cc` | Interleaved int8 runtime attention matmul for 4 learners. Used for both `Q * K^T` and `softmax * V` in the 4-learner fully interleaved path. |
| `sve_gemm_dense_int8_interleaved_2Learners(...)` | `Full_NN/src/gemm_SVE.c`                    | Low-level SVE dense int8 kernel called by `matmulInterleaved2LearnersToInt8(...)` when SIMD is enabled. |
| `sve_gemm_dense_int8_interleaved_4Learners(...)` | `Full_NN/src/gemm_SVE.c`                    | Low-level SVE dense int8 kernel called by `matmulInterleaved4LearnersToInt8(...)` when SIMD is enabled. |

The interleaved attention layout is:

```text
[sequence or row][feature or column][learner]
```

Therefore, for each matrix element, the values of different learners are stored
next to each other in memory. This allows the SVE dense int8 kernels to process
multiple learners together during the attention GEMM.

In the fully interleaved path, these functions are used inside:

```text
SingleHeadSelfAttn::computeInterleaved2Learners(...)
SingleHeadSelfAttn::computeInterleaved4Learners(...)
```

The simplified call flow is:

```text
SingleHeadSelfAttn::computeInterleaved2Learners/4D(...)
  -> compute interleaved Q, K, V using CodebookDense
  -> matmulInterleaved2LearnersToInt8(...) or matmulInterleaved4LearnersToInt8(...)
       -> Q * K^T
       -> sve_gemm_dense_int8_interleaved_2Learners/4D(...) when SIMD
  -> Softmax::computeInterleaved2Learners/4D(...)
  -> matmulInterleaved2LearnersToInt8(...) or matmulInterleaved4LearnersToInt8(...)
       -> softmax * V
       -> sve_gemm_dense_int8_interleaved_2Learners/4D(...) when SIMD
```



---

## 5. Floating-Point Transformer Code

The FP32 Transformer implementation is also located under:

```text
transformer_layers/
```

Main file structure:

```text
transformer_layers/
├── transformerFloat.cc/.h
├── floatTransformerBlock.cc/.h
├── floatSelfAttention.cc/.h
├── floatCodebookDense.cc/.h
├── floatAddNorm.cc/.h
├── floatSoftmax.cc/.h
├── floatDump.cc/.h
└── floatCommon.h
```

Important files:

| File                          | Role                                                         |
| ----------------------------- | ------------------------------------------------------------ |
| `transformerFloat.cc/.h`      | Top-level FP32 Transformer entry point.                      |
| `floatTransformerBlock.cc/.h` | Implements FP32 Transformer block execution.                 |
| `floatSelfAttention.cc/.h`    | Implements FP32 self-attention.                              |
| `floatCodebookDense.cc/.h`    | Implements FP32 codebook-backed dense layers and dispatches FP32 compact GEMM kernels. |
| `floatAddNorm.cc/.h`          | Implements FP32 residual addition and layer normalization.   |
| `floatSoftmax.cc/.h`          | Implements FP32 softmax.                                     |
| `floatDump.cc/.h`             | Provides FP32 tensor dumping and interleave/deinterleave helpers. |
| `floatCommon.h`               | Defines common FP32 matrix types and utilities.              |

### 5.1 FP32 codebook Transformer with SVE

Typical flags:

```text
USE_FP32_TRANSFORMER_FLAG=1
USE_CODEBOOK_GEMM_FLAG=1
SIMD_FLAG=1
```

When `USE_FP32_TRANSFORMER_FLAG=1`, `transformer.cpp` exits the integer path
early and dispatches to the FP32 Transformer implementation:

```cpp
#if CFG_USE_FP32_TRANSFORMER
    TransformerFloat::run(
        learner_count,
        dump_outputs_enabled ? dir_name + "/multiple_learner_outputs/c" : std::string());
    return;
#endif
```

The call flow becomes:

```text
main()
  -> test()
      -> TransformerFloat::run()
```



---

#### 5.1.1 Single-learner FP32 codebook path

For `N_LEARNERS=1`:

```text
TransformerFloat::run()
  -> FloatTransformerBlock::compute()
```

Inside the FP32 Transformer block:

```text
for each attention head:
    FloatSingleHeadSelfAttn::compute()
        -> FloatCodebookDense::compute(q_hN)
        -> FloatCodebookDense::compute(k_hN)
        -> FloatCodebookDense::compute(v_hN)
        -> Q * K^T
        -> FloatSoftmax::compute()
        -> softmax * V

copy heads to multihead buffer

FloatCodebookDense::compute(condense)
FloatAddNormalize::compute(input, condense_out)

FloatCodebookDense::compute(ff0)
FloatCodebookDense::compute(ff1)

FloatAddNormalize::compute(after_attn_addnorm, ff1_out)
```

FP32 codebook dense layers call the FP32 compact GEMM path:

```text
FloatCodebookDense::compute()
  -> gemm_exec_compact_sve()
```

when SVE is enabled.



**<u>The complete function call map is:</u>**

```text
FloatTransformerBlock::compute(...)
  for each head:
      FloatSingleHeadSelfAttn::compute(...)
          -> FloatCodebookDense::compute(...) for Q
              -> gemm_exec_compact(...)
              -> or gemm_exec_compact_sve(...) when SIMD
          -> FloatCodebookDense::compute(...) for K
          -> FloatCodebookDense::compute(...) for V
          -> matmulTransposedRhs(...)       // Q * K^T
          -> FloatSoftmax::compute(...)
          -> matmulRows(...)               // softmax * V
      -> copyHeadToMultihead(...)

  -> FloatCodebookDense::compute(...)       // condense
  -> FloatAddNormalize::compute(...)
  -> FloatCodebookDense::compute(...)       // ff0
  -> FloatCodebookDense::compute(...)       // ff1
  -> FloatAddNormalize::compute(...)
```



---

#### 5.1.2 Multi-learner FP32 fully interleaved path

For `N_LEARNERS=2` or `N_LEARNERS=4` with:

```text
FULL_INTERLEAVED_PIPELINE_FLAG=1
```

the FP32 path uses fully interleaved learner execution:

```text
FloatTransformerBlock::computeGroup2()
  -> FloatTransformerBlock::computeFullInterleavedBlock<2>()

FloatTransformerBlock::computeGroup4()
  -> FloatTransformerBlock::computeFullInterleavedBlock<4>()
```

The FP32 interleaved layout is:

```text
[row][column][learner]
```

or equivalently:

```text
[matrix_element][learner]
```

The data flow is:

```text
separate learner FP32 inputs
  -> interleaveLearnerMatrices()

for each attention head:
    FloatSingleHeadSelfAttn::computeInterleaved2Learners/4D()
        -> interleaved FloatCodebookDense Q
        -> interleaved FloatCodebookDense K
        -> interleaved FloatCodebookDense V
        -> interleaved Q * K^T
        -> FloatSoftmax::computeInterleaved()
        -> interleaved softmax * V
        -> copyHeadToMultiheadInterleaved()

interleaved FloatCodebookDense condense
FloatAddNormalize::computeInterleaved()

interleaved FloatCodebookDense ff0
interleaved FloatCodebookDense ff1

FloatAddNormalize::computeInterleaved()

deinterleaveLearnerMatrices()
  -> per-learner FP32 outputs
```

The FP32 interleaved codebook GEMM calls are:

```text
FloatCodebookDense::computeInterleaved()
  -> gemm_exec_compact_sve_fp32_interleaved_2Learners_same_seq()

FloatCodebookDense::computeInterleaved()
  -> gemm_exec_compact_sve_fp32_interleaved_4Learners_same_seq()

FloatCodebookDense::computeInterleaved()
  -> gemm_exec_compact_sve_fp32_interleaved_4Learners_diff_seq()
```



**<u>The complete function call map is:</u>**

```text
FloatTransformerBlock::computeGroup2/4(...)
  -> FloatTransformerBlock::computeFullInterleavedBlock<2/4>(...)
      -> interleaveLearnerMatrices(...)

      for each head:
          FloatSingleHeadSelfAttn::computeInterleaved2Learners/4D(...)
              -> FloatCodebookDense::computeInterleaved(...) for Q
                  -> gemm_exec_compact_fp32_interleaved_2Learners_same_seq(...)
                  -> or gemm_exec_compact_fp32_interleaved_4Learners_same_seq(...)
                  -> or gemm_exec_compact_fp32_interleaved_4Learners_diff_seq(...)
                  -> SVE versions when SIMD
              -> FloatCodebookDense::computeInterleaved(...) for K
              -> FloatCodebookDense::computeInterleaved(...) for V
              -> matmulInterleavedTransposedRhs(...)  // Q * K^T
              -> FloatSoftmax::computeInterleaved(...)
              -> matmulInterleavedRows(...)           // softmax * V
          -> copyHeadToMultiheadInterleaved(...)

      -> FloatCodebookDense::computeInterleaved(...)   // condense
      -> FloatAddNormalize::computeInterleaved(...)
      -> FloatCodebookDense::computeInterleaved(...)   // ff0
      -> FloatCodebookDense::computeInterleaved(...)   // ff1
      -> FloatAddNormalize::computeInterleaved(...)

      -> deinterleaveLearnerMatrices(...)
```



Important characteristics:

```text
Uses FP32 tensors
Uses generated codebook registry
Uses FP32 compact GEMM
Can use SVE FP32 compact GEMM kernels
Can run single learner or grouped interleaved learners
Bypasses the integer Transformer path
```



### 5.2 Summary of used functions

Functions Used by FP32 Fully Interleaved Mode: 

```text
TransformerFloat::run
FloatTransformerBlock::computeGroup2
FloatTransformerBlock::computeGroup4
FloatTransformerBlock::computeFullInterleavedBlock<2>
FloatTransformerBlock::computeFullInterleavedBlock<4>
FloatSingleHeadSelfAttn::computeInterleaved2Learners
FloatSingleHeadSelfAttn::computeInterleaved4Learners
FloatCodebookDense::computeInterleaved
gemm_exec_compact_fp32_interleaved_2Learners_same_seq
gemm_exec_compact_fp32_interleaved_4Learners_same_seq
gemm_exec_compact_sve_fp32_interleaved_2Learners_same_seq
gemm_exec_compact_sve_fp32_interleaved_4Learners_same_seq
FloatSoftmax::computeInterleaved
FloatAddNormalize::computeInterleaved
interleaveLearnerMatrices
deinterleaveLearnerMatrices
```



---

## 6. Shared Helper Files

Several helper files are shared by the integer and FP32 paths:

```text
transformer_layers/
├── run_mode_config.h
├── registry_adapter.h
├── profile.cc/.h
├── debuggerFunctions.cc/.h
├── transformerBlockInterleavedHelpers.cc/.h
└── floatDump.cc/.h
```

Important files:

| File                                       | Role                                                         |
| ------------------------------------------ | ------------------------------------------------------------ |
| `run_mode_config.h`                        | Normalizes compile-time macros into `CFG_*` macros and performs sanity checks. |
| `registry_adapter.h`                       | Converts generated registry entries into integer `CodebookDense` configurations. |
| `profile.cc/.h`                            | Provides gem5 profiling helpers, such as reset/dump stats calls. |
| `debuggerFunctions.cc/.h`                  | Provides integer tensor dumping and comparison utilities.    |
| `transformerBlockInterleavedHelpers.cc/.h` | Provides helper functions for fully interleaved integer Transformer execution. |
| `floatDump.cc/.h`                          | Provides FP32 dump and interleaving utilities.               |

---

## 7. High-Level Code Organization Summary

```text
TiC-SAT/
├── transformer.cpp
├── transformer.h
├── compile_transformer.sh
│
├── transformer_layers/
│   ├── Integer Transformer path
│   │   ├── transformerBlock.cc/.h
│   │   ├── selfattention.cc/.h
│   │   ├── codebookDense.cc/.h
│   │   ├── dense.cc/.h
│   │   ├── addNorm.cc/.h
│   │   ├── softmax.cc/.h
│   │   └── interleavedPipeline.cc/.h
│   │
│   ├── FP32 Transformer path
│   │   ├── transformerFloat.cc/.h
│   │   ├── floatTransformerBlock.cc/.h
│   │   ├── floatSelfAttention.cc/.h
│   │   ├── floatCodebookDense.cc/.h
│   │   ├── floatAddNorm.cc/.h
│   │   └── floatSoftmax.cc/.h
│   │
│   └── Shared helpers
│       ├── run_mode_config.h
│       ├── registry_adapter.h
│       ├── profile.cc/.h
│       ├── debuggerFunctions.cc/.h
│       └── transformerBlockInterleavedHelpers.cc/.h
│
└── Full_NN/
    ├── generators/
    │   ├── Transformer_generator.ipynb
    │   ├── gemm_layer_generator.py
    │   ├── codebooks_defs_generator.py
    │   └── transformer_debug_utils.py
    │
    ├── gemm_definitions/
    │   ├── codebooks_def.h
    │   ├── generated_codebook_registry.h
    │   ├── input_matrix.h
    │   └── gemm_header_*.h
    │
    ├── inc/
    │   ├── gemm_exec.h
    │   └── gemm_SVE.h
    │
    └── src/
        ├── gemm_exec.c
        └── gemm_SVE.c
```

