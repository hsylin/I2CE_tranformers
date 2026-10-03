"""
Codebook definition and registry header generator.

This module is used by the Transformer/GEMM generator notebooks to emit C/C++
headers consumed by the generated TiC-SAT kernels.

It produces two related outputs:

1. A codebooks_def.h-style configuration header from
   templates/codebooks_defs_template.tpl. This header defines global compile
   time constants such as the number of learners, SVE vector layout, codebook
   size, index bit width, and feature flags.
2. A generated_codebook_registry.h-style registry header. This header contains
   one compact C++ view per generated layer, including packed weight indexes,
   int8/fp32 codebooks, optional biases, and helper functions for looking up a
   layer by name.

The generated C/C++ code uses codebooks to represent weights indirectly:
packed integer indexes select entries from small codebook arrays instead of
storing every weight value directly. This reduces memory footprint and lets the
kernels reuse vectorized lookup/decode logic.
"""

from ast import If
import math
import dense_layer_generator
from string import Template


import re
import numpy as np
from pathlib import Path




def generate_cb_definitions(filename, n_learners, codebook_size, SVE_lanes, use_bias, use_f16, same_seq, use_codebooks, tile_l1_size=None, tile_l2_size=None):
    """
    Generate the common codebook configuration header.

    Args:
        filename: Output header path, normally something like codebooks_def.h.
        n_learners: Number of ensemble learners generated for the model.
        codebook_size: Number of entries in each codebook.
        SVE_lanes: Number of 32-bit lanes available in one SVE vector.
        use_bias: Whether generated layers include bias arrays.
        use_f16: Whether kernels operate on fp16 values instead of fp32 values.
        same_seq: Whether all learners share the same packed weight-index
            sequence. If False, each learner may have its own sequence.
        use_codebooks: Whether layers use codebook lookup. If False, emit
            NO_CODEBOOKS so kernels can use the direct-weight path.
        tile_l1_size: Optional L1 tiling size used by generated kernels.
        tile_l2_size: Optional L2 tiling size used by generated kernels.

    The generated macros are included by generated layer headers and by kernel
    code that needs to know how many indexes fit in a word/register.
    """


    # Treat omitted tile sizes as "disabled" so the generated header always
    # contains concrete integer macros.
    if tile_l1_size is None:
        tile_l1_size = 0
    if tile_l2_size is None:
        tile_l2_size = 0

    # Same width the packing uses, so BITS_PER_CB always matches the stream
    # that dense_layer_generator.py actually emitted.
    idxs_bits = dense_layer_generator.index_bits(codebook_size)
    # for example, if codebook_size=16, we need 4 bits to represent 16 unique values (0 to 15). If codebook_size=256, we need 8 bits to represent 256 unique values (0 to 255).
    
    # Create a binary mask with idxs_bits number of 1's. This will be used to mask out the relevant bits when extracting codeword indices from packed bit representations.
    idx_mask = "0b{}".format("1" * idxs_bits)
    # If idxs_bits=4 → idx_mask = "0b1111"
    # If idxs_bits=5 → idx_mask = "0b11111"
    
    n_regs_cb = math.ceil(codebook_size / SVE_lanes)  # How many SVE registers to hold a full codebook
    n_regs_per_cb_string = "#define N_SVE_REG_CB_{}\t{}\n".format(n_regs_cb, n_regs_cb)

    n_regs_cb_f16 = math.ceil(codebook_size / (SVE_lanes * 2))  # How many SVE registers to hold a full codebook (using fp16 quantization)
    n_regs_per_cb_f16_string = "#define N_SVE_REG_CB_F16_{}\t{}\n".format(n_regs_cb_f16, n_regs_cb_f16)
    
    n_regs_per_cb_string += n_regs_per_cb_f16_string

    # Packed index words are uint32_t in the generated kernels. IDXS_PER_WORD
    # in the template is derived from this width and BITS_PER_CB.
    idxs_bitlen = 32

    # SVE_lanes counts 32-bit lanes. fp16 uses twice as many halfword lanes,
    # and the byte count is derived from those halfword lanes.
    n_lanes = SVE_lanes
    n_halflanes = SVE_lanes * 2
    sve_bytes = n_halflanes * 2

    # Optional feature macros are emitted only when the corresponding path is
    # active. Leaving the string empty keeps the template valid without adding
    # unused defines.
    if use_bias:
        use_bias_def_string = "#define USE_BIAS"
    else:
        use_bias_def_string = ""

    # The generated header advertises the arithmetic bit width used by kernels.
    # Index packing still uses 32-bit words regardless of fp16/fp32 arithmetic.
    if not use_f16:
        word_bitlen = 32
        # n_lanes = SVE_lanes
    else:
        word_bitlen = 16
        # n_lanes = int(SVE_lanes * 2)

    # SAME_SEQ means every learner reuses the same packed weight-index stream.
    # DIFF_SEQ means each learner owns a separate packed stream.
    if same_seq:
        seq_type = "SAME_SEQ"
    else:
        seq_type = "DIFF_SEQ"

    # Some generated C/C++ code has specialized branches for exactly eight
    # learners, so expose that case as a compile-time macro.
    if n_learners == 8:
        learners_8_def = "#define LEARNERS_8\t1"
    else:
        learners_8_def = ""

    if not use_codebooks:
        no_cb_def = "#define NO_CODEBOOKS\t1"
    else:
        no_cb_def = ""
        

    # Load the C header template, substitute all Python-computed constants, and
    # write the final header. The template path is relative to the generator
    # working directory used by the notebooks.
    with open("./templates/codebooks_defs_template.tpl") as f:
        cont = f.read()
        tpl = Template(cont)

        header = tpl.substitute(
            n_learners = n_learners,
            sve_size = n_lanes,
            sve_halfwords = n_halflanes,
            sve_bytes = sve_bytes,
            codebook_size = codebook_size,
            tile_l1_size = tile_l1_size,
            tile_l2_size = tile_l2_size,
            bits_per_codeword = idxs_bits,
            index_bin_mask = idx_mask,
            word_bitlen = idxs_bitlen,
            string_n_regs_per_cb = n_regs_per_cb_string,
            bitwidth_used = word_bitlen,
            use_bias_def = use_bias_def_string,
            sequence_type = seq_type,
            learners_8_def = learners_8_def,
            no_cb_def = no_cb_def
        )
    
    with open(filename, "w") as f:
        f.write(header)





""" Transformer Codebook Registry Generator """

def sanitize_c_identifier(name: str) -> str:
    """
    Convert a generated layer name to a valid C identifier suffix.

    Layer names can contain punctuation such as "/" or "-". C/C++ variable
    names cannot, so this helper replaces non-identifier characters with "_"
    and prefixes names that start with a digit.
    """
    s = re.sub(r'[^0-9a-zA-Z_]', '_', name)
    if re.match(r'^[0-9]', s):
        s = "_" + s
    return s


def format_uint32_array(arr, var_name):
    """
    Format a 1D uint32_t C array.

    Used for packed weight-index words. The trailing "u" keeps literals
    unsigned in the generated C++ header.
    """
    arr = np.asarray(arr, dtype=np.uint32).reshape(-1)
    body = ", ".join(f"{int(x)}u" for x in arr)
    return f"static const uint32_t {var_name}[] = {{ {body} }};\n"


def format_uint32_array_2d(arr, var_name, rows_expr, cols_expr):
    """
    Format a 2D uint32_t C array with symbolic row/column dimensions.

    The expressions are passed as strings so generated arrays can use constants
    like N_LEARNERS_layer or OUTPUT_SIZE_layer instead of hard-coded numbers.
    """
    arr = np.asarray(arr, dtype=np.uint32)
    if arr.ndim != 2:
        raise ValueError(f"{var_name} must be 2D, got shape {arr.shape}")

    lines = [f"static const uint32_t {var_name}[{rows_expr}][{cols_expr}] = {{\n"]
    for row in arr:
        body = ", ".join(f"{int(x)}u" for x in row.reshape(-1))
        lines.append(f"    {{ {body} }},\n")
    lines.append("};\n")
    return "".join(lines)



def format_int8_array(arr, var_name):
    """
    Format a 1D int8_t C array for one learner's quantized codebook.
    """
    arr = np.asarray(arr, dtype=np.int8).reshape(-1)
    body = ", ".join(str(int(x)) for x in arr)
    return f"static const int8_t {var_name}[] = {{ {body} }};\n"


def format_int8_array_2d(arr, var_name, rows_expr, cols_expr):
    """
    Format a 2D int8_t C array for all learners or interleaved codebooks.
    """
    arr = np.asarray(arr, dtype=np.int8)
    if arr.ndim != 2:
        raise ValueError(f"{var_name} must be 2D, got shape {arr.shape}")

    lines = [f"static const int8_t {var_name}[{rows_expr}][{cols_expr}] = {{\n"]
    for row in arr:
        body = ", ".join(str(int(x)) for x in row.reshape(-1))
        lines.append(f"    {{ {body} }},\n")
    lines.append("};\n")
    return "".join(lines)



def format_float_array(arr, var_name):
    """
    Format a 1D float C array.

    Used for fp32 codebooks and bias vectors. Values are written with a fixed
    precision and an "f" suffix so the generated literals are float, not double.
    """
    arr = np.asarray(arr, dtype=np.float32).reshape(-1)
    body = ", ".join(f"{float(x):.8f}f" for x in arr)
    return f"static const float {var_name}[] = {{ {body} }};\n"


def format_float_array_2d(arr, var_name, rows_expr, cols_expr):
    """
    Format a 2D float C array for per-learner fp32 codebooks or biases.
    """
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"{var_name} must be 2D, got shape {arr.shape}")

    lines = [f"static const float {var_name}[{rows_expr}][{cols_expr}] = {{\n"]
    for row in arr:
        body = ", ".join(f"{float(x):.8f}f" for x in row.reshape(-1))
        lines.append(f"    {{ {body} }},\n")
    lines.append("};\n")
    return "".join(lines)



def build_generated_registry_header(entries):
    """
    Build the full content of generated_codebook_registry.h.

    Args:
        entries: List of dictionaries produced by gemm_layer_generator. Each
            dictionary describes one generated layer and contains dimensions,
            packed weight indexes, codebook arrays, and optional bias arrays.

    Returns:
        A complete C++ header as a string.

    The header has three main sections:
    1. GeneratedCodebookLayerView, a pointer-based view over one layer's data.
    2. Static arrays for each layer's indexes, codebooks, and biases.
    3. A registry table plus inline lookup/accessor helpers used by debug or
       verification code to inspect generated layer data by name.
    """
    lines = []

    # Basic includes are enough because the header stores raw pointers and uses
    # std::strcmp for name lookup.
    lines.append("#pragma once\n")
    lines.append("#include <cstddef>\n")
    lines.append("#include <cstdint>\n")
    lines.append("#include <cstring>\n\n")
    lines.append("#define GENERATED_CODEBOOK_REGISTRY_HAS_FP32 1\n\n")

    # This struct deliberately stores flattened pointers instead of owning
    # containers. That keeps the generated header lightweight and easy to use
    # from kernels or small verification programs.
    lines.append("struct GeneratedCodebookLayerView {\n")
    lines.append("    const char* name;\n")
    lines.append("    std::size_t input_size;\n")
    lines.append("    std::size_t output_size;\n")
    lines.append("    std::size_t n_words_row;\n")
    lines.append("    uint8_t bits_per_cb;\n")
    lines.append("    std::size_t n_learners;\n")
    lines.append("    bool same_seq;\n")
    lines.append("    const uint32_t* weight_idx;\n")
    lines.append("    const uint32_t* weight_idx_by_learner;\n")
    lines.append("    const uint32_t* weight_idx_interleaved;\n")
    lines.append("    const int8_t* codebook_int8;\n")
    lines.append("    const int8_t* codebooks_int8;\n")
    lines.append("    const int8_t* codebook_int8_interleaved;\n")
    lines.append("    const float* codebook_fp32;\n")
    lines.append("    const float* codebooks_fp32;\n")
    lines.append("    const float* codebook_fp32_interleaved;\n")
    lines.append("    const float* bias;\n")
    lines.append("    const float* biases;\n")
    lines.append("    const float* bias_interleaved;\n")
    lines.append("};\n\n")

    for entry in entries:
        cname = sanitize_c_identifier(entry["name"])
        n_learners = int(entry.get("n_learners", 1))
        same_seq = bool(entry.get("same_seq", True))

        # Emit one named block per layer. The constexpr dimensions let later
        # array declarations stay readable while still compiling to constants.
        lines.append(f"// ===== {entry['name']} =====\n")
        lines.append(f"static constexpr std::size_t INPUT_SIZE_{cname} = {int(entry['input_size'])};\n")
        lines.append(f"static constexpr std::size_t OUTPUT_SIZE_{cname} = {int(entry['output_size'])};\n")
        lines.append(f"static constexpr std::size_t N_WORDS_ROW_{cname} = {int(entry['n_words_row'])};\n")
        lines.append(f"static constexpr uint8_t BITS_PER_CB_{cname} = {int(entry['bits_per_cb'])};\n")
        lines.append(f"static constexpr std::size_t N_LEARNERS_{cname} = {n_learners};\n")
        lines.append(f"static constexpr bool SAME_SEQ_{cname} = {'true' if same_seq else 'false'};\n")

        lines.append(format_uint32_array(entry["weight_idx"], f"weight_idx_{cname}"))
        lines.append(format_int8_array(entry["codebook_int8"], f"codebook_{cname}"))

        # Optional fields default to nullptr in the registry. If present, the
        # arrays are emitted and the registry stores a pointer to the first
        # element so accessors can treat them as flat contiguous memory.
        weight_idx_by_learner_ref = "nullptr"
        if entry.get("weight_idx_by_learner", None) is not None:
            lines.append(
                format_uint32_array_2d(
                    entry["weight_idx_by_learner"],
                    f"weight_idx_by_learner_{cname}",
                    f"N_LEARNERS_{cname}",
                    f"(OUTPUT_SIZE_{cname} * N_WORDS_ROW_{cname})",
                )
            )
            weight_idx_by_learner_ref = f"&weight_idx_by_learner_{cname}[0][0]"

        weight_idx_interleaved_ref = "nullptr"
        if entry.get("weight_idx_interleaved", None) is not None:
            # Interleaved layout is [word][learner]. It is useful for code that
            # processes all learners' packed indexes for the same position.
            lines.append(
                format_uint32_array_2d(
                    np.asarray(entry["weight_idx_interleaved"], dtype=np.uint32).reshape(-1, n_learners),
                    f"weight_idx_interleaved_{cname}",
                    f"(OUTPUT_SIZE_{cname} * N_WORDS_ROW_{cname})",
                    f"N_LEARNERS_{cname}",
                )
            )
            weight_idx_interleaved_ref = f"&weight_idx_interleaved_{cname}[0][0]"

        codebooks_int8_ref = "nullptr"
        if entry.get("codebooks_int8", None) is not None:
            lines.append(
                format_int8_array_2d(
                    entry["codebooks_int8"],
                    f"codebooks_{cname}",
                    f"N_LEARNERS_{cname}",
                    f"(1u << BITS_PER_CB_{cname})",
                )
            )
            codebooks_int8_ref = f"&codebooks_{cname}[0][0]"

        codebook_int8_interleaved_ref = "nullptr"
        if entry.get("codebook_int8_interleaved", None) is not None:
            # Interleaved codebook layout is [codeword][learner], placing the
            # same codeword from all learners next to each other.
            lines.append(
                format_int8_array_2d(
                    np.asarray(entry["codebook_int8_interleaved"], dtype=np.int8).reshape(-1, n_learners),
                    f"codebook_interleaved_{cname}",
                    f"(1u << BITS_PER_CB_{cname})",
                    f"N_LEARNERS_{cname}",
                )
            )
            codebook_int8_interleaved_ref = f"&codebook_interleaved_{cname}[0][0]"

        codebook_fp32_ref = "nullptr"
        if entry.get("codebook_fp32", None) is not None:
            lines.append(format_float_array(entry["codebook_fp32"], f"codebook_fp32_{cname}"))
            codebook_fp32_ref = f"codebook_fp32_{cname}"

        codebooks_fp32_ref = "nullptr"
        if entry.get("codebooks_fp32", None) is not None:
            lines.append(
                format_float_array_2d(
                    entry["codebooks_fp32"],
                    f"codebooks_fp32_{cname}",
                    f"N_LEARNERS_{cname}",
                    f"(1u << BITS_PER_CB_{cname})",
                )
            )
            codebooks_fp32_ref = f"&codebooks_fp32_{cname}[0][0]"

        codebook_fp32_interleaved_ref = "nullptr"
        if entry.get("codebook_fp32_interleaved", None) is not None:
            # Keep an fp32 interleaved copy too. It is mainly useful for
            # debugging/verification because generated kernels may use int8.
            lines.append(
                format_float_array_2d(
                    np.asarray(entry["codebook_fp32_interleaved"], dtype=np.float32).reshape(-1, n_learners),
                    f"codebook_fp32_interleaved_{cname}",
                    f"(1u << BITS_PER_CB_{cname})",
                    f"N_LEARNERS_{cname}",
                )
            )
            codebook_fp32_interleaved_ref = f"&codebook_fp32_interleaved_{cname}[0][0]"

        if entry.get("bias", None) is not None:
            lines.append(format_float_array(entry["bias"], f"bias_{cname}"))
            bias_ref = f"bias_{cname}"
        else:
            bias_ref = "nullptr"

        biases_ref = "nullptr"
        if entry.get("biases", None) is not None:
            lines.append(
                format_float_array_2d(
                    entry["biases"],
                    f"biases_{cname}",
                    f"N_LEARNERS_{cname}",
                    f"OUTPUT_SIZE_{cname}",
                )
            )
            biases_ref = f"&biases_{cname}[0][0]"

        bias_interleaved_ref = "nullptr"
        if entry.get("bias_interleaved", None) is not None:
            # Biases follow the same interleaving idea as codebooks:
            # [output_channel][learner].
            lines.append(
                format_float_array_2d(
                    np.asarray(entry["bias_interleaved"], dtype=np.float32).reshape(-1, n_learners),
                    f"bias_interleaved_{cname}",
                    f"OUTPUT_SIZE_{cname}",
                    f"N_LEARNERS_{cname}",
                )
            )
            bias_interleaved_ref = f"&bias_interleaved_{cname}[0][0]"

        lines.append("\n")
        # Store the exact pointer expressions needed by the registry row. This
        # avoids recomputing them when the final table is emitted below.
        entry["_refs"] = {
            "weight_idx_by_learner": weight_idx_by_learner_ref,
            "weight_idx_interleaved": weight_idx_interleaved_ref,
            "codebooks_int8": codebooks_int8_ref,
            "codebook_int8_interleaved": codebook_int8_interleaved_ref,
            "codebook_fp32": codebook_fp32_ref,
            "codebooks_fp32": codebooks_fp32_ref,
            "codebook_fp32_interleaved": codebook_fp32_interleaved_ref,
            "bias": bias_ref,
            "biases": biases_ref,
            "bias_interleaved": bias_interleaved_ref,
            "same_seq": same_seq,
            "n_learners": n_learners,
        }

    lines.append("// ===== registry =====\n")
    lines.append("static const GeneratedCodebookLayerView kGeneratedCodebookLayers[] = {\n")

    # Each row ties the generated arrays above to the stable layer name used by
    # callers. Optional arrays appear as nullptr when a layout was not emitted.
    for entry in entries:
        cname = sanitize_c_identifier(entry["name"])
        refs = entry["_refs"]
        lines.append(
            f'    {{ "{entry["name"]}", '
            f'INPUT_SIZE_{cname}, '
            f'OUTPUT_SIZE_{cname}, '
            f'N_WORDS_ROW_{cname}, '
            f'BITS_PER_CB_{cname}, '
            f'N_LEARNERS_{cname}, '
            f'SAME_SEQ_{cname}, '
            f'weight_idx_{cname}, '
            f'{refs["weight_idx_by_learner"]}, '
            f'{refs["weight_idx_interleaved"]}, '
            f'codebook_{cname}, '
            f'{refs["codebooks_int8"]}, '
            f'{refs["codebook_int8_interleaved"]}, '
            f'{refs["codebook_fp32"]}, '
            f'{refs["codebooks_fp32"]}, '
            f'{refs["codebook_fp32_interleaved"]}, '
            f'{refs["bias"]}, '
            f'{refs["biases"]}, '
            f'{refs["bias_interleaved"]} }},\n'
        )

    lines.append("};\n\n")

    lines.append("static constexpr std::size_t kGeneratedCodebookLayerCount =\n")
    lines.append("    sizeof(kGeneratedCodebookLayers) / sizeof(kGeneratedCodebookLayers[0]);\n\n")

    # Generated helper: linear search by the stable layer name saved in each
    # registry row. The registry is small, so a simple loop keeps the header
    # dependency-free and easy to compile anywhere.
    lines.append("inline const GeneratedCodebookLayerView* findGeneratedCodebookLayer(const char* name) {\n")
    lines.append("    for (std::size_t i = 0; i < kGeneratedCodebookLayerCount; ++i) {\n")
    lines.append("        if (std::strcmp(kGeneratedCodebookLayers[i].name, name) == 0) {\n")
    lines.append("            return &kGeneratedCodebookLayers[i];\n")
    lines.append("        }\n")
    lines.append("    }\n")
    lines.append("    return nullptr;\n")
    lines.append("}\n\n")

    # Generated helper: codebook size is 2 ** bits_per_cb. This matches the
    # packing logic used when weights are converted to codeword indexes.
    lines.append("inline std::size_t getGeneratedCodebookSize(const GeneratedCodebookLayerView& view) {\n")
    lines.append("    return static_cast<std::size_t>(1u) << view.bits_per_cb;\n")
    lines.append("}\n\n")

    # Generated helper: one learner's packed index stream stores
    # output_size * n_words_row uint32_t words.
    lines.append("inline std::size_t getGeneratedWeightIdxCount(const GeneratedCodebookLayerView& view) {\n")
    lines.append("    return view.output_size * view.n_words_row;\n")
    lines.append("}\n\n")

    # Generated helper: return the correct packed index stream for a learner.
    # If all learners share the same sequence, every learner points at
    # weight_idx. Otherwise, bounds-check the learner and offset into the
    # per-learner array.
    lines.append("inline const uint32_t* getGeneratedCodebookWeightIdx(const GeneratedCodebookLayerView* view, std::size_t learner = 0) {\n")
    lines.append("    if (view == nullptr) {\n")
    lines.append("        return nullptr;\n")
    lines.append("    }\n")
    lines.append("    if (view->same_seq || view->weight_idx_by_learner == nullptr) {\n")
    lines.append("        return view->weight_idx;\n")
    lines.append("    }\n")
    lines.append("    if (learner >= view->n_learners) {\n")
    lines.append("        return nullptr;\n")
    lines.append("    }\n")
    lines.append("    return view->weight_idx_by_learner + learner * getGeneratedWeightIdxCount(*view);\n")
    lines.append("}\n\n")

    # Generated helper: return the pre-interleaved packed-index view, if one was
    # generated for this layer.
    lines.append("inline const uint32_t* getGeneratedCodebookWeightIdxInterleaved(const GeneratedCodebookLayerView* view) {\n")
    lines.append("    return (view == nullptr) ? nullptr : view->weight_idx_interleaved;\n")
    lines.append("}\n\n")

    # Generated helper: return one learner's int8 codebook. If the registry only
    # has the single codebook_int8 pointer, use that as the fallback.
    lines.append("inline const int8_t* getGeneratedCodebookInt8(const GeneratedCodebookLayerView* view, std::size_t learner = 0) {\n")
    lines.append("    if (view == nullptr) {\n")
    lines.append("        return nullptr;\n")
    lines.append("    }\n")
    lines.append("    if (view->codebooks_int8 == nullptr) {\n")
    lines.append("        return view->codebook_int8;\n")
    lines.append("    }\n")
    lines.append("    if (learner >= view->n_learners) {\n")
    lines.append("        return nullptr;\n")
    lines.append("    }\n")
    lines.append("    return view->codebooks_int8 + learner * getGeneratedCodebookSize(*view);\n")
    lines.append("}\n\n")

    # Generated helper: return the [codeword][learner] int8 layout, if present.
    lines.append("inline const int8_t* getGeneratedCodebookInt8Interleaved(const GeneratedCodebookLayerView* view) {\n")
    lines.append("    return (view == nullptr) ? nullptr : view->codebook_int8_interleaved;\n")
    lines.append("}\n\n")

    # Generated helper: return one learner's fp32 codebook. This is useful for
    # verification because it preserves the unquantized/generated values.
    lines.append("inline const float* getGeneratedCodebookFp32(const GeneratedCodebookLayerView* view, std::size_t learner = 0) {\n")
    lines.append("    if (view == nullptr) {\n")
    lines.append("        return nullptr;\n")
    lines.append("    }\n")
    lines.append("    if (view->codebooks_fp32 == nullptr) {\n")
    lines.append("        return view->codebook_fp32;\n")
    lines.append("    }\n")
    lines.append("    if (learner >= view->n_learners) {\n")
    lines.append("        return nullptr;\n")
    lines.append("    }\n")
    lines.append("    return view->codebooks_fp32 + learner * getGeneratedCodebookSize(*view);\n")
    lines.append("}\n\n")

    # Generated helper: return the [codeword][learner] fp32 layout, if present.
    lines.append("inline const float* getGeneratedCodebookFp32Interleaved(const GeneratedCodebookLayerView* view) {\n")
    lines.append("    return (view == nullptr) ? nullptr : view->codebook_fp32_interleaved;\n")
    lines.append("}\n\n")

    # Generated helper: return one learner's bias vector. Layers without bias
    # keep these pointers as nullptr.
    lines.append("inline const float* getGeneratedCodebookBias(const GeneratedCodebookLayerView* view, std::size_t learner = 0) {\n")
    lines.append("    if (view == nullptr) {\n")
    lines.append("        return nullptr;\n")
    lines.append("    }\n")
    lines.append("    if (view->biases == nullptr) {\n")
    lines.append("        return view->bias;\n")
    lines.append("    }\n")
    lines.append("    if (learner >= view->n_learners) {\n")
    lines.append("        return nullptr;\n")
    lines.append("    }\n")
    lines.append("    return view->biases + learner * view->output_size;\n")
    lines.append("}\n\n")

    # Generated helper: return the [output_channel][learner] bias layout, if
    # present.
    lines.append("inline const float* getGeneratedCodebookBiasInterleaved(const GeneratedCodebookLayerView* view) {\n")
    lines.append("    return (view == nullptr) ? nullptr : view->bias_interleaved;\n")
    lines.append("}\n")

    return "".join(lines)



def write_generated_registry_header(entries, filename):
    """
    Write generated_codebook_registry.h to disk.

    The parent directory is created if needed because notebooks often point this
    at generated output folders that may not exist yet.
    """
    filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)
    text = build_generated_registry_header(entries)
    filename.write_text(text, encoding="utf-8")
    print(f"Wrote registry header to: {filename}")
