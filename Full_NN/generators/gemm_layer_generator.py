"""
GEMM layer header and TiC-SAT RWMA weight export generator.

This module is used by the Transformer/GEMM generator notebooks to emit C/C++
artifacts consumed by generated TiC-SAT kernels.

It produces two related kinds of output:

1. Legacy gemm_header_X.h-style headers from templates/gemm_weights_template.tpl.
   These headers contain per-layer GEMM constants, generated codebooks, packed
   codebook indexes, optional raw weights, and optional biases.
2. TiC-SAT RWMA weight text files for the systolic-array/RWMA execution path.
   These files store int8 weights as blockwise uint32 words in the exact order
   expected by the C++ loader and its blockWise2RowWise conversion.

The generated C/C++ code can represent weights either directly as dense values
or indirectly through codebooks. In the codebook path, packed integer indexes
select entries from small codebook arrays instead of storing every weight value
directly. This reduces memory footprint and lets the kernels reuse vectorized
lookup/decode logic.
"""

import math
import os

import numpy as np
from string import Template

from dense_layer_generator import (
    compute_cb_parameters,
    gen_biases_strings,
    gen_codebooks,
    gen_codebooks_int,
    generate_dense_indexes,
    generate_weights_no_CB,
)



""" Jerry: added registry_meta return value to generate_template_gemm, which contains the metadata dict for registry export."""
def generate_template_gemm(
    same_seq,
    filename,
    layer_ID,
    layer_name,
    n_learners,
    codebook_size,
    tile_size,
    in_size,
    out_size,
    use_f16,
    use_codebooks,
    use_bias,
    registry_learner=0,
    use_fp32_transformer=False,
):
    """
    Generate the legacy gemm_header_X.h file as before,
    and also return a registry-friendly metadata dict.
    """

    words_bitlen = 32

    if n_learners <= 4:
        groups_of_4_learners = 1
    else:
        groups_of_4_learners = int(n_learners / 4)

    biases_strings, biases_values_learners = gen_biases_strings(
        layer_ID, n_learners, out_size, groups_of_4_learners, use_f16
    )

    registry_meta = None

    if use_codebooks:
        # Same helper the packing and the codebooks_def.h macro use, so the
        # registry's per-layer bits_per_cb cannot disagree with the stream that
        # was emitted. This is the width the GEMM kernels are actually passed.
        idxs_bits, idxs_per_word = compute_cb_parameters(codebook_size, words_bitlen)
        words_per_row = math.ceil(in_size / idxs_per_word)

        if use_fp32_transformer:
            cb_string, cb_interl_string, codebooks_ensembles = gen_codebooks(
                layer_ID, n_learners, codebook_size, use_f16, groups_of_4_learners
            )
        else:
            cb_string, cb_interl_string, codebooks_ensembles = gen_codebooks_int(
                layer_ID, n_learners, codebook_size, use_f16, groups_of_4_learners
            )

        indexes_string, values, indexes_tiled_packed_string, packed_idx_words = generate_dense_indexes(
            same_seq,
            codebooks_ensembles,
            n_learners,
            layer_ID,
            codebook_size,
            in_size,
            out_size,
            words_per_row,
            tile_size,
            words_bitlen,
            use_f16,
        )

        no_cb_weights_string = ""
        bias_string_tot = biases_strings[0] + "\n\n\n" + biases_strings[1]

        weight_idx_for_registry = (
            np.asarray(packed_idx_words, dtype=np.uint32)
            if same_seq
            else np.asarray(packed_idx_words[registry_learner], dtype=np.uint32)
        )

        codebooks_arr = np.rint(np.asarray(codebooks_ensembles, dtype=np.float32)).clip(-128, 127).astype(np.int8)
        codebooks_fp32_arr = np.asarray(codebooks_ensembles, dtype=np.float32)
        codebook_for_registry = np.asarray(
            codebooks_arr[registry_learner], dtype=np.int8
        )
        codebook_fp32_for_registry = np.asarray(
            codebooks_ensembles[registry_learner], dtype=np.float32
        )
        codebook_interleaved = np.ascontiguousarray(
            codebooks_arr.transpose(1, 0)
        ).reshape(-1)
        codebook_fp32_interleaved = np.ascontiguousarray(
            codebooks_fp32_arr.transpose(1, 0)
        ).reshape(-1)

        weight_idx_by_learner = None
        weight_idx_interleaved = None
        if not same_seq:
            weight_idx_by_learner = np.asarray(packed_idx_words, dtype=np.uint32)
            weight_idx_interleaved = np.ascontiguousarray(
                weight_idx_by_learner.transpose(1, 0)
            ).reshape(-1)

        bias_for_registry = None
        biases_arr = None
        bias_interleaved = None
        if use_bias:
            biases_arr = np.asarray(biases_values_learners, dtype=np.float32)
            bias_for_registry = np.asarray(
                biases_values_learners[registry_learner], dtype=np.float32
            )
            bias_interleaved = np.ascontiguousarray(
                biases_arr.transpose(1, 0)
            ).reshape(-1)

        registry_meta = {
            "name": layer_name,
            "input_size": int(in_size),
            "output_size": int(out_size),
            "n_words_row": int(words_per_row),
            "bits_per_cb": int(idxs_bits),
            "n_learners": int(n_learners),
            "same_seq": bool(same_seq),
            "weight_idx": weight_idx_for_registry,
            "weight_idx_by_learner": weight_idx_by_learner,
            "weight_idx_interleaved": weight_idx_interleaved,
            "codebook_int8": codebook_for_registry,
            "codebooks_int8": codebooks_arr,
            "codebook_int8_interleaved": codebook_interleaved,
            "codebook_fp32": codebook_fp32_for_registry,
            "codebooks_fp32": codebooks_fp32_arr,
            "codebook_fp32_interleaved": codebook_fp32_interleaved,
            "bias": bias_for_registry,
            "biases": biases_arr,
            "bias_interleaved": bias_interleaved,
        }

    else:
        no_cb_weights_string, values = generate_weights_no_CB(
            layer_ID, n_learners, in_size, out_size, use_f16
        )

        cb_string = ""
        cb_interl_string = ""
        indexes_string = ""
        indexes_tiled_packed_string = ""
        bias_string_tot = biases_strings[0]

    with open("./templates/gemm_weights_template.tpl") as f:
        cont = f.read()
        tpl = Template(cont)

        header = tpl.substitute(
            ID=layer_ID,
            input_size=in_size,
            otuput_size=out_size,
            codebook_string=cb_string,
            codebook_string_interleaved=cb_interl_string,
            indexes_packed_string=indexes_string,
            indexes_packed_tiled_string=indexes_tiled_packed_string,
            no_cb_weights_string=no_cb_weights_string,
            bias_strings=bias_string_tot,
        )

    with open(filename, "w") as f:
        f.write(header)

    return values, biases_values_learners, registry_meta # Return the registry_meta dict for registry export


def generate_gemm_data_file(filename, n_layers):
    header = "#ifndef _GEMM_DATA_H_\n"
    header += "#define _GEMM_DATA_H_\n\n"

    header += "#include <gemm_exec.h>\n\n"
    header += "#include <./input_matrix.h>\n"

    for layer_id in range(n_layers):
        header += "#include <./gemm_header_{}.h>\n".format(layer_id)

    header += "\n#define N_GEMM_LAYERS {}\n\n".format(n_layers)

    for layer_id in range(n_layers):
        header += "static const gemm_t gemm_{} = {{\n".format(layer_id)
        header += "\t.seq_len = GEMM_M,\n"
        header += "\t.input_size = INPUT_SIZE_{},\n".format(layer_id)
        header += "\t.output_size = OUTPUT_SIZE_{},\n".format(layer_id)
        header += "\t.n_words_row = N_WORDS_ROW_{}\n".format(layer_id)
        header += "};\n\n"

    header += "#endif\n"

    with open(filename, "w") as f:
        f.write(header)



""" Jerry: Export weights for TiC-SAT RWMA path."""

def _pack_int8_row_to_words(row_values):
    """
    Pack one row of int8 output-channel weights into little-endian uint32 words.

    The RWMA path consumes weights four int8 values at a time. Each group of
    four output channels is stored in one 32-bit word:

        byte 0 <- row_values[i + 0]
        byte 1 <- row_values[i + 1]
        byte 2 <- row_values[i + 2]
        byte 3 <- row_values[i + 3]

    Values are first viewed as unsigned bytes with np.uint8 so negative int8
    weights keep their two's-complement representation inside the packed word.
    """
    packed = []
    for word_idx in range(0, len(row_values), 4):
        word = 0
        chunk = row_values[word_idx:word_idx + 4]
        for byte_idx, value in enumerate(chunk):
            word |= (int(np.uint8(value)) & 0xFF) << (8 * byte_idx)
        packed.append(word)
    return packed


def _reverse_input_groups_of_4_rows(row_major_matrix):
    """
    Match the input-lane order used by the TiC-SAT RWMA kernel.

    Parameters
    ----------
    row_major_matrix:
        Weight matrix after transposing from PyTorch Linear layout into
        [input_size, output_size] row-major layout.

    The RWMA kernel reads input rows in groups of four lanes, but the hardware
    lane order is reversed within each group. This helper performs exactly that
    local permutation:

        rows [0, 1, 2, 3, 4, 5, 6, 7]
          -> [3, 2, 1, 0, 7, 6, 5, 4]

    The total input dimension must therefore be divisible by four. The output
    dimension is not changed; only the order of input rows is rearranged.
    """
    arr = np.asarray(row_major_matrix)
    if arr.shape[0] % 4 != 0:
        raise ValueError(
            f"input_size={arr.shape[0]} must be divisible by 4 for RWMA row reordering"
        )
    return arr.reshape(arr.shape[0] // 4, 4, arr.shape[1])[:, ::-1, :].reshape(arr.shape)


def export_ticsat_weight_file(
    filename,
    weights,
    input_size,
    output_size,
    kernel_dim=16,
    quant_scale=1.0,
    reverse_input_groups_of_4=True,
):
    """
    Export one GEMM/Linear layer's weights for the TiC-SAT RWMA path.

    This function converts a standard dense weight matrix into the compact text
    format consumed by the TiC-SAT systolic-array/RWMA C++ code. The output file
    is a whitespace-separated list of decimal uint32 words. Each word packs four
    int8 weights, and the words are ordered by output-channel tiles so the C++
    loader can reconstruct the row-wise packed matrix efficiently.

    Expected source layout
    ----------------------
    weights shape == [output_size, input_size]
        This is the same layout used by PyTorch Linear.weight, where each row is
        one output channel and each column is one input feature.

    Exported layout
    ---------------
    blockwise packed uint32 words for TiC-SAT
        The matrix is first transposed to [input_size, output_size], optionally
        reordered to match RWMA input lanes, packed four output channels per
        word, and finally emitted in blockwise column-tile order.

    Assumptions tied to the C++ macros
    ----------------------------------
    W_DATA     = 4
        Four int8 weights are packed into one uint32 word.
    KERNEL_DIM = SA_SIZE = kernel_dim
        The systolic-array tile width used by the generated kernel.
    MAX_COL    = kernel_dim // 4
        Number of packed uint32 columns handled per kernel tile.
    """

    W_DATA = 4

    # Validate the shape constraints required by the C++ RWMA loader. The
    # exporter deliberately fails early here because a misaligned weight file
    # would otherwise be hard to diagnose once loaded into the generated kernel.
    if kernel_dim % W_DATA != 0:
        raise ValueError(
            f"kernel_dim={kernel_dim} must be divisible by W_DATA={W_DATA}"
        )

    if input_size % kernel_dim != 0:
        raise ValueError(
            f"input_size={input_size} must be divisible by kernel_dim={kernel_dim}"
        )

    if output_size % W_DATA != 0:
        raise ValueError(
            f"output_size={output_size} must be divisible by W_DATA={W_DATA}"
        )

    max_col = kernel_dim // W_DATA
    packed_cols = output_size // W_DATA

    # The blockwise traversal below emits max_col packed columns per output
    # tile. Requiring an integer number of such tiles keeps the file layout
    # consistent with blockWise2RowWise on the C++ side.
    if packed_cols % max_col != 0:
        raise ValueError(
            f"packed output columns={packed_cols} must be divisible by max_col={max_col}; "
            f"output_size likely needs to be a multiple of kernel_dim={kernel_dim}"
        )

    # If weights are already int8 / ternary expanded values, keep them.
    # If they are float weights, uncomment the quantization path below instead.
    weights_arr = np.asarray(weights, dtype=np.int8).reshape(output_size, input_size)

    # Optional float quantization path:
    # weights_arr = np.asarray(weights, dtype=np.float32).reshape(output_size, input_size)
    # weights_arr = np.rint(weights_arr * quant_scale).clip(-128, 127).astype(np.int8)

    # Step 1: convert [out, in] -> [in, out].
    # PyTorch stores Linear weights by output channel first. The RWMA loader
    # reconstructs rows by input feature, so the exporter changes the logical
    # row axis before packing any bytes.
    row_major = weights_arr.transpose(1, 0)

    # Step 2: match RWMA input-lane ordering.
    # When enabled, each group of four input rows is reversed to match the lane
    # order expected by the RWMA kernel.
    if reverse_input_groups_of_4:
        row_major = _reverse_input_groups_of_4_rows(row_major)

    # Step 3: pack every 4 output channels into one uint32.
    # After this step, packed_row_major[row][col] contains the four int8
    # weights for output channels [4*col, 4*col+3] at one input row.
    packed_row_major = [_pack_int8_row_to_words(row) for row in row_major]

    # Step 4: export in blockwise order expected by blockWise2RowWise / loader.
    # The nested loops walk one packed output tile at a time. For each tile, all
    # input rows are emitted before moving to the next tile. This is the inverse
    # of the checker function blockwise_to_rowwise below.
    blockwise_words = []
    for col_tile in range(packed_cols // max_col):
        col_start = col_tile * max_col
        for row in range(input_size):
            for jj in range(max_col):
                blockwise_words.append(
                    packed_row_major[row][col_start + jj]
                )

    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "w") as fout:
        fout.write(" ".join(str(int(word)) for word in blockwise_words))

    return blockwise_words



""" Checker function to verify correctness of exported weights against original weight matrix."""

def unpack_word_to_int8s(word):
    """
    Decode one packed uint32 word back into four signed int8 values.

    This is the inverse of _pack_int8_row_to_words for a single word. It is used
    by the checker helpers below to reconstruct the matrix that the C++ RWMA
    path should see after loading the exported text file.
    """
    vals = []
    for b in range(4):
        v = (word >> (8 * b)) & 0xFF
        if v >= 128:
            v -= 256
        vals.append(v)
    return vals


def blockwise_to_rowwise(blockwise_words, input_size, output_size, kernel_dim):
    """
    Convert exported blockwise words back into row-wise packed order.

    The exporter writes one output-column tile at a time. The RWMA checker often
    needs the easier-to-inspect row-wise layout where all packed output columns
    for input row 0 come first, followed by input row 1, and so on.

    The loop order mirrors export_ticsat_weight_file:

    1. Iterate over packed output tiles.
    2. Visit every input row for the current tile.
    3. Copy each packed column within the tile into its row-wise destination.
    """
    W_DATA = 4
    max_col = kernel_dim // W_DATA
    packed_cols = output_size // W_DATA

    rowwise = [0] * (input_size * packed_cols)
    src = 0
    for col_tile in range(packed_cols // max_col):
        base = col_tile * max_col
        for row in range(input_size):
            for i in range(max_col):
                rowwise[row * packed_cols + base + i] = blockwise_words[src]
                src += 1
    return rowwise


def unpack_rowwise_words_to_matrix(rowwise_words, input_size, output_size):
    """
    Expand row-wise packed uint32 words into a signed int8 matrix.

    The returned array has shape [input_size, output_size], matching the
    row-major layout produced inside export_ticsat_weight_file after transpose
    and optional RWMA row-group reversal. This makes it convenient to compare
    exported/reloaded weights against the expected transformed source matrix.
    """
    packed_cols = output_size // 4
    mat = []
    for r in range(input_size):
        row = []
        for c in range(packed_cols):
            row.extend(unpack_word_to_int8s(rowwise_words[r * packed_cols + c]))
        mat.append(row[:output_size])
    return np.array(mat, dtype=np.int8)
