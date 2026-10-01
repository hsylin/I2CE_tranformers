
import numpy as np
import random as rand
from string import Template
import math
import os
from tqdm import tqdm



def to_bin_N_digits(num, N_digits=2):
    binary = bin(num)
    if len(binary.split("b")[1]) < N_digits:
        n_to_insert = N_digits - len(binary.split("b")[1])
        for i in range(n_to_insert):
            binary = binary[:2] + "0" + binary[2:]
    return binary[2:]



def index_bits(codebook_size):
    """Bits per packed codebook index.

    The natural width is ceil(log2(codebook_size)). Setting
    I2CE_IDX_BITS_BYTE_ALIGNED=1 rounds it up to the next width that divides 8
    (1, 2, 4, 8), so an index never straddles a byte boundary and the SVE
    kernels can decode a packed word straight into svdot byte lanes.

    The cost is index memory: an 8-entry codebook goes from 3 bits to 4, a
    third more. That only pays off where the region is instruction-bound, so
    it stays opt-in and the two settings are meant to be compared.
    """
    bits = math.ceil(math.log2(codebook_size))
    if os.environ.get("I2CE_IDX_BITS_BYTE_ALIGNED", "0") == "1":
        for aligned in (1, 2, 4, 8):
            if bits <= aligned:
                return aligned
        raise ValueError(
            "I2CE_IDX_BITS_BYTE_ALIGNED cannot pad %d-bit indexes; "
            "codebook_size %d exceeds 256" % (bits, codebook_size))
    return bits


def compute_cb_parameters(codebook_size, words_bitlen=32):
    idxs_bits = index_bits(codebook_size)
    idxs_per_word = int(math.modf(words_bitlen / idxs_bits)[1])

    return idxs_bits, idxs_per_word



"""
    Rearrange the matrix according to tiles and order them sliding by width.
    This means that starting from the raw matrix, tiles are extracted and saved first along the width-axis
"""
def tile_matrix_W_wise(mat, h_size, w_size, tile_size):
    
    tiled = []
    
    for wtiles in range(0, w_size, tile_size):
        for htiles in range(0, h_size, tile_size):

            tile = []
            # print()
            # for tw in range(tile_size):
            #     for th in range(tile_size):

            # Save the tiles, each one stored colum-major
            for th in range(tile_size):
                for tw in range(tile_size):
                    hindex = htiles + th
                    windex = wtiles + tw
                    # print(hindex, " - ", windex, " ({}, {})".format(h_size, w_size), end="")
                    if (hindex < h_size) and (windex < w_size):
                        # print(" --> ", mat[hindex][windex], end="")
                        tile.append(mat[hindex][windex])
                    # print()
            # print(tile)
            tiled.append(tile)
    # print(tiled)
    return tiled




def generate_template_dense(same_seq, filename, layer_ID, n_learners, codebook_size, tile_size, in_size, out_size, use_f16, use_codebooks):

    words_bitlen= 32
    # if not use_f16:
    #     words_bitlen= 32
    # else:
    #     words_bitlen = 16

    if n_learners <= 4:
        groups_of_4_learners = 1
    else:
        groups_of_4_learners = int(n_learners / 4)

    biases_strings, biases_values_learners = gen_biases_strings(layer_ID, n_learners, out_size, groups_of_4_learners, use_f16)

    if use_codebooks:
        idxs_bits, idxs_per_word = compute_cb_parameters(codebook_size, words_bitlen)
        words_per_row = math.ceil(in_size / idxs_per_word)

        cb_string, cb_interl_string, codebooks_ensembles = gen_codebooks(layer_ID, n_learners, codebook_size, use_f16, groups_of_4_learners)
        indexes_string, values, indexes_tiled_packed_string, _packed_idx_words = generate_dense_indexes(same_seq, codebooks_ensembles, n_learners, layer_ID, codebook_size, in_size, out_size, words_per_row, tile_size, words_bitlen, use_f16)
        # indexes_string, values, indexes_tiled_packed_string = generate_dense_indexes_old(same_seq, codebooks_ensembles, n_learners, layer_ID, codebook_size, in_size, out_size, words_per_row, tile_size, words_bitlen=32)

        no_cb_weights_string = ""
        bias_string_tot = biases_strings[0] + "\n\n\n" + biases_strings[1]
    else:
        no_cb_weights_string, values = generate_weights_no_CB(layer_ID, n_learners, in_size, out_size, use_f16)

        cb_string = ""
        cb_interl_string = ""
        indexes_string = ""
        indexes_tiled_packed_string = ""
        bias_string_tot = biases_strings[0]

    # no_cb_weights_string = gen_no_cb_weights_string(layer_ID, values)

    with open("./templates/dense_weights_template.tpl") as f:
        cont = f.read()
        tpl = Template(cont)

        header = tpl.substitute(
            ID = layer_ID,
            input_size = in_size,
            otuput_size = out_size,
            codebook_string = cb_string,
            codebook_string_interleaved = cb_interl_string,
            indexes_packed_string = indexes_string,
            indexes_packed_tiled_string = indexes_tiled_packed_string,
            no_cb_weights_string = no_cb_weights_string,
            bias_strings = bias_string_tot
            # codebook_string_interleaved_f16 = cb_interl_string_f16,
        )
    
    with open(filename, "w") as f:
        f.write(header)

    return values, biases_values_learners
    

def gen_codebooks(layer_ID, n_learners, codebook_size, use_f16, groups_of_4_learners):
    # Generate the codebooks

    codebook_string = "static const float{} codebooks{}_{}[N_LEARNERS][CB_SIZE] = {{\n"

    if not use_f16:
        codebook_string = codebook_string.format("", "", layer_ID)
    else:
        codebook_string = codebook_string.format("16_t", "_f16", layer_ID)
        

    codebooks = []
    # codebooks_strings = []
    # cb_interleaved_string = "\n"


    for ens_i in range(n_learners):
        codebook_string += "\t{\n"
        # cb_string = "\n\t{\n"
        cb = []

        # print("Codebook ens ", ens_i)
        # print("{")
        for codeword_i in range(codebook_size):

            entry = rand.uniform(-0.1, 0.1)


            codebook_string += "\t\t{},\n".format(entry)
            # cb_string += "\t\t{},\n".format(entry)
            cb.append(entry)

        codebook_string += "\t},\n"
        # print("}")
        # print()
        codebooks.append(cb)
        # cb_string += "\t},\n"
        # codebooks_strings.append(cb_string)

    codebook_string += "};\n"


    if groups_of_4_learners == 1:
        cb_interleaved_string = "static const float{} codebook_interleaved{}_{}[CB_SIZE * N_LEARNERS] = {{\n"
    else:
        cb_interleaved_string = "#define GROUPS_OF_4_LEARNERS {}\n\n".format(groups_of_4_learners)
        cb_interleaved_string += "static const float{} codebook_interleaved{}_{}[GROUPS_OF_4_LEARNERS][CB_SIZE * N_LEARNERS] = {{\n"

    if not use_f16:
        cb_interleaved_string = cb_interleaved_string.format("", "", layer_ID)
    else:
        cb_interleaved_string = cb_interleaved_string.format("16_t", "_f16", layer_ID)

    # This is to store the codebooks in an interleaved manner
    # print("Codebooks interleaved [CB0[0], CB1[0], CB0[1], CB1[1] ... ] :")
    # print("{")

    if groups_of_4_learners == 1:
        for i in range(codebook_size):
            for ens_i in range(n_learners):
                # print("\t{},".format(codebooks[ens_i][i]))
                cb_interleaved_string += "\t{},\n".format(codebooks[ens_i][i])
        cb_interleaved_string += "};\n"
    else:
        for group in range(groups_of_4_learners):
            cb_interleaved_string += "\t{\n"
            for i in range(codebook_size):
                for ens_i in range(4):
                    cb_interleaved_string += "\t\t{},\n".format(codebooks[(group * 4) + ens_i][i])
            cb_interleaved_string += "\t},\n"
        cb_interleaved_string += "};\n"

    return codebook_string, cb_interleaved_string, codebooks



def gen_codebooks_int(layer_ID, n_learners, codebook_size, use_f16, groups_of_4_learners): # generates int8 codebooks, to be used with TiC-SAT
    # Generate the codebooks

    codebook_string = "static const int8_t codebooks{}_{}[N_LEARNERS][CB_SIZE] = {{\n"

    if not use_f16:
        codebook_string = codebook_string.format("", "", layer_ID)
    else:
        codebook_string = codebook_string.format("16_t", "_f16", layer_ID)
        

    codebooks = []
    # codebooks_strings = []
    # cb_interleaved_string = "\n"


    for ens_i in range(n_learners):
        codebook_string += "\t{\n"
        # cb_string = "\n\t{\n"
        cb = []

        # print("Codebook ens ", ens_i)
        # print("{")
        for codeword_i in range(codebook_size):

            # entry = rand.uniform(-0.1, 0.1)
            entry = np.random.randint(-2, 2, dtype=np.int8)

            codebook_string += "\t\t{},\n".format(entry)
            # cb_string += "\t\t{},\n".format(entry)
            cb.append(entry)

        codebook_string += "\t},\n"
        # print("}")
        # print()
        codebooks.append(cb)
        # cb_string += "\t},\n"
        # codebooks_strings.append(cb_string)

    codebook_string += "};\n"


    if groups_of_4_learners == 1:
        cb_interleaved_string = "static const int8_t codebook_interleaved{}_{}[CB_SIZE * N_LEARNERS] = {{\n"
    else:
        cb_interleaved_string = "#define GROUPS_OF_4_LEARNERS {}\n\n".format(groups_of_4_learners)
        cb_interleaved_string += "static const int8_t codebook_interleaved{}_{}[GROUPS_OF_4_LEARNERS][CB_SIZE * N_LEARNERS] = {{\n"

    if not use_f16:
        cb_interleaved_string = cb_interleaved_string.format("", "", layer_ID)
    else:
        cb_interleaved_string = cb_interleaved_string.format("16_t", "_f16", layer_ID)

    # This is to store the codebooks in an interleaved manner
    # print("Codebooks interleaved [CB0[0], CB1[0], CB0[1], CB1[1] ... ] :")
    # print("{")

    if groups_of_4_learners == 1:
        for i in range(codebook_size):
            for ens_i in range(n_learners):
                # print("\t{},".format(codebooks[ens_i][i]))
                cb_interleaved_string += "\t{},\n".format(codebooks[ens_i][i])
        cb_interleaved_string += "};\n"
    else:
        for group in range(groups_of_4_learners):
            cb_interleaved_string += "\t{\n"
            for i in range(codebook_size):
                for ens_i in range(4):
                    cb_interleaved_string += "\t\t{},\n".format(codebooks[(group * 4) + ens_i][i])
            cb_interleaved_string += "\t},\n"
        cb_interleaved_string += "};\n"

    return codebook_string, cb_interleaved_string, codebooks




def get_idxs_and_values(same_seq, n_learners, in_size, out_size, codebook_size, codebooks_ensembles):

    values = [[] for _ in range(n_learners)]    # To store all the weights, as extracted from the codebooks
    indexes = []                                # To store all the indexes to access the codebooks

    if same_seq:
        for i in range(in_size * out_size):
            new_idx = np.random.randint(0, codebook_size)
            indexes.append(new_idx)

            for ens in range(n_learners):
                new_val = codebooks_ensembles[ens][new_idx]
                values[ens].append(new_val)
    else:
        for ens in range(n_learners):
            learner_idxs = []
            for i in range(in_size * out_size):
                new_idx = np.random.randint(0, codebook_size)
                learner_idxs.append(new_idx)

                new_val = codebooks_ensembles[ens][new_idx]
                values[ens].append(new_val)
            indexes.append(learner_idxs)

    return values, indexes



def get_idxs_and_values_fast(same_seq, n_learners, in_size, out_size, codebook_size, codebooks_ensembles):
    total_size = in_size * out_size
    codebooks_ensembles = [np.asarray(cb) for cb in codebooks_ensembles]
    if same_seq:
        # Generate a single random index array
        indexes = np.random.randint(0, codebook_size, size=total_size)

        # Use vectorized indexing for all learners
        values = [
            codebooks_ensembles[ens][indexes]
            for ens in range(n_learners)
        ]
        return values, indexes.tolist()  # Convert to lists if needed

    else:
        # Generate one index array per learner
        indexes = np.random.randint(0, codebook_size, size=(n_learners, total_size))

        # Use vectorized indexing again
        values = [
            codebooks_ensembles[ens][indexes[ens]]
            for ens in range(n_learners)
        ]
        return values, indexes.tolist()



# def gen_idxs_string_per_learner(indexes, idxs_bits, idx_per_word, in_size, out_size, use_f16=False):

#     bin_string = "" # binary indexes formatted as a string for the header file
#     bin_idxs = []   # binary indexes list


#     # For the f16 implementation, if we have 3 bits per index, we want to align the indexes so that there is no index 
#     # split by the lowest and highest half of a word
#     if (idxs_bits == 3) and (use_f16):
#         pad_bits = True
#         n_0_bits_pad = 1
#         n_idxs_before_pad = 5
#     else:
#         pad_bits = False

#     idx_pointer = 0
#     for r in range(out_size):

#         bin_string += "\t"
#         w_row = []
#         row_in_binray = []
#         word_idxs = ""
#         idxs_inserted = 0
#         bits_inserted = 0

#         for c in range(in_size):
#             binary_idx = to_bin_N_digits(indexes[idx_pointer], N_digits=idxs_bits)
#             idx_pointer += 1

#             if idxs_inserted < idx_per_word:
#                 word_idxs = binary_idx + word_idxs
#                 idxs_inserted += 1
#             else:
#                 word_idxs = "0b" + word_idxs
#                 bin_string += "{}, ".format(word_idxs)

#                 row_in_binray.append(word_idxs)
#                 idxs_inserted = 0
#                 word_idxs = ""
#                 word_idxs  = binary_idx + word_idxs
#                 idxs_inserted += 1
        
#             if pad_bits:
#                 if idxs_inserted == n_idxs_before_pad:
#                     word_idxs = ("0" * n_0_bits_pad) + word_idxs
            
    
#         word_idxs = "0b" + word_idxs
#         bin_string += "{},\n".format(word_idxs)
#         row_in_binray.append(word_idxs)
#         idxs_inserted = 0
#         word_idxs = ""
        
#     return bin_string, bin_idxs


""" Jerry: Add new return value: the string for the packed indexes and also the list of packed uint32 words, in row-major order:"""
def gen_idxs_string_per_learner(indexes, idxs_bits, idx_per_word, in_size, out_size, use_f16=False):
    """
    Generate the C string for packed codebook indices and also return
    the raw packed uint32 words in row-major order:
    [row0_word0, row0_word1, ..., row1_word0, ...]
    """

    bin_string = ""
    packed_words = []

    # For f16 + 3-bit indices, keep the same alignment rule as the original code.
    if (idxs_bits == 3) and use_f16:
        pad_bits = True
        n_0_bits_pad = 1
        n_idxs_before_pad = 5
    else:
        pad_bits = False

    idx_pointer = 0

    for r in range(out_size):
        bin_string += "\t"
        word_idxs = ""
        idxs_inserted = 0

        for c in range(in_size):
            binary_idx = to_bin_N_digits(indexes[idx_pointer], N_digits=idxs_bits)
            idx_pointer += 1

            if idxs_inserted < idx_per_word:
                word_idxs = binary_idx + word_idxs
                idxs_inserted += 1
            else:
                word_str = "0b" + word_idxs
                bin_string += "{}, ".format(word_str)
                packed_words.append(np.uint32(int(word_str, 2)))

                idxs_inserted = 0
                word_idxs = ""
                word_idxs = binary_idx + word_idxs
                idxs_inserted += 1

            if pad_bits and idxs_inserted == n_idxs_before_pad:
                word_idxs = ("0" * n_0_bits_pad) + word_idxs

        word_str = "0b" + word_idxs
        bin_string += "{},\n".format(word_str)
        packed_words.append(np.uint32(int(word_str, 2)))

    return bin_string, packed_words # new return value: the list of packed uint32 words, in row-major order



# def generate_dense_indexes(same_seq, codebooks_ensembles, n_learners, layer_ID, codebook_size, in_size, out_size, words_per_row, tile_size, words_bitlen, use_f16):
    
#     idxs_bits, idxs_per_word = compute_cb_parameters(codebook_size, words_bitlen)

#     # values, indexes = get_idxs_and_values(same_seq, n_learners, in_size, out_size, codebook_size, codebooks_ensembles)
#     values, indexes = get_idxs_and_values_fast(same_seq, n_learners, in_size, out_size, codebook_size, codebooks_ensembles)


#     if same_seq:
#         if not use_f16:
#             compact_indexes_string = "static uint{}_t weight_idx_compact{}_{}[OUTPUT_SIZE_{} * N_WORDS_ROW_{}] = {{\n".format("32", "", layer_ID, layer_ID, layer_ID)
#         else:
#             compact_indexes_string = "static uint{}_t weight_idx_compact{}_{}[OUTPUT_SIZE_{} * N_WORDS_ROW_{}] = {{\n".format("32", "_f16", layer_ID, layer_ID, layer_ID)
#             # compact_indexes_string = "static uint{}_t weight_idx_compact{}_{}[OUTPUT_SIZE_{} * N_WORDS_ROW_{}] = {{\n".format("16", "_f16", layer_ID, layer_ID, layer_ID)

#         bin_idxs_string, bin_idxs = gen_idxs_string_per_learner(indexes, idxs_bits, idxs_per_word, in_size, out_size, use_f16=use_f16)
#         compact_indexes_string += bin_idxs_string
#         compact_indexes_string += "};\n\n"
    
#     else:
#         if not use_f16:
#             compact_indexes_string = "static uint{}_t weight_idx_compact{}_{}[N_LEARNERS][OUTPUT_SIZE_{} * N_WORDS_ROW_{}] = {{\n".format("32", "", layer_ID, layer_ID, layer_ID)
#         else:
#             compact_indexes_string = "static uint{}_t weight_idx_compact{}_{}[N_LEARNERS][OUTPUT_SIZE_{} * N_WORDS_ROW_{}] = {{\n".format("32", "_f16", layer_ID, layer_ID, layer_ID)
#             # compact_indexes_string = "static uint{}_t weight_idx_compact{}_{}[N_LEARNERS][OUTPUT_SIZE_{} * N_WORDS_ROW_{}] = {{\n".format("16", "_f16", layer_ID, layer_ID, layer_ID)
        
#         for ens in range(n_learners):
#             compact_indexes_string += "\t{\n\t"
#             bin_idxs_string, bin_idxs = gen_idxs_string_per_learner(indexes[ens], idxs_bits, idxs_per_word, in_size, out_size, use_f16=use_f16)
#             compact_indexes_string += "\t" + bin_idxs_string
#             compact_indexes_string += "\t},\n"
#         compact_indexes_string += "};\n\n"


#     tiled_string = ""

#     return compact_indexes_string, values, tiled_string

""" Jerry: Add new return value: the string for the packed indexes and also the list of packed uint32 words, in row-major order:"""
def generate_dense_indexes(
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
):
    """
    Generate packed codebook indices both as C strings and as raw arrays.
    """

    idxs_bits, idxs_per_word = compute_cb_parameters(codebook_size, words_bitlen)

    values, indexes = get_idxs_and_values_fast(
        same_seq, n_learners, in_size, out_size, codebook_size, codebooks_ensembles
    )

    if same_seq:
        if not use_f16:
            compact_indexes_string = (
                "static uint32_t weight_idx_compact_{}[OUTPUT_SIZE_{} * N_WORDS_ROW_{}] = {{\n"
                .format(layer_ID, layer_ID, layer_ID)
            )
        else:
            compact_indexes_string = (
                "static uint32_t weight_idx_compact_f16_{}[OUTPUT_SIZE_{} * N_WORDS_ROW_{}] = {{\n"
                .format(layer_ID, layer_ID, layer_ID)
            )

        bin_idxs_string, packed_idx_words = gen_idxs_string_per_learner(
            indexes, idxs_bits, idxs_per_word, in_size, out_size, use_f16=use_f16
        )

        compact_indexes_string += bin_idxs_string
        compact_indexes_string += "};\n\n"

        packed_idx_words = np.asarray(packed_idx_words, dtype=np.uint32)

    else:
        if not use_f16:
            compact_indexes_string = (
                "static uint32_t weight_idx_compact_{}[N_LEARNERS][OUTPUT_SIZE_{} * N_WORDS_ROW_{}] = {{\n"
                .format(layer_ID, layer_ID, layer_ID)
            )
        else:
            compact_indexes_string = (
                "static uint32_t weight_idx_compact_f16_{}[N_LEARNERS][OUTPUT_SIZE_{} * N_WORDS_ROW_{}] = {{\n"
                .format(layer_ID, layer_ID, layer_ID)
            )

        all_packed_idx_words = []

        for ens in range(n_learners):
            compact_indexes_string += "\t{\n\t"

            bin_idxs_string, packed_idx_words = gen_idxs_string_per_learner(
                indexes[ens], idxs_bits, idxs_per_word, in_size, out_size, use_f16=use_f16
            )
            compact_indexes_string += "\t" + bin_idxs_string
            compact_indexes_string += "\t},\n"

            all_packed_idx_words.append(np.asarray(packed_idx_words, dtype=np.uint32))

        compact_indexes_string += "};\n\n"
        packed_idx_words = all_packed_idx_words

    tiled_string = ""

    return compact_indexes_string, values, tiled_string, packed_idx_words # new return value: the list of packed uint32 words, in row-major order





def generate_dense_indexes_old(same_seq, codebooks_ensembles, n_learners, layer_ID, codebook_size, in_size, out_size, words_per_row, tile_size, words_bitlen=32):
    
    idxs_bits, idxs_per_word = compute_cb_parameters(codebook_size, words_bitlen)
    
    # To store all the weights per kernel, as extracted from the codebooks
    values = [[] for _ in range(n_learners)]    

    weights = []
    bin_rows = []

    weight_string = "\n"

    # compact_indexes_string = "static const uint32_t weight_idx_compact_{}[OUTPUT_SIZE_{}][N_WORDS_ROW_{}] = {{\n".format(layer_ID, layer_ID, layer_ID)
    compact_indexes_string = "static const uint32_t weight_idx_compact_{}[OUTPUT_SIZE_{} * N_WORDS_ROW_{}] = {{\n".format(layer_ID, layer_ID, layer_ID)

    # print("{")
    for r in range(out_size):

        # compact_indexes_string += "\t{"
        compact_indexes_string += "\t"

        w_row = []
        # print("\t{", end="")
        row_in_binary = []
        word_idxs = ""
        idxs_inserted = 0
        bits_inserted = 0
        weight_string += "\t{"

        for c in range(in_size):
            new_idx = rand.randint(0, codebook_size-1)
            binary_idx = to_bin_N_digits(new_idx, N_digits=idxs_bits)
        
            # Insert the binary indexes
            if idxs_inserted < idxs_per_word:
                word_idxs = binary_idx + word_idxs
                idxs_inserted += 1
            else:
                word_idxs = "0b" + word_idxs
                compact_indexes_string += "{}, ".format(word_idxs)

                row_in_binary.append(word_idxs)
                idxs_inserted = 0
                word_idxs = ""
                word_idxs = binary_idx + word_idxs
                idxs_inserted += 1
                
            # print("{}, ".format(new_idx), end="")
            weight_string += "{}, ".format(new_idx)
            w_row.append(new_idx)

            for ens in range(n_learners):
                new_val = codebooks_ensembles[ens][new_idx]
                values[ens].append(new_val)

        # print("},")
        weight_string += "},\n"
        word_idxs = "0b" + word_idxs
        compact_indexes_string += "{},\n".format(word_idxs)
        row_in_binary.append(word_idxs)
        idxs_inserted = 0
        bits_inserted = 0
        word_idxs = ""
        bin_rows.append(row_in_binary)
        weights.append(w_row)
    compact_indexes_string += "};\n"

    tile_indexes_compact_string = tile_weights_indexes(layer_ID, bin_rows, out_size, words_per_row, tile_size)

    return compact_indexes_string, values, tile_indexes_compact_string

    # print("};")



def tile_weights_indexes(layer_ID, bin_rows, out_size, words_per_row, TILE_SIZE):
    
    # TILE Compact weights #
    tiled_compact_weights = tile_matrix_W_wise(bin_rows, out_size, words_per_row, TILE_SIZE)

    tiled_w_compact_string = "static const uint32_t weight_indexes_compact_tiled_{}[N_WORDS_ROW_{}*OUTPUT_SIZE_{}] = {{\n".format(layer_ID, layer_ID, layer_ID)

    for tile in tiled_compact_weights:
        for e in tile:
            tiled_w_compact_string += "\t{}, ".format(e)
        tiled_w_compact_string += "\n"
    tiled_w_compact_string += "};\n"

    return tiled_w_compact_string






def generate_weights_no_CB(layer_ID, n_learners, in_size, out_size, use_f16):

    weights_no_CB_string = "static const float{} weights{}_{}[N_LEARNERS][OUTPUT_SIZE_{} * INPUT_SIZE_{}] = {{\n"

    if not use_f16:
        weights_no_CB_string = weights_no_CB_string.format("", "", layer_ID, layer_ID, layer_ID)
    else:
        weights_no_CB_string = weights_no_CB_string.format("_f16", "_f16", layer_ID, layer_ID, layer_ID)
    
    raw_weights = np.random.uniform(-1.0, 1.0, size=(n_learners, in_size * out_size))

    for ens in range(n_learners):
        weights_no_CB_string += "\t{\n"
        for w in raw_weights[ens]:
            weights_no_CB_string += "\t\t{},\n".format(w)
        weights_no_CB_string += "\t},\n"
    weights_no_CB_string += "};\n\n"

    return weights_no_CB_string, raw_weights










def gen_no_cb_weights_string(layer_ID, values):

    weights_string = "#define N_WEIGHTS_{}  (INPUT_SIZE_{} * OUTPUT_SIZE_{})\n\n".format(layer_ID, layer_ID, layer_ID)
    weights_string += "static float non_cb_weights_{}[N_LEARNERS][N_WEIGHTS_{}] = {{\n".format(layer_ID, layer_ID)

    for ens in values:
        weights_string += "\t{\n"

        for v in ens:
            weights_string += "\t\t{},\n".format(v)
        weights_string += "\t},\n"
    weights_string += "};\n"
    return weights_string






def gen_biases_strings(layer_ID, n_learners, out_size, groups_of_4_learners, use_f16):
    
    bias_c_type = "float{}"
    
    if use_f16:
        bias_c_type = bias_c_type.format("16_t")
    else:
        bias_c_type = bias_c_type.format("")

    bias_string = "static " + bias_c_type + " bias_{}[N_LEARNERS][OUTPUT_SIZE_{}] = {{\n".format(layer_ID, layer_ID)

    if groups_of_4_learners == 1:
        bias_string_interleaved = "static " + bias_c_type + " bias_interleaved_{}[N_LEARNERS * OUTPUT_SIZE_{}] = {{\n".format(layer_ID, layer_ID)
    else:
        bias_string_interleaved = "static " + bias_c_type + " bias_interleaved_{}[GROUPS_OF_4_LEARNERS][N_LEARNERS * OUTPUT_SIZE_{}] = {{\n".format(layer_ID, layer_ID)

    bias_ensembles = [] # Holds all the bias of the learners

    # Generate the codebook for all the learners
    for i in range(n_learners):
        bias_ensembles.append(np.random.uniform(0.0, 0.9, out_size))

    # Fill the bias string in C format
    for bias_learn in bias_ensembles:
        bias_string += "\t{\n"
        for v in bias_learn:
            bias_string += "\t\t{},\n".format(v)
        bias_string += "\t},\n"
    bias_string += "};"


    if groups_of_4_learners == 1:
        for i in range(out_size):
            for ens in range(n_learners):
                bias_string_interleaved += "\t{},\n".format(bias_ensembles[ens][i])
        bias_string_interleaved += "};"
    else:
        for group in range(groups_of_4_learners):
            bias_string_interleaved += "\t{\n"
            for i in range(out_size):
                for ens in range(4):
                    bias_string_interleaved += "\t\t{},\n".format(bias_ensembles[(group * 4) + ens][i])
            bias_string_interleaved += "\t},\n"
        bias_string_interleaved += "};\n"

    return (bias_string, bias_string_interleaved), bias_ensembles

