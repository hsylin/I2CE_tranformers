//
// Created by alireza on 4/24/23.
//

#ifndef FVLLMONTITRANSFORMER_DEBUGGERFUNCTIONS_H
#define FVLLMONTITRANSFORMER_DEBUGGERFUNCTIONS_H
#include "util.h"
#include "linearLayer.h"
#include <iostream>
#include <fstream>
#include <cstdint>
#include <string>



void print_weight(uint32_t* kernel, int n_row, int n_col);
void blockWise2RowWise(const uint32_t * blockWise, uint32_t* rowWise, int n_row, int n_col);
void rowWise2BlockWise(const uint32_t* rowWise, uint32_t* blockWise, int n_row, int n_col);
void write_weight_to_file(const std::string& filename, uint32_t* kernel, int n_row, int n_col);
void read_weight_from_file(const std::string& filename, uint32_t* kernel, int n_row, int n_col);
void interleave_hidden_flag(uint32_t* kernel, int n_row, int n_col, uint32_t hidden_flag);
void interleave_hidden_flag_zero_free(uint32_t*& kernel, int n_row, int n_col, uint32_t hidden_flag);
void printOutputPreview(const uint32_t *out, int size);
void saveOutput(int size, const uint32_t *array, const std::string &dir_name);

#ifdef SIMD
uint64_t getSveLengthBytes();
uint64_t getSveInt32Lanes();
#endif

// New debug helpers moved from transformerBlock.cc

void printPackedMatrix(const char *label,
                       const uint32_t *buffer,
                       std::size_t rows,
                       std::size_t cols);

void savePackedMatrixText(const char *filename,
                          const uint32_t *buffer,
                          std::size_t rows,
                          std::size_t cols);

void dumpPackedMatrixIfEnabled(const std::string& dump_dir,
                               const std::string& filename,
                               const uint32_t* buffer,
                               std::size_t rows,
                               std::size_t cols);

void printPackedTensorAsPythonList(const std::string& var_name,
                                   const uint32_t* packed,
                                   std::size_t rows,
                                   std::size_t cols);

bool tryComputeGroupedCodebookDense4(LinearLayer* const layers[4],
                                     std::size_t seq_len,
                                     uint32_t* const inputs[4],
                                     uint32_t* const outputs[4]);

bool tryComputeGroupedCodebookDense2(LinearLayer* const layers[2],
                                     std::size_t seq_len,
                                     uint32_t* const inputs[2],
                                     uint32_t* const outputs[2]);

void printPackedPreview(const char *label,
                        const uint32_t *buffer,
                        std::size_t packed_size);

void savePackedBuffer(const char *filename,
                      const uint32_t *buffer,
                      std::size_t packed_size);

int8_t unpackPackedValue(const uint32_t *buffer, std::size_t elem_idx);

void comparePackedBuffers(const char *label,
                          const uint32_t *dense_reference,
                          const uint32_t *candidate,
                          std::size_t packed_size);

/*
 * comparePackedBuffers() reports differences on stdout but cannot fail a run on
 * its own, so a numerically wrong build used to exit 0 and its results were
 * indistinguishable from a correct one. These accessors let the caller turn an
 * observed mismatch into a non-zero exit status without changing the signature
 * used by the ~24 existing call sites.
 *
 * Counts are process-wide and are not thread-safe; reference comparison is a
 * single-threaded debug/validation mode.
 */
std::size_t packedBufferMismatchCount();
std::size_t packedBufferComparisonCount();
void resetPackedBufferMismatchCount();

#endif //FVLLMONTITRANSFORMER_DEBUGGERFUNCTIONS_H
