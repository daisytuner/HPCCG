#pragma once

#include <cstdint>

namespace tt::daisy {

enum class EllpackHwImpl {
    None = 0,
    FPU = 1,
    SFPU = 2
};

/**
 * @brief Parameters for ELLPACK matrix-vector multiplication
 */
struct ELLPACKMatVecParams {
    // Data pointers
    const float * vals;
    const int * inds;

    // Matrix dimensions and properties
    int nrow;
    int ncol;
    int ellpack_nnz;
    int ellpack_cols;

    // Input vector properties
    const uint32_t* row_min_cols;
    const uint32_t* row_max_cols;

    // Hardware implementation to use
    EllpackHwImpl hw_impl = EllpackHwImpl::FPU;
};

/**
 * @brief Perform ELLPACK sparse matrix-vector multiplication: y = A * x
 * 
 * @param params Parameters for the ELLPACK matrix-vector multiplication
 * @param x Input vector
 * @param y Output vector
 */
void tt_ellpack_matVec(const ELLPACKMatVecParams& params, const float * x, float * y);

}   // namespace tt::daisy
