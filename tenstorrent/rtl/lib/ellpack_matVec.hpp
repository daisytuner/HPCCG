#pragma once

#include <cstdint>

namespace tt::daisy {

enum class EllpackHwImpl {
    None = 0,
    FPU = 1,
    SFPU = 2
};

constexpr EllpackHwImpl default_ellpack_hw_impl = EllpackHwImpl::None;

// Perform ELLPACK sparse matrix-vector multiplication: y = A * x
// Parameters:
//   nrow: number of rows in the matrix
//   ncol: number of columns in the matrix
//   ellpack_nnz: total size of ELLPACK arrays (nrow * ellpack_cols)
//   ellpack_cols: number of columns in the ELLPACK representation
//   vals: ELLPACK values array (size nnz)
//   inds: ELLPACK column indices array (size nnz)
//   x: input vector (size ncol)
//   y: output vector (size nrow)
void tt_ellpack_matVec(int nrow, int ncol, int ellpack_nnz, int ellpack_cols,
                     const float * const vals,
                     const int * const inds,
		 const float * const x, float * const y);

}   // namespace tt::daisy
