
//@HEADER
// ************************************************************************
// 
//               HPCCG: Simple Conjugate Gradient Benchmark Code
//                 Copyright (2006) Sandia Corporation
// 
// Under terms of Contract DE-AC04-94AL85000, there is a non-exclusive
// license for use of this work by or on behalf of the U.S. Government.
// 
// BSD 3-Clause License
// 
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
// 
// * Redistributions of source code must retain the above copyright notice, this
//   list of conditions and the following disclaimer.
// 
// * Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the following disclaimer in the documentation
//   and/or other materials provided with the distribution.
// 
// * Neither the name of the copyright holder nor the names of its
//   contributors may be used to endorse or promote products derived from
//   this software without specific prior written permission.
// 
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
// 
// Questions? Contact Michael A. Heroux (maherou@sandia.gov) 
// 
// ************************************************************************
//@HEADER

/////////////////////////////////////////////////////////////////////////

// Routine to read a sparse matrix, right hand side, initial guess, 
// and exact solution (as computed by a direct solver).

/////////////////////////////////////////////////////////////////////////

// nrow - number of rows of matrix (on this processor)

#include <iostream>
using std::cout;
using std::cerr;
using std::endl;
#include <cstdlib>
#include <cstdio>
#include <cassert>
#include <algorithm>
#include <vector>
#include <cstdint>
#include "generate_matrix.hpp"
void generate_matrix(int nx, int ny, int nz, HPC_Sparse_Matrix **A, float **x, float **b, float **xexact)

{
#ifdef DEBUG
  int debug = 1;
#else
  int debug = 0;
#endif

  // Step 1: Create CSR matrix for a 27-point stencil on a
  // nx by ny by nz domain

  *A = new HPC_Sparse_Matrix; // Allocate matrix struct and fill it
  (*A)->title = 0;

  // Set this bool to true if you want a 7-pt stencil instead of a 27 pt stencil
  bool use_7pt_stencil = false;

  int nrow = nx*ny*nz; // This is the size of our subblock
  assert(nrow>0); // Must have something to work with
  int nnz = 27*nrow; // Approximately 27 nonzeros per row (except for boundary nodes)  

  // Allocate CSR buffers
  (*A)->nnzs = new int[nrow+1];
  (*A)->nnzs[0] = 0;
  (*A)->vals = new float[nnz];
  (*A)->inds = new int   [nnz];

  // Allocate additional arrays
  *x = new float[nrow];
  *b = new float[nrow];
  *xexact = new float[nrow];

  float * curvalptr = (*A)->vals;
  int * curindptr = (*A)->inds;

  for (int iz=0; iz<nz; iz++) {
    for (int iy=0; iy<ny; iy++) {
      for (int ix=0; ix<nx; ix++) {
        int currow = iz*nx*ny+iy*nx+ix;
        int nnzrow = 0;
        for (int sz=-1; sz<=1; sz++) {
          for (int sy=-1; sy<=1; sy++) {
            for (int sx=-1; sx<=1; sx++) {
      	      int curcol = currow+sz*nx*ny+sy*nx+sx;
//            Since we have a stack of nx by ny by nz domains , stacking in the z direction, we check to see
//            if sx and sy are reaching outside of the domain, while the check for the curcol being valid
//            is sufficient to check the z values
              if ((ix+sx>=0) && (ix+sx<nx) && (iy+sy>=0) && (iy+sy<ny) && (curcol>=0 && curcol<nrow)) {
                if (!use_7pt_stencil || (sz*sz+sy*sy+sx*sx<=1)) { // This logic will skip over point that are not part of a 7-pt stencil
                  if (curcol==currow) {
	            	    *curvalptr++ = 27.0;
		              } else {
		                *curvalptr++ = -1.0;
                  }
		              *curindptr++ = curcol;
		              nnzrow++;
	              } 
              }
	          } // end sx loop
          } // end sy loop
        } // end sz loop
        (*A)->nnzs[currow+1] = (*A)->nnzs[currow]+nnzrow;
        (*x)[currow] = 0.0;
        (*b)[currow] = 27.0 - ((float) (nnzrow-1));
        (*xexact)[currow] = 1.0;
      } // end ix loop
     } // end iy loop
  } // end iz loop  

  (*A)->nrow = nrow;
  (*A)->ncol = nrow;
  (*A)->nnz = nnz;

  // Step 2: Add ellpack layout

  // Find the maximum number of nonzeros per row
  int max_nnz_per_row = 0;
  for (int i = 0; i < (*A)->nrow; i++) {
    int nnz_in_row = (*A)->nnzs[i+1] - (*A)->nnzs[i];
    max_nnz_per_row = std::max(max_nnz_per_row, nnz_in_row);
  }

  // Ensure we don't exceed the ELLPACK width
  int ellpack_cols = 32; // Default ELLPACK width
  if (max_nnz_per_row > ellpack_cols) {
    cerr << "Warning: Max nonzeros per row (" << max_nnz_per_row 
         << ") exceeds ELLPACK width (" << ellpack_cols << "). Truncating." << endl;
    max_nnz_per_row = ellpack_cols;
  }

  // Allocate new ELLPACK arrays
  int ellpack_size = (*A)->nrow * ellpack_cols;
  float* ellpack_vals = new float[ellpack_size];
  int* ellpack_inds = new int[ellpack_size];

  // Initialize arrays with zeros and invalid indices  
  for (int i = 0; i < ellpack_size; i++) {
    ellpack_vals[i] = 0.0f;
    ellpack_inds[i] = UINT32_MAX;  // Use UINT32_MAX as invalid index
  }

  // Convert CSR to ELLPACK format
  for (int row = 0; row < (*A)->nrow; row++) {
    int nnz_in_row = std::min((*A)->nnzs[row+1] - (*A)->nnzs[row], ellpack_cols);
    
    for (int j = 0; j < nnz_in_row; j++) {
      int ellpack_idx = row * ellpack_cols + j;
      int csr_idx = (*A)->nnzs[row] + j;
      
      ellpack_vals[ellpack_idx] = (*A)->vals[csr_idx];
      ellpack_inds[ellpack_idx] = (*A)->inds[csr_idx];
    }
  }
  
  (*A)->ellpack_vals = ellpack_vals;
  (*A)->ellpack_inds = ellpack_inds;
  (*A)->ellpack_cols = ellpack_cols;
  (*A)->ellpack_nnz = ellpack_size;

  return;
}
