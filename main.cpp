
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

// Main routine of a program that reads a sparse matrix, right side
// vector, solution vector and initial guess from a file  in HPC
// format.  This program then calls the HPCCG conjugate gradient
// solver to solve the problem, and then prints results.

// Calling sequence:

// test_HPCCG linear_system_file

// Routines called:

// mytimer - Timing routine (compile with -DWALL to get wall clock
//           times

// HPCCG - CG Solver

// compute_residual - Compares HPCCG solution to known solution.

#include <iostream>
using std::cout;
using std::cerr;
using std::endl;
#include <cstdio>
#include <cstdlib>
#include <cctype>
#include <cassert>
#include <string>
#include <cmath>
#include "generate_matrix.hpp"
#include "mytimer.hpp"
#include "HPC_sparsemv.hpp"
#include "compute_residual.hpp"
#include "HPCCG.hpp"
#include "HPC_Sparse_Matrix.hpp"

#include "YAML_Element.hpp"
#include "YAML_Doc.hpp"

#undef DEBUG

int main(int argc, char *argv[])
{

  HPC_Sparse_Matrix *A;
  float *x, *b, *xexact;
  float norm, d;
  int ierr = 0;
  int i, j;
  int ione = 1;
  double times[7];
  double t6 = 0.0;
  int nx,ny,nz;

  int size = 1; // Serial case (not using MPI)
  int rank = 0; 

  if(argc!=4) {
    cerr << "Usage:" << endl
	   << "Mode 1: " << argv[0] << " nx ny nz" << endl
	   << "     where nx, ny and nz are the local sub-block dimensions, or" << endl
	   << "Mode 2: " << argv[0] << " HPC_data_file " << endl
	   << "     where HPC_data_file is a globally accessible file containing matrix data." << endl;
    exit(1);
  }

  nx = atoi(argv[1]);
  ny = atoi(argv[2]);
  nz = atoi(argv[3]);
  generate_matrix(nx, ny, nz, &A, &x, &b, &xexact);

  double t1 = mytimer();   // Initialize it (if needed)
  int niters = 0;
  float normr = 0.0f;
  int max_iter = 150;
  float tolerance = 0.0f; // Set tolerance to zero to make all runs do max_iter iterations
  HPCCG( A, b, x, max_iter, tolerance, niters, normr, times);

  // initialize YAML doc

    double fniters = niters; 
    double fnrow = A->nrow;
    double fnnz = A->nnz;
    double fnops_ddot = fniters*4*fnrow;
    double fnops_waxpby = fniters*6*fnrow;
    double fnops_sparsemv = fniters*2*fnnz;
    double fnops = fnops_ddot+fnops_waxpby+fnops_sparsemv;

    YAML_Doc doc("hpccg", "1.0");

    doc.add("Parallelism","");

        doc.get("Parallelism")->add("MPI not enabled","");

        doc.get("Parallelism")->add("OpenMP not enabled","");

    doc.add("Dimensions","");
  doc.get("Dimensions")->add("nx",nx);
  doc.get("Dimensions")->add("ny",ny);
  doc.get("Dimensions")->add("nz",nz);



    doc.add("Number of iterations", niters);
    doc.add("Final residual", normr);
    doc.add("#********** Performance Summary (times in sec) ***********","");

    doc.add("Time Summary","");
    doc.get("Time Summary")->add("Total   ",times[0]);
    doc.get("Time Summary")->add("DDOT    ",times[1]);
    doc.get("Time Summary")->add("WAXPBY  ",times[2]);
    doc.get("Time Summary")->add("SPARSEMV",times[3]);

    doc.add("FLOPS Summary","");
    doc.get("FLOPS Summary")->add("Total   ",fnops);
    doc.get("FLOPS Summary")->add("DDOT    ",fnops_ddot);
    doc.get("FLOPS Summary")->add("WAXPBY  ",fnops_waxpby);
    doc.get("FLOPS Summary")->add("SPARSEMV",fnops_sparsemv);

    doc.add("MFLOPS Summary","");
    doc.get("MFLOPS Summary")->add("Total   ",fnops/times[0]/1.0E6);
    doc.get("MFLOPS Summary")->add("DDOT    ",fnops_ddot/times[1]/1.0E6);
    doc.get("MFLOPS Summary")->add("WAXPBY  ",fnops_waxpby/times[2]/1.0E6);
    doc.get("MFLOPS Summary")->add("SPARSEMV",fnops_sparsemv/(times[3])/1.0E6);

    std::string yaml = doc.generateYAML();
    cout << yaml;

  // Compute difference between known exact solution and computed solution
  // All processors are needed here.

  float residual = 0.0f;
  //  if ((ierr = compute_residual(A->local_nrow, x, xexact, &residual)))
  //  cerr << "Error in call to compute_residual: " << ierr << ".\n" << endl;

  // if (rank==0)
  //   cout << "Difference between computed and exact  = " 
  //        << residual << ".\n" << endl;


  // Finish up
  return 0 ;
} 
