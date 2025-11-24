# Case Study: HPCCG mini-app

This repository demonstrates the Daisytuner Optimizing Compiler Collection (DOCC) applied to the HPCCG mini-app, showcasing advanced auto-parallelization (OpenMP) and auto-offloading capabilities (CUDA). DOCC automatically analyzes computational kernels and generates optimized code for each target. The repository includes [Daisy workflows](https://github.com/daisytuner/HPCCG/blob/master/.daisy/hpccg.yml) for comprehensive performance analysis through our [dashboard](https://app.daisytuner.com/daisytuner/HPCCG/runs).

The case study focuses on optimizing three critical conjugate gradient kernels: **SPMV** (Sparse Matrix-Vector Multiplication), **WAXPBY** (vector operations), and **DDOT** (dot product), which are executed iteratively as part of the solver. Each kernel has been optimized by DOCC to exploit parallelism, optimize memory access patterns, and minimize data transfers for offloading targets with dedicated memory.
