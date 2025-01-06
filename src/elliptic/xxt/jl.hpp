#if !defined(_NEKRS_JL_HPP_)
#define _NEKRS_JL_HPP_

#include "elliptic.h"
#include "gslib.h"
#include "platform.hpp"

int jl_setup_aux(uint *ntot, ulong **gids, uint *nnz, uint **ia, uint **ja,
                 double **a, elliptic_t *elliptic, elliptic_t *ellipticf);

int jl_setup(MPI_Comm comm, uint n, const ulong *id, uint nnz,
             const uint *Ai, const uint *Aj, const double *A, uint null,
             uint verbose);

int jl_solve(occa::memory o_x, occa::memory o_rhs);

int jl_solve(float *x, float *rhs);

int jl_free();

// Internal API: DON'T USE !!
int xxt_setup(MPI_Comm comm, uint n, const ulong *id, uint nnz, const uint *Ai,
              const uint *Aj, const double *A, uint null, uint verbose);

int xxt_solve(double *h_x, double *h_b);

int xxt_free();

#endif
