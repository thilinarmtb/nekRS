#if !defined(_NEKRS_JL_H_)
#define _NEKRS_JL_H_

#include "gslib.h"
#include "elliptic.h"

int jl_setup_aux(uint *ntot, ulong **gids_, uint *nnz, uint **ia_, uint **ja_,
                 double **a_, elliptic_t *elliptic, elliptic_t *ellipticf,
                 dfloat lambda);

int jl_setup(uint type, parAlmond::solver_t* M, uint n, const ulong *id,
             uint nnz, const uint* Ai, const uint* Aj, const double* A,
             uint null, uint verbose);

int jl_solve(occa::memory o_x, occa::memory o_rhs);

int jl_free();

#endif
