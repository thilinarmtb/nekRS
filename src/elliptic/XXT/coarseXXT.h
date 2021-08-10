#if !defined(_NEKRS_ELLIPTIC_XXT_H_)
#define _NEKRS_ELLIPTIC_XXT_H_

#include "elliptic.h"
#include "gslib.h"

int setup_h1_crs(uint *ntot, ulong **gids_, uint *nnz, uint **ia_, uint **ja_,
                 double **a_, elliptic_t *elliptic, elliptic_t *ellipticf,
                 dfloat lambda);

void xxt_setup(parAlmond::solver_t* M,
               uint n,
               const ulong *id, 
               uint nnz,
               const uint* Ai,
               const uint* Aj,
               const double* A,
               uint null_space);

void xxt_solve(double *x, double *rhs);

void xxt_free();

#endif
