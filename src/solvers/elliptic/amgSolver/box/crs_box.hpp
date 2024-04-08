#if !defined(_CRS_BOX_HPP_)
#define _CRS_BOX_HPP_

// Inclusion of occa.hpp here is a workaround for a weird issue. Fix it before
// releasing rhe code.
#include "occa.hpp"
#define OMPI_SKIP_MPICXX 1
#include "gslib.h"

struct box;
struct box *crs_box_setup(uint n, const ulong *id, uint nnz, const uint *Ai,
                          const uint *Aj, const double *A, uint null,
                          const struct comm *comm, const MPI_Comm *inter_comm,
                          gs_dom dom);
void crs_box_solve(void *x, struct box *data, const void *b);
void crs_box_solve_go_gs(occa::memory &o_x, struct box *data,
                         occa::memory &o_rhs);
void crs_box_solve2(occa::memory &o_x, struct box *data, occa::memory &o_rhs);
void crs_box_free(struct box *data);

#endif
