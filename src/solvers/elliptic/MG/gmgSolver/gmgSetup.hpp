#if !defined(__GMG_SETUP_HPP__)
#define __GMG_SETUP_HPP__

#include <vector>

#include "gmgTypes.hpp"

#include "elliptic.h"

void setupCoarseSystem(VecLong_t &gIds, VecUInt_t &Ai, VecUInt_t &Aj,
                       VecDouble_t &Av, elliptic_t *const ecrs,
                       elliptic_t *const efine);

void setupOverlappedSystem(unsigned *nei, long long *eids, unsigned nv,
                           long long *vids, double *xyz, double *mask,
                           double *mat, int *frontier, unsigned nw, int *wids,
                           MPI_Comm comm, unsigned max_ne);

#endif // __GMG_SETUP_HPP__
