#if !defined(__GMG_SETUP_HPP__)
#define __GMG_SETUP_HPP__

#include <vector>

#include "elliptic.h"

using Long_t   = std::vector<hlong>;
using DLong_t  = std::vector<dlong>;
using UInt_t   = std::vector<unsigned>;
using Double_t = std::vector<double>;
using DFloat_t = std::vector<dfloat>;

void setupCoarseSystem(Long_t &gIds, UInt_t &Ai, UInt_t &Aj, Double_t &Av,
                       elliptic_t *const ecrs, elliptic_t *const efine);

void setupOverlappedSystem(unsigned *nei, long long *eids, unsigned nv,
                           long long *vids, double *xyz, double *mask,
                           double *mat, int *frontier, unsigned nw, int *wids,
                           MPI_Comm comm, unsigned max_ne);

#endif // __GMG_SETUP_HPP__
