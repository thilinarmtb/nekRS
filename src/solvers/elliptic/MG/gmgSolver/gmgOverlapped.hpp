#if !defined(__GMG_OVERLAPPED_HPP__)
#define __GMG_OVERLAPPED_HPP__

#include "gmgTypes.hpp"

void gmgSetupOverlappedSystem(unsigned *nei, long long *eids, unsigned nv,
                              long long *vids, double *xyz, double *mask,
                              double *mat, int *frontier, unsigned nw,
                              int *wids, MPI_Comm comm, unsigned max_ne);

#endif // __GMG_OVERLAPPED_HPP__
