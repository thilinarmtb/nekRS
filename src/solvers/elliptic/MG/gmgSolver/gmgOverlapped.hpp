#if !defined(__GMG_OVERLAPPED_HPP__)
#define __GMG_OVERLAPPED_HPP__

#include "gmgTypes.hpp"

void gmgFindOverlappedSystem(VecLong_t &Aids, VecUInt_t &Ai, VecUInt_t &Aj,
                             VecDouble_t &Av, VecInt_t &frontier,
                             const MPI_Comm comm);

#endif // __GMG_OVERLAPPED_HPP__
