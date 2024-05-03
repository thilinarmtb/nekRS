#if !defined(__GMG_SETUP_HPP__)
#define __GMG_SETUP_HPP__

#include "gmgTypes.hpp"

#include "elliptic.h"

void gmgSetupCoarseSystem(VecLong_t &gIds, VecUInt_t &Ai, VecUInt_t &Aj,
                          VecDouble_t &Av, elliptic_t *const ecrs,
                          elliptic_t *const efine);

#endif // __GMG_SETUP_HPP__
