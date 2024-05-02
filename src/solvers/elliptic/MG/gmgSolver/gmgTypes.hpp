#if !defined(__GMG_TYPES_HPP__)
#define __GMG_TYPES_HPP__

#include <vector>

#include "nrssys.hpp"

using Int_t  = std::vector<int>;
using UInt_t = std::vector<unsigned>;

using Idx_t = std::vector<size_t>;

using Long_t  = std::vector<hlong>;
using DLong_t = std::vector<dlong>;

using Double_t = std::vector<double>;
using DFloat_t = std::vector<dfloat>;

#endif // __GMG_TYPES_HPP__
