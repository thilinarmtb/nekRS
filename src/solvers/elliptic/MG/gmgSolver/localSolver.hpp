#if !defined(__LOCAL_SOLVER_HPP__)
#define __LOCAL_SOLVER_HPP__

#include "nrssys.hpp"

#include "gslib.h"

using Idx_t    = std::vector<size_t>;
using Int_t    = std::vector<int>;
using Double_t = std::vector<double>;
using Long_t   = std::vector<hlong>;

enum class Algorithm_t : std::int8_t { Gemv = 0, Xxt, Cholmod };

template <typename val_t> class AlgorithmInterface_t {
  using Vec_t = std::vector<val_t>;

public:
  virtual void Setup(const Idx_t &row_offsets, const Idx_t &col_indices,
                     const std::vector<double> &values,
                     const std::string &backend, const int device_id) = 0;

  virtual void Solve(Vec_t &x, const Vec_t &rhs) = 0;

  virtual ~AlgorithmInterface_t() = default;
};

template <typename val_t> class LocalSolver_t {
  using Vec_t = std::vector<val_t>;

public:
  LocalSolver_t();

  void Setup(const Long_t &vtx, const Idx_t &ia, const Idx_t &ja,
             const Double_t &va, const Algorithm_t algorithm,
             const std::string &backend, const int device_id);

  void Solve(Vec_t &x, const Vec_t &rhs);

  ~LocalSolver_t();

private:
  void SetupUserToCompressMap(const Long_t &vtx, buffer *bfr);

  void SetupSolver(const Long_t &vtx, const Idx_t &ia, const Idx_t &ja,
                   const Double_t &va, const Algorithm_t algorithm,
                   const std::string &backend, const int device_id,
                   buffer *bfr);

private:
  unsigned                     input_size, compressed_size;
  Vec_t                        x, rhs;
  Int_t                        u_to_c;
  AlgorithmInterface_t<val_t> *solver;
};

template class LocalSolver_t<float>;
template class LocalSolver_t<double>;

#endif // __LOCAL_SOLVER_HPP__
