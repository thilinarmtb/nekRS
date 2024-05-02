#if !defined(__LOCAL_SOLVER_HPP__)
#define __LOCAL_SOLVER_HPP__

#include "gmgTypes.hpp"

#include "gslib.h"

enum class Algorithm_t : std::int8_t { Gemv = 0, Xxt, Cholmod };

template <typename val_t> class AlgorithmInterface_t {
  using Vec_t = std::vector<val_t>;

public:
  virtual void Setup(const VecIdx_t &row_offsets, const VecIdx_t &col_indices,
                     const std::vector<double> &values,
                     const std::string &backend, const int device_id) = 0;

  virtual void Solve(Vec_t &x, const Vec_t &rhs) = 0;

  virtual ~AlgorithmInterface_t() = default;
};

template <typename val_t> class LocalSolver_t {
  using Vec_t = std::vector<val_t>;

public:
  LocalSolver_t();

  void Setup(const VecLong_t &vtx, const VecIdx_t &ia, const VecIdx_t &ja,
             const VecDouble_t &va, const Algorithm_t algorithm,
             const std::string &backend, const int device_id);

  void Solve(Vec_t &x, const Vec_t &rhs);

  ~LocalSolver_t();

private:
  void SetupUserToCompressMap(const VecLong_t &vtx, buffer *bfr);

  void SetupSolver(const VecLong_t &vtx, const VecIdx_t &ia, const VecIdx_t &ja,
                   const VecDouble_t &va, const Algorithm_t algorithm,
                   const std::string &backend, const int device_id,
                   buffer *bfr);

private:
  unsigned                     input_size, compressed_size;
  Vec_t                        x, rhs;
  VecInt_t                     u_to_c;
  AlgorithmInterface_t<val_t> *solver;
};

template class LocalSolver_t<float>;
template class LocalSolver_t<double>;

#endif // __LOCAL_SOLVER_HPP__
