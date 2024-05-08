#if !defined(__GMG_LOCAL_SOLVER_HPP__)
#define __GMG_LOCAL_SOLVER_HPP__

#include "gmgTypes.hpp"

#include "gslib.h"

enum class GMGAlgorithm_t : std::int8_t { Gemv = 0, Xxt, Cholmod };

template <typename val_t> class GMGLocalSolverInterface_t {
  using Vec_t = std::vector<val_t>;

public:
  virtual void Setup(const VecUInt_t &row_offsets, const VecUInt_t &col_indices,
                     const std::vector<double> &values,
                     const std::string &backend, const int device_id) = 0;

  virtual void Solve(Vec_t &x, const Vec_t &rhs) = 0;

  virtual ~GMGLocalSolverInterface_t() = default;
};

template <typename val_t> class GMGLocalSolver_t {
  using Vec_t = std::vector<val_t>;

public:
  GMGLocalSolver_t(const VecLong_t &vtx, const VecUInt_t &ia,
                   const VecUInt_t &ja, const VecDouble_t &va,
                   const GMGAlgorithm_t &algorithm, const std::string &backend,
                   const int device_id);

  void Solve(Vec_t &x, const Vec_t &rhs);

  ~GMGLocalSolver_t();

private:
  void SetupUserToCompressMap(const VecLong_t &vtx, buffer *bfr);

  void SetupSolver(const VecUInt_t &ia, const VecUInt_t &ja,
                   const VecDouble_t &va, const GMGAlgorithm_t &algorithm,
                   const std::string &backend, const int device_id,
                   buffer *bfr);

private:
  unsigned                          input_size, compressed_size;
  Vec_t                             x, rhs;
  VecInt_t                          u_to_c;
  GMGLocalSolverInterface_t<val_t> *solver;
};

template class GMGLocalSolver_t<float>;
template class GMGLocalSolver_t<double>;

#endif // __GMG_LOCAL_SOLVER_HPP__
