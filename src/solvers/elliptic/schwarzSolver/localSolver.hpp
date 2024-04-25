#if !defined(__LOCAL_SOLVER_HPP__)
#define __LOCAL_SOLVER_HPP__

#include "nrssys.hpp"

#include <vector>

#include "gslib.h"

enum class Algorithm_t { Gemv, Xxt, Cholmod };

template <typename val_t> class AlgorithmInterface_t {
  using vec_t = std::vector<val_t>;

public:
  virtual void Setup(const unsigned num_rows, unsigned *row_offsets,
                     unsigned *col_indices, double *values,
                     const std::string &backend, const int device_id) = 0;

  virtual void Solve(vec_t &x, const vec_t &rhs) = 0;

  virtual ~AlgorithmInterface_t() = default;
};

template <typename val_t> class LocalSolver_t {
  using vec_t = std::vector<val_t>;

public:
  LocalSolver_t();

  void Setup(const unsigned input_size, const hlong *vtx, const unsigned nnz,
             const unsigned *ia, const unsigned *ja, const double *va,
             const Algorithm_t algorithm, const std::string &backend,
             const int device_id);

  void Solve(vec_t &x, const vec_t &rhs);

  ~LocalSolver_t();

private:
  void SetupUserToCompressMap(const hlong *vtx, buffer *bfr);

  void SetupSolver(const hlong *vtx, const unsigned nnz, const unsigned *ia,
                   const unsigned *ja, const double *va,
                   const Algorithm_t algorithm, const std::string &backend,
                   const int device_id, buffer *bfr);

private:
  unsigned                     input_size, compressed_size;
  vec_t                        x, rhs;
  sint                        *u_to_c;
  AlgorithmInterface_t<val_t> *solver;
};

template class LocalSolver_t<float>;
template class LocalSolver_t<double>;

#endif // __LOCAL_SOLVER_HPP__
