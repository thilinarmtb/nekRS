#if !defined(__LOCAL_SOLVER_HPP__)
#define __LOCAL_SOLVER_HPP__

#include "gslib.h"

#include <vector>

enum class Algorithm_t { Gemv, Xxt, Cholmod };

template <typename val_t> class AlgorithmInterface_t {
  using vec_t = std::vector<val_t>;

public:
  virtual void Setup(const uint num_rows, uint *row_offsets, uint *col_indices,
                     double *values, const std::string &backend,
                     const int device_id) = 0;

  virtual void Solve(vec_t &x, const vec_t &rhs) = 0;

  virtual ~AlgorithmInterface_t() = default;
};

template <typename val_t> class LocalSolver_t {
  using vec_t = std::vector<val_t>;

public:
  LocalSolver_t();

  void Setup(const uint input_size, const slong *vtx, const uint nnz,
             const uint *ia, const uint *ja, const double *va,
             const Algorithm_t algorithm, const std::string &backend,
             const int device_id);

  void Solve(vec_t &x, const vec_t &rhs);

  ~LocalSolver_t();

private:
  void SetupUserToCompressMap(const slong *vtx, buffer *bfr);

  void SetupSolver(const slong *vtx, const uint nnz, const uint *ia,
                   const uint *ja, const double *va,
                   const Algorithm_t algorithm, const std::string &backend,
                   const int device_id, buffer *bfr);

private:
  uint                         input_size, compressed_size;
  vec_t                        x, rhs;
  sint                        *u_to_c;
  AlgorithmInterface_t<val_t> *solver;
};

template class LocalSolver_t<float>;
template class LocalSolver_t<double>;

#endif // __LOCAL_SOLVER_HPP__
