#if !defined(__LOCAL_SOLVER_HPP__)
#define __LOCAL_SOLVER_HPP__

#include "gslib.h"

enum class Algorithm_t { Gemv, Xxt, Cholmod };

template <typename val_t> class AlgorithmInterface_t {
public:
  virtual void Setup(const uint num_rows, uint *row_offsets, uint *col_indices,
                     double *values, const std::string &backend,
                     const int device_id) = 0;

  virtual void Solve(val_t *x, const val_t *rhs) = 0;

  virtual ~AlgorithmInterface_t() = default;
};

template <typename val_t> class LocalSolver_t {
public:
  LocalSolver_t();

  void Setup(const uint input_size, const slong *vtx, const uint nnz,
             const uint *ia, const uint *ja, const double *va, const double tol,
             const Algorithm_t algorithm, const std::string &backend,
             const int device_id);

  void Solve(val_t *x, const val_t *rhs);

  ~LocalSolver_t();

private:
  void SetupUserToCompressMap(const slong *vtx);

  void SetupCSRMatrix(const slong *vtx, const uint nnz, const uint *ia,
                      const uint *ja, const double *va, const double tol);

  void SetupAlgorithm(const Algorithm_t algorithm, const std::string &backend,
                      const int device_id);

private:
  uint                         input_size, compressed_size, num_rows;
  uint                        *row_offsets, *col_indices;
  double                      *values;
  sint                        *u_to_c;
  buffer                       bfr;
  AlgorithmInterface_t<val_t> *solver;
};

template class LocalSolver_t<float>;
template class LocalSolver_t<double>;

#endif // __LOCAL_SOLVER_HPP__
