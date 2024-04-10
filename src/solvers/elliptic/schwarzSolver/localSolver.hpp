#if !defined(__LOCAL_SOLVER_HPP__)
#define __LOCAL_SOLVER_HPP__

#include <cstdlib>
#include <string>

#include "gslib.h"

enum class Algorithm_t { Gemv, Xxt, Cholmod };

class AlgorithmInterface_t {
public:
  virtual void Setup(const uint num_rows, uint *row_offsets, uint *col_indices,
                     double *values, const gs_dom dom,
                     const std::string &backend, const int device_id) = 0;

  virtual void Solve(void *x, const void *rhs) = 0;

  virtual ~AlgorithmInterface_t() = default;
};

class LocalSolver_t {
public:
  LocalSolver_t();

  void Setup(const uint input_size, const slong *vtx, const uint nnz,
             const uint *ia, const uint *ja, const double *va, const double tol,
             const gs_dom dom, const Algorithm_t algorithm,
             const std::string &backend, const int device_id);

  void Solve(void *x, const void *rhs);

  ~LocalSolver_t();

private:
  void SetupUserToCompressMap(const slong *vtx);

  void SetupCSRMatrix(const slong *vtx, const uint nnz, const uint *ia,
                      const uint *ja, const double *va, const double tol);

  void SetupAlgorithm(const Algorithm_t algorithm, const gs_dom dom,
                      const std::string &backend, const int device_id);

private:
  uint                  input_size, compressed_size, num_rows;
  uint                 *row_offsets, *col_indices;
  double               *values;
  sint                 *u_to_c;
  buffer                bfr;
  AlgorithmInterface_t *solver;
};

#endif // __LOCAL_SOLVER_HPP__
