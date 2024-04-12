#if !defined(__SCHWARZ_SOLVE_IMPL_HPP__)
#define __SCHWARZ_SOLVE_IMPL_HPP__

#include "localSolver.hpp"

#include "schwarzSolver.hpp"

template <typename val_t>
class SchwarzSolverImpl_t : public SchwarzSolverInterface_t {
public:
  SchwarzSolverImpl_t(const size_t user_size, const size_t shared_size,
                      const size_t crs_size);

  void Setup(const long long *vtx, const double *xyz, const double *amat,
             const double *mask, const int *frontier, const int num_elements,
             const double tol, const MPI_Comm comm, const std::string &backend,
             const int device_id) override;

  void Solve(occa::memory &o_x, const occa::memory &o_rhs) override;

  ~SchwarzSolverImpl_t();

private:
  void SetupLocalSolver(const slong *vtx, const double *va, const double tol,
                        const std::string &backend, const int device_id);

  void SetupCoarseAverage(const slong *vtx, const MPI_Comm comm);

  void SetupCoarseMatrix(const double *A);

  void CoarseAverage(void *vector);

private:
  size_t          user_size, shared_size, crs_size;
  gs_dom          dom;
  val_t          *A, *x, *rhs, *inv_mul;
  buffer          bfr;
  struct gs_data *gsh;
  LocalSolver_t<val_t>  *local_solver;
};

#endif // __SCHWARZ_SOLVE_IMPL_HPP__
