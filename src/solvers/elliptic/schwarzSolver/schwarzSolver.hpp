#if !defined(__SCHWARZ_SOLVE_HPP__)
#define __SCHWARZ_SOLVE_HPP__

#include <mpi.h>
#include <string>

#include "occa.hpp"

template <typename> class SchwarzSolverImpl_t;

class SchwarzSolverInterface_t {
public:
  virtual void Setup(const long long *vtx, const double *xyz,
                     const double *amat, const double *mask,
                     const int *frontier, const int num_elements,
                     const double tol, const MPI_Comm comm,
                     const std::string &backend, const int device_id) = 0;

  virtual void Solve(occa::memory &o_x, const occa::memory &o_rhs) = 0;

  virtual ~SchwarzSolverInterface_t() = default;
};

class SchwarzSolver_t {
public:
  SchwarzSolver_t(const unsigned num_dofs, const unsigned long long *vertices,
                  const unsigned nnz, const unsigned *Ai, const unsigned *Aj,
                  const double *A, const unsigned null_space, const int usefp32,
                  const int device_id, const MPI_Comm comm,
                  const std::string &backend);
  void Solve(occa::memory &o_x, const occa::memory &o_rhs);

  ~SchwarzSolver_t();

private:
  SchwarzSolverInterface_t *solver;
};

#endif
