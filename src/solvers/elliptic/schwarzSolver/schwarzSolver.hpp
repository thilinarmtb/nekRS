#if !defined(__SCHWARZ_SOLVER_HPP__)
#define __SCHWARZ_SOLVER_HPP__

#include <mpi.h>
#include <string>

#include "nrssys.hpp"
#include "occa.hpp"

class SchwarzSolverInterface_t {
public:
  virtual void Setup(const hlong *vtx, const double *xyz, const double *amat,
                     const double *mask, const int *frontier,
                     const int num_elements, const MPI_Comm comm,
                     const std::string &backend, const int device_id) = 0;

  virtual void Solve(occa::memory &o_x, const occa::memory &o_rhs) = 0;

  virtual ~SchwarzSolverInterface_t() = default;
};

template <typename val_t> class SchwarzSolver_t {
public:
  SchwarzSolver_t();

  void Solve(occa::memory &o_x, const occa::memory &o_rhs);

  ~SchwarzSolver_t();

private:
  SchwarzSolverInterface_t *solver;
};

template class SchwarzSolver_t<float>;
template class SchwarzSolver_t<double>;

#endif // __SCHWARZ_SOLVER_HPP__
