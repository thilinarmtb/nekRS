#if !defined(__SCHWARZ_SOLVER_HPP__)
#define __SCHWARZ_SOLVER_HPP__

#include <mpi.h>
#include <string>

#include "nrssys.hpp"
#include "occa.hpp"

class SchwarzSolverInterface_t;

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
