#include "nekInterfaceAdapter.hpp"

#include "schwarzSolverImpl.hpp"

template <typename val_t> class SchwarzSolverImpl_t;

SchwarzSolver_t::SchwarzSolver_t(const unsigned            num_dofs,
                                 const unsigned long long *vertices,
                                 const unsigned nnz, const unsigned *Ai,
                                 const unsigned *Aj, const double *A,
                                 const unsigned null_space, const int usefp32,
                                 const int device_id, const MPI_Comm comm,
                                 const std::string &backend) {
  nek::box_crs_setup();
  const size_t un = num_dofs;
  const size_t nc = nnz / num_dofs;
  const size_t sn = (*nekData.schwz_ne) * nc;

  if (usefp32)
    solver = new SchwarzSolverImpl_t<float>(un, sn, nc);
  else
    solver = new SchwarzSolverImpl_t<double>(un, sn, nc);

  solver->Setup(
      (const long long *)nekData.schwz_vtx, (const double *)nekData.schwz_xyz,
      (const double *)nekData.schwz_amat, (const double *)nekData.schwz_mask,
      (const int *)nekData.schwz_frontier, *(nekData.schwz_ne), 1e-12, comm,
      backend, device_id);
}

void SchwarzSolver_t::Solve(occa::memory &o_x, const occa::memory &o_rhs) {
  solver->Solve(o_x, o_rhs);
}

SchwarzSolver_t::~SchwarzSolver_t() { delete solver; }
