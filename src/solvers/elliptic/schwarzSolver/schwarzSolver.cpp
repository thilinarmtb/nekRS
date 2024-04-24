#include "nekInterfaceAdapter.hpp"

#include "schwarzSolverImpl.hpp"

template <typename val_t> class SchwarzSolverImpl_t;

template <typename val_t> SchwarzSolver_t<val_t>::SchwarzSolver_t() {
  nek::box_crs_setup();

  const size_t un  = nekData.nelv;
  const size_t ncr = nekData.schwz_ncr;
  const size_t sn  = nekData.schwz_ne * ncr;

  solver = new SchwarzSolverImpl_t<val_t>(un, sn, ncr);

  MPI_Comm    comm      = platform->comm.mpiComm;
  std::string backend   = platform->device.mode();
  int         device_id = platform->device.id();

  solver->Setup(
      (const long long *)nekData.schwz_vtx, (const double *)nekData.schwz_xyz,
      (const double *)nekData.schwz_amat, (const double *)nekData.schwz_mask,
      (const int *)nekData.schwz_frontier, nekData.schwz_ne, comm, backend,
      device_id);
}

template <typename val_t>
void SchwarzSolver_t<val_t>::Solve(occa::memory       &o_x,
                                   const occa::memory &o_rhs) {
  solver->Solve(o_x, o_rhs);
}

template <typename val_t> SchwarzSolver_t<val_t>::~SchwarzSolver_t() {
  delete solver;
}
