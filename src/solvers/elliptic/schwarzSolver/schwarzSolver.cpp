#include "nekInterfaceAdapter.hpp"

#include "schwarzSolverImpl.hpp"

template <typename val_t> class SchwarzSolverImpl_t;

template <typename val_t> SchwarzSolver_t<val_t>::SchwarzSolver_t() {
  nek::box_crs_setup();

  const size_t user_size   = nekData.nelv;
  const size_t crs_size    = nekData.schwz_ncr;
  const size_t shared_size = nekData.schwz_ne * crs_size;
  const size_t nnz         = shared_size * crs_size;

  solver = new SchwarzSolverImpl_t<val_t>(user_size, shared_size, crs_size);

  MPI_Comm    comm      = platform->comm.mpiComm;
  std::string backend   = platform->device.mode();
  int         device_id = platform->device.id();

  const hlong *p_vtx = (const hlong *)nekData.schwz_vtx;
  Long_t       vtx(p_vtx, p_vtx + shared_size);

  const double *p_amat = (const double *)nekData.schwz_amat;
  Double_t      amat(p_amat, p_amat + nnz);

  const double *p_mask = (const double *)nekData.schwz_mask;
  Double_t      mask(p_mask, p_mask + shared_size);

  const int *p_frontier = (const int *)nekData.schwz_frontier;
  Int_t      frontier(p_frontier, p_frontier + shared_size);

  solver->Setup(vtx, amat, mask, frontier, nekData.schwz_ne, comm, backend,
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
