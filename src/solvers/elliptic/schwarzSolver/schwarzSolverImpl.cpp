#include <cassert>

#include "nekInterfaceAdapter.hpp"

#include "schwarzSolverImpl.hpp"

template <typename val_t>
void SchwarzSolverImpl_t<val_t>::SetupCoarseAverage(const Long_t  &vtx,
                                                    const MPI_Comm comm) {
  struct comm c;
  comm_init(&c, comm);

  Long_t vtx_(shared_size);
  for (size_t i = 0; i < user_size; i++) vtx_[i] = vtx[i];
  for (size_t i = user_size; i < shared_size; i++) vtx_[i] = -vtx[i];
  gsh = gs_setup(vtx_.data(), shared_size, &c, 0, gs_auto, 0);

  for (size_t i = 0; i < user_size; i++) inv_mul[i] = 1.0;
  gs(inv_mul.data(), dom, gs_add, 0, gsh, &bfr);
  for (size_t i = 0; i < shared_size; i++) inv_mul[i] = 1.0 / inv_mul[i];

  comm_free(&c);
}

template <typename val_t>
void SchwarzSolverImpl_t<val_t>::SetupLocalSolver(const Long_t      &vtx,
                                                  const Double_t    &va,
                                                  const std::string &backend,
                                                  const int device_id) {
  // FIXME: The following should be part of the input.
  const size_t nnz = shared_size * crs_size;
  Idx_t        ia(nnz);
  Idx_t        ja(nnz);
  const size_t ne = shared_size / crs_size;
  for (size_t e = 0; e < ne; e++) {
    for (size_t j = 0; j < crs_size; j++) {
      for (size_t i = 0; i < crs_size; i++) {
        ia[e * crs_size * crs_size + j * crs_size + i] = e * crs_size + i;
        ja[e * crs_size * crs_size + j * crs_size + i] = e * crs_size + j;
      }
    }
  }
  local_solver = new LocalSolver_t<val_t>{};
  local_solver->Setup(shared_size, vtx, nnz, ia, ja, va, Algorithm_t::Gemv,
                      backend, device_id);
}

template <typename val_t>
void SchwarzSolverImpl_t<val_t>::CoarseAverage(Vec_t &vec) {
  gs(vec.data(), dom, gs_add, 0, gsh, &bfr);
  for (size_t i = 0; i < shared_size; i++) rhs[i] *= inv_mul[i];
}

template <typename val_t>
void SchwarzSolverImpl_t<val_t>::Setup(
    const Long_t &vtx, const Double_t &amat, const Double_t &mask,
    const Int_t &frontier, const size_t num_elements, const MPI_Comm comm,
    const std::string &backend, const int device_id) {

  Long_t vtx_ll(shared_size);
  double maskm = std::numeric_limits<double>::max();
  for (size_t i = 0; i < shared_size; i++) {
    const double maski = (frontier[i] == 1) ? 0 : mask[i];
    vtx_ll[i]          = (maski < 0.1) ? 0 : vtx[i];
    if (maski < maskm) maskm = maski;
  }

  // Sanity check:
  const size_t null_space = (maskm < 1e-10) ? 0 : 1;
  assert(null_space == 0);

  // Setup local Schwarz solver.
  SetupLocalSolver(vtx_ll, amat, backend, device_id);

  // Setup the gather-scatter handle for coarse average.
  SetupCoarseAverage(vtx_ll, comm);

  // Setup the A matrix:
  const size_t nnz = shared_size * crs_size;
  A.reserve(nnz);
  for (size_t i = 0; i < nnz; i++) A[i] = amat[i];
}

template <typename val_t>
void SchwarzSolverImpl_t<val_t>::Solve(occa::memory       &o_x,
                                       const occa::memory &o_rhs) {
  const size_t size = user_size * sizeof(val_t);
  o_rhs.copyTo(rhs.data(), size, 0);

  CoarseAverage(rhs);

  local_solver->Solve(x, rhs);

  CoarseAverage(x);

  // Multiplicative is the default and the only way.
  const int N = crs_size;
  for (size_t e = 0; e < user_size; e++) {
    for (size_t c = 0; c < N; c++) {
      for (size_t k = 0; k < N; k++)
        rhs[k + N * e] -= x[c + N * e] * A[k + c * N + N * N * e];
    }
  }

  for (size_t i = 0; i < user_size; i++) nekData.box_r[i] = rhs[i];

  nek::box_map_vtx_to_box();
  nek::box_crs_solve();
  nek::box_map_box_to_vtx();

  for (size_t i = 0; i < user_size; i++) x[i] += nekData.box_e[i];

  CoarseAverage(x);

  o_x.copyFrom(x.data(), size, 0);
}

template <typename val_t>
SchwarzSolverImpl_t<val_t>::SchwarzSolverImpl_t(const size_t user_size_,
                                                const size_t shared_size_,
                                                const size_t crs_size_) {
  user_size   = user_size_;
  shared_size = shared_size_;
  crs_size    = crs_size_;

  x.reserve(shared_size), rhs.reserve(shared_size);
  inv_mul.reserve(shared_size);

  if (sizeof(val_t) == sizeof(double)) dom = gs_double;
  if (sizeof(val_t) == sizeof(float)) dom = gs_float;

  buffer_init(&bfr, 1024);
  gsh          = nullptr;
  local_solver = nullptr;
}

template <typename val_t> SchwarzSolverImpl_t<val_t>::~SchwarzSolverImpl_t() {
  buffer_free(&bfr);
  if (gsh) gs_free(gsh);
  delete local_solver;
}
