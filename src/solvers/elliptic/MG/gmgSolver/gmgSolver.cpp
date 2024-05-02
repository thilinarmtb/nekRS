#include <cassert>

#include "nekInterfaceAdapter.hpp"

#include "gmgSolver.hpp"

template <typename val_t>
void GMGSolver_t<val_t>::SetupCoarseAverage(const VecLong_t &vtx,
                                            const MPI_Comm   comm) {
  struct comm c;
  comm_init(&c, comm);

  VecLong_t vtx_(shared_size);
  for (size_t i = 0; i < user_size; i++) vtx_[i] = vtx[i];
  for (size_t i = user_size; i < shared_size; i++) vtx_[i] = -vtx[i];
  gsh = gs_setup(vtx_.data(), shared_size, &c, 0, gs_auto, 0);

  for (size_t i = 0; i < user_size; i++) inv_mul[i] = 1.0;
  gs(inv_mul.data(), dom, gs_add, 0, gsh, &bfr);
  for (size_t i = 0; i < shared_size; i++) inv_mul[i] = 1.0 / inv_mul[i];

  comm_free(&c);
}

template <typename val_t>
void GMGSolver_t<val_t>::SetupLocalSolver(const VecLong_t   &vtx,
                                          const VecDouble_t &va,
                                          const Algorithm_t &algo,
                                          const std::string &backend,
                                          const int          device_id) {
  // FIXME: The following should be part of the input.
  const size_t nnz = shared_size * crs_size;
  VecIdx_t     ia(nnz);
  VecIdx_t     ja(nnz);
  const size_t num_elements = shared_size / crs_size;
  for (size_t e = 0; e < num_elements; e++) {
    for (size_t j = 0; j < crs_size; j++) {
      for (size_t i = 0; i < crs_size; i++) {
        ia[e * crs_size * crs_size + j * crs_size + i] = e * crs_size + i;
        ja[e * crs_size * crs_size + j * crs_size + i] = e * crs_size + j;
      }
    }
  }

  solver = new LocalSolver_t<val_t>{};
  solver->Setup(vtx, ia, ja, va, algo, backend, device_id);
}

template <typename val_t> void GMGSolver_t<val_t>::CoarseAverage(Vec_t &vec) {
  gs(vec.data(), dom, gs_add, 0, gsh, &bfr);
  for (size_t i = 0; i < shared_size; i++) rhs[i] *= inv_mul[i];
}

template <typename val_t>
void GMGSolver_t<val_t>::Setup(const VecLong_t &vtx, const VecDouble_t &amat,
                               const VecDouble_t &mask,
                               const VecInt_t    &frontier,
                               const Algorithm_t &algo) {
  VecLong_t vtx_ll(shared_size);
  double    maskm = std::numeric_limits<double>::max();
  for (size_t i = 0; i < shared_size; i++) {
    const double maski = (frontier[i] == 1) ? 0 : mask[i];
    vtx_ll[i]          = (maski < 0.1) ? 0 : vtx[i];
    if (maski < maskm) maskm = maski;
  }

  // Sanity check:
  const size_t null_space = (maskm < 1e-10) ? 0 : 1;
  assert(null_space == 0);

  // Setup local Schwarz solver.
  const auto backend   = platform->device.mode();
  const auto device_id = platform->device.id();
  SetupLocalSolver(vtx_ll, amat, algo, backend, device_id);

  // Setup the gather-scatter handle for coarse average.
  const auto comm = platform->comm.mpiComm;
  SetupCoarseAverage(vtx_ll, comm);

  // Setup the A matrix:
  const size_t nnz = shared_size * crs_size;
  A.reserve(nnz);
  for (size_t i = 0; i < nnz; i++) A[i] = amat[i];
}

template <typename val_t>
void GMGSolver_t<val_t>::Solve(occa::memory &o_x, const occa::memory &o_rhs) {
  const size_t size = user_size * sizeof(val_t);
  o_rhs.copyTo(rhs.data(), size, 0);

  CoarseAverage(rhs);

  solver->Solve(x, rhs);

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

template <typename val_t> GMGSolver_t<val_t>::GMGSolver_t() {
  nek::box_crs_setup();

  crs_size    = nekData.schwz_ncr;
  user_size   = nekData.nelv * crs_size;
  shared_size = nekData.schwz_ne * crs_size;

  const hlong *p_vtx = (const hlong *)nekData.schwz_vtx;
  VecLong_t    vtx(p_vtx, p_vtx + shared_size);

  const double *p_mask = (const double *)nekData.schwz_mask;
  VecDouble_t   mask(p_mask, p_mask + shared_size);

  const int *p_frontier = (const int *)nekData.schwz_frontier;
  VecInt_t   frontier(p_frontier, p_frontier + shared_size);

  const double *p_amat = (const double *)nekData.schwz_amat;
  const size_t  nnz    = shared_size * crs_size;
  VecDouble_t   amat(p_amat, p_amat + nnz);

  Setup(vtx, amat, mask, frontier, Algorithm_t::Gemv);

  if (sizeof(val_t) == sizeof(double)) dom = gs_double;
  if (sizeof(val_t) == sizeof(float)) dom = gs_float;

  x.reserve(shared_size);
  rhs.reserve(shared_size);
  inv_mul.reserve(shared_size);

  buffer_init(&bfr, 1024);
  gsh    = nullptr;
  solver = nullptr;
}

template <typename val_t> GMGSolver_t<val_t>::~GMGSolver_t() {
  buffer_free(&bfr);
  if (gsh) gs_free(gsh);
  if (solver) delete solver;
}
