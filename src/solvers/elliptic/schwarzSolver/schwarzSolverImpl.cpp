#include <cassert>

#include "nekInterfaceAdapter.hpp"

#include "schwarzSolverImpl.hpp"

template class SchwarzSolverImpl_t<float>;
template class SchwarzSolverImpl_t<double>;

template <typename val_t>
void SchwarzSolverImpl_t<val_t>::SetupCoarseMatrix(const double *A_) {
  const size_t N = shared_size * crs_size;
  A              = new val_t[N];
  for (uint i = 0; i < N; i++) A[i] = A_[i];
}

template <typename val_t>
void SchwarzSolverImpl_t<val_t>::SetupCoarseAverage(const slong   *vtx,
                                                    const MPI_Comm comm) {
  struct comm c;
  comm_init(&c, comm);

  slong *vtx_ = new slong[shared_size];
  for (uint i = 0; i < user_size; i++) vtx_[i] = vtx[i];
  for (uint i = user_size; i < shared_size; i++) vtx_[i] = -vtx[i];
  gsh = gs_setup(vtx_, shared_size, &c, 0, gs_auto, 0);
  delete[] vtx_;

  for (uint i = 0; i < user_size; i++) inv_mul[i] = 1.0;
  gs(inv_mul, dom, gs_add, 0, gsh, &bfr);
  for (uint i = 0; i < shared_size; i++) inv_mul[i] = 1.0 / inv_mul[i];

  comm_free(&c);
}

template <typename val_t>
void SchwarzSolverImpl_t<val_t>::SetupLocalSolver(const slong       *vtx,
                                                  const double      *va,
                                                  const double       tol,
                                                  const std::string &backend,
                                                  const int device_id) {
  // FIXME: The following should be part of the input.
  const uint nnz = shared_size * crs_size;
  uint      *ia  = new uint[nnz];
  uint      *ja  = new uint[nnz];
  const uint ne  = shared_size / crs_size;
  for (uint e = 0; e < ne; e++) {
    for (uint j = 0; j < crs_size; j++) {
      for (uint i = 0; i < crs_size; i++) {
        ia[e * crs_size * crs_size + j * crs_size + i] = e * crs_size + i;
        ja[e * crs_size * crs_size + j * crs_size + i] = e * crs_size + j;
      }
    }
  }
  local_solver = new LocalSolver_t{};
  local_solver->Setup(shared_size, vtx, nnz, ia, ja, va, tol, dom,
                      Algorithm_t::Gemv, backend, device_id);
  delete[] ia, ja;
}

template <typename val_t>
void SchwarzSolverImpl_t<val_t>::CoarseAverage(void *vector) {
  gs(vector, dom, gs_add, 0, gsh, &bfr);
  for (uint i = 0; i < shared_size; i++) rhs[i] *= inv_mul[i];
}

template <typename val_t>
void SchwarzSolverImpl_t<val_t>::Setup(const long long *vtx, const double *xyz,
                                       const double *va, const double *mask,
                                       const int *frontier,
                                       const int num_elements, const double tol,
                                       const MPI_Comm     comm,
                                       const std::string &backend,
                                       const int          device_id) {
  slong *vtx_ll = new slong[shared_size];
  double maskm  = std::numeric_limits<double>::max();
  for (uint i = 0; i < shared_size; i++) {
    const double maski = (frontier[i] == 1) ? 0 : mask[i];
    vtx_ll[i]          = (maski < 0.1) ? 0 : vtx[i];
    if (maski < maskm) maskm = maski;
  }

  // Sanity check:
  const uint null_space = (maskm < 1e-10) ? 0 : 1;
  assert(null_space == 0);

  // Setup local Schwarz solver.
  SetupLocalSolver(vtx_ll, va, tol, backend, device_id);

  // Setup the gather-scatter handle for coarse average.
  SetupCoarseAverage(vtx_ll, comm);
  delete[] vtx_ll;

  // Copy the matrix.
  SetupCoarseMatrix(va);
}

template <typename val_t>
void SchwarzSolverImpl_t<val_t>::Solve(occa::memory       &o_x,
                                       const occa::memory &o_rhs) {
  const size_t size = user_size * sizeof(val_t);
  o_rhs.copyTo(rhs, size, 0);

  CoarseAverage(rhs);

  local_solver->Solve(x, rhs);

  CoarseAverage(x);

  // Multiplicative is the default and the only way.
  const int N = crs_size;
  for (uint e = 0; e < user_size; e++) {
    for (uint c = 0; c < N; c++) {
      for (uint k = 0; k < N; k++)
        rhs[k + N * e] -= x[c + N * e] * A[k + c * N + N * N * e];
    }
  }

  for (uint i = 0; i < user_size; i++) nekData.box_r[i] = rhs[i];

  nek::box_map_vtx_to_box();
  nek::box_crs_solve();
  nek::box_map_box_to_vtx();

  for (uint i = 0; i < user_size; i++) x[i] += nekData.box_e[i];

  CoarseAverage(x);

  o_x.copyFrom(x, size, 0);
}

template <typename val_t>
SchwarzSolverImpl_t<val_t>::SchwarzSolverImpl_t(const size_t user_size_,
                                                const size_t shared_size_,
                                                const size_t crs_size_) {
  user_size   = user_size_;
  shared_size = shared_size_;
  crs_size    = crs_size_;

  x       = new val_t[shared_size];
  rhs     = new val_t[shared_size];
  inv_mul = new val_t[shared_size];

  if (sizeof(val_t) == sizeof(double)) dom = gs_double;
  if (sizeof(val_t) == sizeof(float)) dom = gs_float;

  buffer_init(&bfr, 1024);
  gsh          = nullptr;
  local_solver = nullptr;
}

template <typename val_t> SchwarzSolverImpl_t<val_t>::~SchwarzSolverImpl_t() {
  delete[] A, x, rhs, inv_mul;
  buffer_free(&bfr);
  if (gsh) gs_free(gsh);
  delete local_solver;
}
