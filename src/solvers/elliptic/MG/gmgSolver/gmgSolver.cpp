#include <cassert>

#include "nekInterfaceAdapter.hpp"

#include "gmgSolver.hpp"

#include "gmgOverlapped.hpp"

template <typename val_t>
void GMGSolver_t<val_t>::SetupCoarseAverage(const VecLong_t &vtx,
                                            const MPI_Comm   comm) {
  // Setup gs handle:
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

  // Setup the gs domain:
  if (sizeof(val_t) == sizeof(double)) dom = gs_double;
  if (sizeof(val_t) == sizeof(float)) dom = gs_float;
}

template <typename val_t> void GMGSolver_t<val_t>::CoarseAverage(Vec_t &vec) {
  gs(vec.data(), dom, gs_add, 0, gsh, &bfr);
  for (size_t i = 0; i < shared_size; i++) rhs[i] *= inv_mul[i];
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

template <typename val_t>
GMGSolver_t<val_t>::GMGSolver_t(const VecLong_t &Aids_, const VecUInt_t &Ai_,
                                const VecUInt_t &Aj_, const VecDouble_t &Av_) {
  // Sanity checks:
  assert(Ai_.size() == Aj_.size() && Ai_.size() == Av_.size());

  user_size = Aids_.size();
  crs_size  = Ai_.size() / user_size;

  VecLong_t   Aids = Aids_;
  VecUInt_t   Ai   = Ai_;
  VecUInt_t   Aj   = Aj_;
  VecDouble_t Av   = Av_;

  auto comm = platform->comm.mpiComm;

  VecInt_t frontier(user_size);
  gmgFindOverlappedSystem(Aids, Ai, Ai, Av, frontier, comm);
  shared_size = Aids.size();

  // Setup local solver:
  unsigned maskm = std::numeric_limits<unsigned>::max();
  for (size_t i = 0; i < shared_size; i++) {
    const unsigned mask = (frontier[i] == 1 || Aids[i] == 0) ? 0 : 1;
    Aids[i]             = (double)mask * Aids[i];
    if (mask < maskm) maskm = mask;
  }
  assert(maskm == 0);

  auto algo = GMGAlgorithm_t::Gemv;
  auto mode = platform->device.mode();
  auto id   = platform->device.id();
  solver    = new GMGLocalSolver_t<val_t>(Aids, Ai, Aj, Av, algo, mode, id);

  // Setup the gather-scatter handle for coarse average:
  SetupCoarseAverage(Aids, comm);

  // Copy the unassembled A matrix:
  const size_t nnz = shared_size * crs_size;
  A.reserve(nnz);
  for (size_t i = 0; i < nnz; i++) A[i] = Av[i];

  // Reserve and initialize work arrays:
  x.reserve(shared_size);
  rhs.reserve(shared_size);
  inv_mul.reserve(shared_size);
  buffer_init(&bfr, 1024);
}

template <typename val_t> GMGSolver_t<val_t>::~GMGSolver_t() {
  buffer_free(&bfr);
  if (gsh) gs_free(gsh);
  if (solver) delete solver;
}
