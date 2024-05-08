#include <cassert>
#include <cmath>
#include <cstdlib>

#include "gmgGemv.hpp"

#include "gmgLocalSolver.hpp"

template <typename val_t>
void GMGLocalSolver_t<val_t>::SetupSolver(const VecUInt_t      &Ai,
                                          const VecUInt_t      &Aj,
                                          const VecDouble_t    &Av,
                                          const GMGAlgorithm_t &algorithm,
                                          const std::string    &backend,
                                          const int device_id, buffer *bfr) {
  typedef struct {
    uint   r, c;
    double v;
  } entry_t;

  const size_t nnz = Ai.size();
  struct array entries;
  {
    array_init(entry_t, &entries, nnz);

    entry_t eij;
    for (unsigned z = 0; z < nnz; z++) {
      int i = u_to_c[Ai[z]], j = u_to_c[Aj[z]];
      if (i < 0 || j < 0) continue;
      eij.r = i, eij.c = j, eij.v = Av[z];
      array_cat(entry_t, &entries, &eij, 1);
    }

    if (entries.n == 0) return;
  }

  // Aseemble the matrix.
  struct array assembled_entries;
  {
    sarray_sort_2(entry_t, entries.ptr, entries.n, r, 0, c, 0, bfr);
    entry_t *pe = (entry_t *)entries.ptr;

    array_init(entry_t, &assembled_entries, nnz);
    unsigned s = 0;
    for (unsigned i = 1; i < entries.n; i++) {
      if ((pe[i].r != pe[s].r) || (pe[i].c != pe[s].c)) {
        array_cat(entry_t, &assembled_entries, &pe[s], 1);
        s = i;
      } else {
        pe[s].v += pe[i].v;
      }
    }
    array_cat(entry_t, &assembled_entries, &pe[s], 1);

    array_free(&entries);
  }

  // Allocate the csr data structures
  VecUInt_t   row_offsets(compressed_size + 1);
  VecUInt_t   col_indices(assembled_entries.n);
  VecDouble_t values(assembled_entries.n);

  // Convert the assembled matrix to a CSR matrix.
  {
    entry_t *pc = (entry_t *)assembled_entries.ptr;

    // Sanity check:
    assert((pc[assembled_entries.n - 1].r + 1) == compressed_size);

    row_offsets[0] = 0;
    unsigned i     = 0, j;
    for (unsigned r = 0; r < compressed_size; r++) {
      j = i;
      while (j < assembled_entries.n && pc[j].r == r)
        col_indices[j] = pc[j].c, values[j] = pc[j].v, j++;
      row_offsets[r + 1] = row_offsets[r] + j - i;
      i                  = j;
    }

    array_free(&assembled_entries);
  }

  switch (algorithm) {
  case GMGAlgorithm_t::Gemv: solver = new GMGGemv_t<val_t>{}; break;
  case GMGAlgorithm_t::Xxt: break;
  case GMGAlgorithm_t::Cholmod: break;
  default: break;
  }

  solver->Setup(row_offsets, col_indices, values, backend, device_id);
}

template <typename val_t>
void GMGLocalSolver_t<val_t>::SetupUserToCompressMap(const VecLong_t &Aids,
                                                     buffer          *bfr) {
  typedef struct {
    ulong id;
    uint  idx;
    int   perm;
  } vertex_id_t;

  struct array vids;
  {
    array_init(vertex_id_t, &vids, input_size);

    vertex_id_t vid;
    for (unsigned i = 0; i < input_size; i++) {
      vid.id = Aids[i], vid.idx = i;
      array_cat(vertex_id_t, &vids, &vid, 1);
    }
  }

  compressed_size = 0;
  {
    sarray_sort(vertex_id_t, vids.ptr, vids.n, id, 1, bfr);

    vertex_id_t *pv = (vertex_id_t *)vids.ptr;
    ulong        id = 0;
    for (unsigned i = 0; i < vids.n; i++) {
      if (pv[i].id != id) id = pv[i].id, compressed_size++;
      pv[i].perm = (int)compressed_size - 1;
    }
  }

  // Reserve sizes for internal vectors.
  x.reserve(compressed_size);
  rhs.reserve(compressed_size);
  u_to_c.reserve(input_size);

  {
    sarray_sort(vertex_id_t, vids.ptr, vids.n, idx, 0, bfr);
    const vertex_id_t *pv = (const vertex_id_t *)vids.ptr;
    for (unsigned i = 0; i < input_size; i++) u_to_c[i] = pv[i].perm;
  }

  array_free(&vids);
}

template <typename val_t>
void GMGLocalSolver_t<val_t>::Solve(Vec_t &x_, const Vec_t &rhs_) {
  for (unsigned i = 0; i < compressed_size; i++) rhs[i] = 0;
  for (unsigned i = 0; i < input_size; i++)
    if (u_to_c[i] >= 0) rhs[u_to_c[i]] += rhs_[i];

  if (solver != nullptr) solver->Solve(x, rhs);

  for (unsigned i = 0; i < input_size; i++) {
    if (u_to_c[i] >= 0) x_[i] = x[u_to_c[i]];
    else x_[i] = 0;
  }
}

template <typename val_t>
GMGLocalSolver_t<val_t>::GMGLocalSolver_t(
    const VecLong_t &Aids, const VecUInt_t &Ai, const VecUInt_t &Aj,
    const VecDouble_t &Av, const GMGAlgorithm_t &algorithm,
    const std::string &backend, const int device_id) {
  input_size = Aids.size();

  buffer bfr;
  buffer_init(&bfr, 1024);

  SetupUserToCompressMap(Aids, &bfr);

  SetupSolver(Ai, Aj, Av, algorithm, backend, device_id, &bfr);

  buffer_free(&bfr);
}

template <typename val_t> GMGLocalSolver_t<val_t>::~GMGLocalSolver_t() {
  delete solver;
}
