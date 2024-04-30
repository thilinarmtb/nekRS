#include <cassert>
#include <cmath>
#include <cstdlib>

#include "algorithmGemv.hpp"

#include "localSolver.hpp"

template <typename val_t>
void LocalSolver_t<val_t>::SetupSolver(const Long_t &vtx, const Idx_t &ia,
                                       const Idx_t &ja, const Double_t &va,
                                       const Algorithm_t  algorithm,
                                       const std::string &backend,
                                       const int device_id, buffer *bfr) {
  typedef struct {
    uint   r, c;
    double v;
  } entry_t;

  const size_t nnz = ia.size();
  struct array entries;
  {
    array_init(entry_t, &entries, nnz);

    entry_t eij;
    for (unsigned z = 0; z < nnz; z++) {
      int i = u_to_c[ia[z]], j = u_to_c[ja[z]];
      if (i < 0 || j < 0) continue;
      eij.r = i, eij.c = j, eij.v = va[z];
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
  Idx_t    row_offsets(compressed_size + 1);
  Idx_t    col_indices(assembled_entries.n);
  Double_t values(assembled_entries.n);

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
  case Algorithm_t::Gemv: solver = new AlgorithmGemv_t<val_t>{}; break;
  case Algorithm_t::Xxt: break;
  case Algorithm_t::Cholmod: break;
  default: break;
  }

  solver->Setup(row_offsets, col_indices, values, backend, device_id);
}

template <typename val_t>
void LocalSolver_t<val_t>::SetupUserToCompressMap(const Long_t &vtx,
                                                  buffer       *bfr) {
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
      vid.id = vtx[i], vid.idx = i;
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
  {
    x.reserve(compressed_size);
    rhs.reserve(compressed_size);
    u_to_c.reserve(input_size);
  }

  {
    sarray_sort(vertex_id_t, vids.ptr, vids.n, idx, 0, bfr);
    const vertex_id_t *pv = (const vertex_id_t *)vids.ptr;
    for (unsigned i = 0; i < input_size; i++) u_to_c[i] = pv[i].perm;
  }

  array_free(&vids);
}

template <typename val_t>
void LocalSolver_t<val_t>::Setup(const Long_t &vtx, const Idx_t &ia,
                                 const Idx_t &ja, const Double_t &va,
                                 const Algorithm_t  algorithm,
                                 const std::string &backend,
                                 const int          device_id) {
  input_size = vtx.size();

  // Sanity checks:
  assert(ia.size() == ja.size() && ia.size() == va.size());

  buffer bfr;
  buffer_init(&bfr, 1024);

  SetupUserToCompressMap(vtx, &bfr);

  SetupSolver(vtx, ia, ja, va, algorithm, backend, device_id, &bfr);

  buffer_free(&bfr);
}

template <typename val_t>
void LocalSolver_t<val_t>::Solve(Vec_t &x_, const Vec_t &rhs_) {
  for (unsigned i = 0; i < compressed_size; i++) rhs[i] = 0;
  for (unsigned i = 0; i < input_size; i++)
    if (u_to_c[i] >= 0) rhs[u_to_c[i]] += rhs_[i];

  if (solver != nullptr) solver->Solve(x, rhs);

  for (unsigned i = 0; i < input_size; i++) {
    if (u_to_c[i] >= 0)
      x_[i] = x[u_to_c[i]];
    else
      x_[i] = 0;
  }
}

template <typename val_t> LocalSolver_t<val_t>::LocalSolver_t() {
  input_size      = 0;
  compressed_size = 0;
  solver          = nullptr;
  u_to_c          = Int_t();
}

template <typename val_t> LocalSolver_t<val_t>::~LocalSolver_t() {
  delete solver;
}
