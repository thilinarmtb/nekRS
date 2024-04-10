#include <math.h>

#include "algorithmGemv.hpp"

#include "localSolver.hpp"

LocalSolver_t::LocalSolver_t() {
  input_size      = 0;
  compressed_size = 0;
  num_rows        = 0;
  row_offsets     = nullptr;
  col_indices     = nullptr;
  values          = nullptr;
  u_to_c          = nullptr;
  buffer_init(&bfr, 1024);
}

void LocalSolver_t::SetupAlgorithm(const Algorithm_t algorithm,
                                   const gs_dom dom, const std::string &backend,
                                   const int device_id) {
  switch (algorithm) {
  case Algorithm_t::Gemv: solver = new AlgorithmGemv_t{}; break;
  case Algorithm_t::Xxt: break;
  case Algorithm_t::Cholmod: break;
  default: break;
  }

  solver->Setup(num_rows, row_offsets, col_indices, values, dom, backend,
                device_id);
}

void LocalSolver_t::SetupCSRMatrix(const slong *vtx, const uint nnz,
                                   const uint *ia, const uint *ja,
                                   const double *va, const double tol) {
  typedef struct {
    uint   r, c;
    double v;
  } entry_t;

  struct array entries;
  {
    array_init(entry_t, &entries, nnz);

    entry_t eij;
    for (uint z = 0; z < nnz; z++) {
      sint i = u_to_c[ia[z]], j = u_to_c[ja[z]];
      if (i < 0 || j < 0 || fabs(va[z]) < tol) continue;
      eij.r = i, eij.c = j, eij.v = va[z];
      array_cat(entry_t, &entries, &eij, 1);
    }
  }

  if (entries.n == 0) return;

  // Aseemble the matrix.
  struct array assembled_entries;
  {
    array_init(entry_t, &assembled_entries, nnz);

    sarray_sort_2(entry_t, entries.ptr, entries.n, r, 0, c, 0, &bfr);

    entry_t *pe = (entry_t *)entries.ptr;
    uint     s  = 0;
    for (uint i = 1; i < entries.n; i++) {
      if ((pe[i].r != pe[s].r) || (pe[i].c != pe[s].c)) {
        array_cat(entry_t, &assembled_entries, &pe[s], 1);
        s = i;
      } else {
        pe[s].v += pe[i].v;
      }
    }
    array_cat(entry_t, &assembled_entries, &pe[s], 1);
  }
  array_free(&entries);

  // Convert the assembled matrix to a CSR matrix.
  {
    entry_t *pc = (entry_t *)assembled_entries.ptr;
    if (assembled_entries.n > 0) num_rows = pc[assembled_entries.n - 1].r + 1;

    // Allocate the csr data structures
    row_offsets    = new uint[num_rows];
    col_indices    = new uint[assembled_entries.n];
    values         = new double[assembled_entries.n];
    row_offsets[0] = 0;
    for (uint r = 0, i = 0; r < num_rows; r++) {
      uint j = i;
      while (j < assembled_entries.n && pc[j].r == r)
        col_indices[j] = pc[j].c, values[j] = pc[j].v, j++;
      row_offsets[r + 1] = row_offsets[r] + j - i, i = j;
    }
  }
  array_free(&assembled_entries);
}

void LocalSolver_t::SetupUserToCompressMap(const slong *vtx) {
  typedef struct {
    ulong id;
    uint  idx;
    sint  perm;
  } vertex_id_t;

  struct array vids;
  array_init(vertex_id_t, &vids, input_size);

  {
    vertex_id_t vid;
    for (uint i = 0; i < input_size; i++) {
      vid.id = vtx[i], vid.idx = i;
      array_cat(vertex_id_t, &vids, &vid, 1);
    }
  }

  {
    sarray_sort(vertex_id_t, vids.ptr, vids.n, id, 1, &bfr);

    vertex_id_t *pv           = (vertex_id_t *)vids.ptr;
    ulong        id           = 0;
    sint         num_compress = 0;
    for (uint i = 0; i < vids.n; i++) {
      if (pv[i].id != id) id = pv[i].id, num_compress++;
      pv[i].perm = num_compress - 1;
    }

    compressed_size = num_compress;
  }

  {
    sarray_sort(vertex_id_t, vids.ptr, vids.n, idx, 0, &bfr);

    const vertex_id_t *pv = (const vertex_id_t *)vids.ptr;
    u_to_c                = new sint[input_size];
    for (uint i = 0; i < input_size; i++) u_to_c[i] = pv[i].perm;
  }

  array_free(&vids);
}

void LocalSolver_t::Setup(const uint input_size_, const slong *vtx,
                          const uint nnz, const uint *ia, const uint *ja,
                          const double *va, const double tol, const gs_dom dom,
                          const Algorithm_t  algorithm,
                          const std::string &backend, const int device_id) {
  input_size = input_size_;

  SetupUserToCompressMap(vtx);

  SetupCSRMatrix(vtx, nnz, ia, ja, va, tol);

  SetupAlgorithm(algorithm, dom, backend, device_id);
}

void LocalSolver_t::Solve(void *x, const void *rhs) {
  // TODO: Apply u_to_c mapping.
  solver->Solve(x, rhs);
}

LocalSolver_t::~LocalSolver_t() {
  delete[] row_offsets, col_indices, values, u_to_c;
  buffer_free(&bfr);
  delete solver;
}
