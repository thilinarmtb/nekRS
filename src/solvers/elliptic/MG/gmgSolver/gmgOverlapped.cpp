#include <cassert>
#include <cstdio>
#include <cstdlib>

#include "gmgOverlapped.hpp"

#include "gslib.h"

#define MAX_NE_IS_TOO_SMALL   -1
#define MAX_NBRS_IS_TOO_SMALL -2
#define ELEMENT_NOT_FOUND     -4

typedef struct {
  uint   n;
  uint  *offs;
  ulong *nbrs;
  uint  *proc;
} element_to_neighbor_map_t;

typedef struct {
  uint   n;
  ulong *elist;
  uint  *wlist, *plist;
} element_to_processor_map_t;

typedef struct {
  ulong  eid, vid[8];
  double xyz[8 * 3], mat[8 * 8], mask[8];
  int    frontier[8];
  uint   p, seq;
} element_t;

typedef struct {
  ulong eid;
  uint  e;
} element_id_t;

typedef struct {
  ulong vid, eid, nid;
  uint  seq, p, np;
} vertex_t;

typedef struct {
  ulong eid;
  uint  p, seq;
} request_t;

typedef struct {
  ulong eid, nid;
  uint  p, np;
} response_t;

#ifdef __cplusplus
extern "C" {
#endif

static void comm_split_(const struct comm *s, int bin, int key, struct comm *d,
                        const char *file, unsigned line);

static void check_error_(int status, struct comm *c, const char *file,
                         unsigned line);

static int find_frontier(struct array *uelems, uint ne, const long long *eids,
                         unsigned nv, struct comm *ci, buffer *bfr);

static int binary_search(ulong eid, const void *const pe_, uint n);

static void find_immediate_neighbors(struct array *const v_to_e, const uint ne,
                                     const long long *eids, const uint nv,
                                     const long long      *vids,
                                     struct crystal *const cr, buffer *bfr);

static int
build_element_to_neighbor_and_proc_map(element_to_neighbor_map_t *const  e_to_n,
                                       element_to_processor_map_t *const e_to_p,
                                       const struct array *const vtx_to_e,
                                       const uint max_ne, const uint max_nbrs,
                                       const uint ne, struct comm *c);

static int update_maps_with_frontier(element_to_neighbor_map_t *const  e_to_n,
                                     element_to_processor_map_t *const e_to_p,
                                     struct crystal *cr, const uint nw,
                                     const uint ne, const long long *eids,
                                     const uint max_ne, const uint max_nbrs,
                                     buffer *bfr);

static void gather_extended_data(struct array                     *extended,
                                 const element_to_processor_map_t *e_to_p,
                                 const struct array *original, const uint ndim,
                                 struct crystal *const cr, buffer *bfr);

static void setup_output_arrays(unsigned *nei, long long *eids, uint ndim,
                                long long *vids, double *xyz, double *mask,
                                double *mat, int *frontier, const int nw,
                                int *wids, struct array *extended,
                                element_to_processor_map_t *const e_to_p,
                                struct comm *c, buffer *bfr);

static void find_overlapped_system(unsigned *nei, long long *eids, unsigned nv,
                                   long long *vids, double *xyz, double *mask,
                                   double *mat, int *frontier, unsigned nw,
                                   int *wids, MPI_Comm comm, unsigned max_ne);
#ifdef __cplusplus
}
#endif

/* comm_split_() and comm_split() should be removed once gslib is updated */
static void comm_split_(const struct comm *s, int bin, int key, struct comm *d,
                        const char *file, unsigned line) {
#if defined(MPI)
  MPI_Comm nc;
  MPI_Comm_split(s->c, bin, key, &nc);
  comm_init(d, nc);
  MPI_Comm_free(&nc);
#else
  if (s->np != 1) fail(1, file, line, "%s not compiled with -DMPI\n", file);
#endif
}
#define comm_split(s, bin, key, d)                                             \
  comm_split_(s, bin, key, d, __FILE__, __LINE__)

static void check_error_(const int status_, struct comm *c, const char *file,
                         const unsigned line) {
  sint status = status_, bfr[2];
  comm_allreduce(c, gs_int, gs_add, &status, 1, bfr);

  if (status == 0) return;

  if (c->id == 0) {
    fprintf(stderr, "find_overlapped_system: Error in file \"%s\" at line %u.",
            file, line);
    fflush(stderr);
  }

  MPI_Abort(c->c, 1);
}
#define check_error(call, c) check_error_(call, c, __FILE__, __LINE__)

static int find_frontier(struct array *uelems, uint ne, const long long *eids,
                         unsigned nv, struct comm *ci, buffer *bfr) {
  typedef struct {
    ulong id;
  } idx_t;

  struct array ids;
  array_init(idx_t, &ids, ne + 1);

  idx_t id;
  for (uint i = 0; i < ne; i++) {
    id.id = eids[i];
    array_cat(idx_t, &ids, &id, 1);
  }

  sarray_sort(idx_t, ids.ptr, ids.n, id, 1, bfr);

  // Initialize the frontier to -1. We will copy the frontier value for all the
  // input elements as is and for the new elements, it will be max(frontier
  // value of input elements) + 1.
  const size_t nu       = uelems->n;
  const size_t size     = nu * nv;
  int         *frontier = tcalloc(int, size);
  for (uint i = 0; i < nu * nv; i++) frontier[i] = -1;

  element_t *pu   = (element_t *)uelems->ptr;
  idx_t     *pi   = (idx_t *)ids.ptr;
  int        maxf = -1;
  uint       i = 0, j = 0;
  while (i < ne) {
    for (; j < nu && pu[j].eid < pi[i].id; j++)
      ;
    for (; j < nu && pu[j].eid == pi[i].id; j++) {
      for (unsigned v = 0; v < nv; v++) {
        frontier[j * nv + v] = pu[j].frontier[v];
        if (frontier[j * nv + v] > maxf) maxf = frontier[j * nv + v];
      }
    }
    i++;
  }

  maxf++;
  for (uint e = 0; e < nu; e++) {
    for (unsigned v = 0; v < nv; v++)
      if (frontier[e * nv + v] == -1) frontier[e * nv + v] = maxf;
  }

  for (uint e = 0; e < nu; e++)
    for (unsigned v = 0; v < nv; v++) pu[e].frontier[v] = frontier[e * nv + v];

  free(frontier), array_free(&ids);

  return maxf;
}

static int binary_search(ulong eid, const void *const pe_, uint n) {
  if (n == 0) return -1;

  const element_id_t *const pe = (const element_id_t *)pe_;
  uint                      l  = 0;
  uint                      u  = n - 1;
  while (u - l > 1) {
    uint mid = (u + l) / 2;
    if (pe[mid].eid == eid) return mid;
    if (pe[mid].eid < eid) l = mid;
    else u = mid;
  }

  if (pe[l].eid == eid) return l;
  if (pe[u].eid == eid) return u;
  return -1;
}

static void find_immediate_neighbors(struct array *const v_to_e, const uint ne,
                                     const long long *eids, const uint nv,
                                     const long long      *vids,
                                     struct crystal *const cr, buffer *bfr) {
  struct array vtxs;
  const size_t vtxs_size = (size_t)ne * (size_t)nv;
  array_init(vertex_t, &vtxs, ne * nv);

  struct comm *c = &cr->comm;

  vertex_t vt = {.np = c->id};
  for (uint e = 0; e < ne; e++) {
    vt.eid = eids[e], vt.seq = e;
    for (unsigned v = 0; v < nv; v++) {
      vt.vid = vids[e * nv + v], vt.p = vt.vid % c->np;
      array_cat(vertex_t, &vtxs, &vt, 1);
    }
  }

  sarray_transfer(vertex_t, &vtxs, p, 1, cr);
  sarray_sort(vertex_t, vtxs.ptr, vtxs.n, vid, 1, bfr);

  array_init(vertex_t, v_to_e, vtxs.n);
  vertex_t *pv = (vertex_t *)vtxs.ptr;
  uint      s  = 0;
  while (s < vtxs.n) {
    uint e = s + 1;
    while (e < vtxs.n && pv[s].vid == pv[e].vid) e++;
    for (uint i = s; i < e; i++) {
      vt = pv[i];
      for (uint j = s; j < e; j++) {
        vt.np = pv[j].p, vt.nid = pv[j].eid;
        array_cat(vertex_t, v_to_e, &vt, 1);
      }
    }
    s = e;
  }
  array_free(&vtxs);

  sarray_transfer(vertex_t, v_to_e, p, 0, cr);
  sarray_sort_2(vertex_t, v_to_e->ptr, v_to_e->n, seq, 0, nid, 1, bfr);
}

static int
build_element_to_neighbor_and_proc_map(element_to_neighbor_map_t *const  e_to_n,
                                       element_to_processor_map_t *const e_to_p,
                                       const struct array *const vtx_to_e,
                                       const uint max_ne, const uint max_nbrs,
                                       const uint ne, struct comm *c) {
  e_to_p->elist = tcalloc(ulong, max_ne);
  e_to_p->wlist = tcalloc(uint, max_ne);
  e_to_p->plist = tcalloc(uint, max_ne);

  e_to_n->offs = tcalloc(uint, max_ne + 1);
  e_to_n->nbrs = tcalloc(ulong, max_nbrs);
  e_to_n->proc = tcalloc(uint, max_nbrs);

  const vertex_t *pv  = (vertex_t *)vtx_to_e->ptr;
  uint            s   = 0;
  uint            cnt = 0;
  e_to_n->offs[0]     = 0;
  while (s < vtx_to_e->n) {
    e_to_p->elist[cnt] = pv[s].eid;
    e_to_p->wlist[cnt] = 0;
    e_to_p->plist[cnt] = c->id;

    uint e = s + 1;
    while (e < vtx_to_e->n && pv[s].eid == pv[e].eid) e++;

    uint s0 = e_to_n->offs[cnt];
    if (s0 + e - s > max_nbrs) MAX_NE_IS_TOO_SMALL;

    e_to_n->nbrs[s0] = pv[s].nid;
    e_to_n->proc[s0] = pv[s].np;
    s0++;
    for (uint i = s + 1; i < e; i++) {
      if (e_to_n->nbrs[s0 - 1] != pv[i].nid) {
        e_to_n->nbrs[s0] = pv[i].nid;
        e_to_n->proc[s0] = pv[i].np;
        s0++;
      }
    }
    cnt++;
    e_to_n->offs[cnt] = s0;
    s                 = e;
  }
  // Sanity checks.
  assert(cnt == ne);

  return 0;
}

static int update_maps_with_frontier(element_to_neighbor_map_t *const  e_to_n,
                                     element_to_processor_map_t *const e_to_p,
                                     struct crystal *cr, const uint nw,
                                     const uint ne, const long long *eids,
                                     const uint max_ne, const uint max_nbrs,
                                     buffer *bfr) {
  struct array front;
  array_init(element_id_t, &front, 3 * (ne + 1) / 2);

  element_id_t element;
  for (uint e = 0; e < ne; e++) {
    element.eid = eids[e], element.e = e;
    array_cat(element_id_t, &front, &element, 1);
  }
  sarray_sort(element_id_t, front.ptr, front.n, eid, 1, bfr);

  struct array input;
  array_init(element_id_t, &input, ne);
  array_cat(element_id_t, &input, front.ptr, front.n);

  struct array rqsts;
  array_init(request_t, &rqsts, ne);

  struct array respns;
  array_init(response_t, &respns, rqsts.n * 10);

  uint s, e;
  uint fs = 0, fe = ne;
  for (unsigned w = 1; w <= nw; w++) {
    // Find all the new elements appearing in the map in last wave.
    for (uint i = fs; i < fe; i++) {
      for (s = e_to_n->offs[i], e = e_to_n->offs[i + 1]; s < e; s++) {
        int index = binary_search(e_to_n->nbrs[s], front.ptr, front.n);
        if (index >= 0) continue;

        element_id_t element = {.eid = e_to_n->nbrs[s]};
        array_cat(element_id_t, &front, &element, 1);
        // FIXME: Maybe we can avoid doing a sort everytime.
        sarray_sort(element_id_t, front.ptr, front.n, eid, 1, bfr);

        request_t request = {.eid = e_to_n->nbrs[s], .p = e_to_n->proc[s]};
        array_cat(request_t, &rqsts, &request, 1);
      }
    }

    // Get the neighbors of the new elements.
    sarray_transfer(request_t, &rqsts, p, 1, cr);
    sarray_sort(request_t, rqsts.ptr, rqsts.n, eid, 1, bfr);

    request_t *pr = (request_t *)rqsts.ptr;
    for (uint i = 0; i < rqsts.n; i++) {
      int idx = binary_search(pr[i].eid, (element_id_t *)input.ptr, input.n);
      if (idx < 0 || idx >= ne) return ELEMENT_NOT_FOUND;

      response_t request = {.eid = pr[i].eid, .p = pr[i].p};
      for (s = e_to_n->offs[idx], e = e_to_n->offs[idx + 1]; s < e; s++) {
        request.nid = e_to_n->nbrs[s];
        request.np  = e_to_n->proc[s];
        array_cat(response_t, &respns, &request, 1);
      }
    }

    sarray_transfer(response_t, &respns, p, 1, cr);
    sarray_sort_2(response_t, respns.ptr, respns.n, eid, 1, nid, 1, bfr);

    // Update the map with the new elements and their neighbors.
    response_t *prs = (response_t *)respns.ptr;
    fs = fe, s = 0;
    while (s < respns.n) {
      if (fe >= max_ne) return MAX_NE_IS_TOO_SMALL;

      e_to_p->elist[fe] = prs[s].eid;
      e_to_p->plist[fe] = prs[s].p;
      e_to_p->wlist[fe] = w;
      fe++;

      e = s + 1;
      while (e < respns.n && prs[s].eid == prs[e].eid) e++;

      e_to_n->offs[fe] = e_to_n->offs[fe - 1] + e - s;
      if (max_nbrs < e_to_n->offs[fe]) return MAX_NBRS_IS_TOO_SMALL;

      for (uint i = 0; i < e - s; i++) {
        e_to_n->proc[e_to_n->offs[fe - 1] + i] = prs[s + i].np;
        e_to_n->nbrs[e_to_n->offs[fe - 1] + i] = prs[s + i].nid;
      }
      s = e;
    }

    rqsts.n = respns.n = 0;
  }
  e_to_n->n = e_to_p->n = fe;

  array_free(&front);
  array_free(&rqsts), array_free(&respns), array_free(&input);
  free(e_to_n->offs), free(e_to_n->proc), free(e_to_n->nbrs);

  return 0;
}

static void fill_array_with_input(struct array *original, const uint ne,
                                  const uint nv, const uint ndim,
                                  const long long *eids, const long long *vids,
                                  const double *mask, const double *mat,
                                  const double *xyz, const struct comm *c,
                                  buffer *bfr) {
  array_init(element_t, original, ne);

  element_t element;
  for (uint i = 0; i < ne; i++) {
    element.eid = eids[i], element.p = c->id;
    for (uint v = 0; v < nv; v++) {
      element.vid[v]  = vids[i * nv + v];
      element.mask[v] = mask[i * nv + v];
      for (uint j = 0; j < nv; j++)
        element.mat[v * nv + j] = mat[i * nv * nv + v * nv + j];
      for (uint d = 0; d < ndim; d++)
        element.xyz[v * ndim + d] = xyz[i * nv * ndim + v * ndim + d];
    }
    array_cat(element_t, original, &element, 1);
  }
  sarray_sort(element_t, original->ptr, original->n, eid, 1, bfr);
}

static void gather_extended_data(struct array                     *extended,
                                 const element_to_processor_map_t *e_to_p,
                                 const struct array *original, const uint ndim,
                                 struct crystal *const cr, buffer *bfr) {
  const uint fe = e_to_p->n;

  struct array requests;
  array_init(request_t, &requests, fe);

  for (uint i = 0; i < fe; i++) {
    request_t r = {.eid = e_to_p->elist[i], .p = e_to_p->plist[i], .seq = i};
    array_cat(request_t, &requests, &r, 1);
  }

  sarray_transfer(request_t, &requests, p, 1, cr);
  sarray_sort(request_t, requests.ptr, requests.n, eid, 1, bfr);

  array_init(element_t, extended, requests.n);

  const request_t *const pr = (const request_t *)requests.ptr;
  const element_t *const po = (const element_t *)original->ptr;
  const uint             ne = original->n;
  const uint             nv = (ndim == 3) ? 8 : 4;
  element_t              element;
  for (uint i = 0, j = 0; i < requests.n; i++) {
    while (j < ne && po[j].eid < pr[i].eid) j++;
    // Sanity check.
    assert(j < ne && po[j].eid == pr[i].eid);

    element.eid = po[j].eid;
    element.p   = pr[i].p;
    element.seq = pr[i].seq;
    for (unsigned v = 0; v < nv; v++) {
      element.vid[v]  = po[j].vid[v];
      element.mask[v] = po[j].mask[v];
      for (unsigned k = 0; k < nv; k++)
        element.mat[v * nv + k] = po[j].mat[v * nv + k];
      for (unsigned d = 0; d < ndim; d++)
        element.xyz[v * ndim + d] = po[j].xyz[v * ndim + d];
    }
    array_cat(element_t, extended, &element, 1);
  }
  array_free(&requests);

  sarray_transfer(element_t, extended, p, 0, cr);
  sarray_sort(element_t, extended->ptr, extended->n, seq, 0, bfr);
  assert(e_to_p->n == extended->n);
}

static void setup_output_arrays(unsigned *nei, long long *eids, uint ndim,
                                long long *vids, double *xyz, double *mask,
                                double *mat, int *frontier, const int nw,
                                int *wids, struct array *extended,
                                element_to_processor_map_t *const e_to_p,
                                struct comm *c, buffer *bfr) {
  element_t *pe = (element_t *)extended->ptr;
  const uint nv = (ndim == 3) ? 8 : 4;
  const uint fe = *nei = e_to_p->n;
  for (uint i = 0; i < fe; i++) {
    // Sanity check.
    assert(e_to_p->elist[i] == pe[i].eid);
    eids[i] = e_to_p->elist[i];
    wids[i] = e_to_p->wlist[i];
    for (unsigned v = 0; v < nv; v++) {
      vids[i * nv + v] = pe[i].vid[v];
      mask[i * nv + v] = pe[i].mask[v];
      for (unsigned j = 0; j < nv; j++)
        mat[i * nv * nv + v * nv + j] = pe[i].mat[v * nv + j];
      for (unsigned d = 0; d < ndim; d++)
        xyz[i * nv * ndim + v * ndim + d] = pe[i].xyz[v * ndim + d];
    }
  }
  free(e_to_p->elist), free(e_to_p->wlist), free(e_to_p->plist);

  // Setup the Frontier array.
  for (uint e = 0; e < fe; e++) {
    if (wids[e] == nw)
      for (unsigned v = 0; v < nv; v++) frontier[e * nv + v] = 1;
    else
      for (unsigned v = 0; v < nv; v++) frontier[e * nv + v] = 0;
  }

  // Make sure frontier values are consistent.
  struct comm lc;
  comm_split(c, c->id, c->id, &lc);
  struct gs_data *gsh = gs_setup(vids, fe * nv, &lc, 0, gs_pairwise, 0);
  gs(frontier, gs_int, gs_min, 0, gsh, bfr);
  gs_free(gsh), comm_free(&lc);
}

static void find_overlapped_system(unsigned *nei, long long *eids, unsigned nv,
                                   long long *vids, double *xyz, double *mask,
                                   double *mat, int *frontier, unsigned nw,
                                   int *wids, MPI_Comm comm, unsigned max_ne) {
  const uint ne   = *nei;
  const uint ndim = (nv == 8) ? 3 : 2;

  struct comm c;
  comm_init(&c, comm);

  struct crystal cr;
  crystal_init(&cr, &c);

  buffer bfr;
  buffer_init(&bfr, 1024);

  // Find neighbor elements of input elements based on vertex connectivity.
  struct array v_to_e;
  find_immediate_neighbors(&v_to_e, ne, eids, nv, vids, &cr, &bfr);

  // Build element to neighbor map and element to processor map for input
  // elements.
  const uint                 max_nbrs = 100 * max_ne;
  element_to_neighbor_map_t  e_to_n;
  element_to_processor_map_t e_to_p;
  int status = build_element_to_neighbor_and_proc_map(&e_to_n, &e_to_p, &v_to_e,
                                                      max_ne, max_nbrs, ne, &c);
  check_error(status, &c);
  array_free(&v_to_e);

  // Put all local elements in frontier array and sort by element id.
  // We will keep updating this and the map as we update the frontier.
  status = update_maps_with_frontier(&e_to_n, &e_to_p, &cr, nw, ne, eids,
                                     max_ne, max_nbrs, &bfr);
  check_error(status, &c);

  // Now we have the element ids of the extended domain. We need to bring
  // other data in now. First we will put input data into an array and then
  // sort by element id (it could be that they are not sorted in input).
  struct array original;
  fill_array_with_input(&original, ne, nv, ndim, eids, vids, mask, mat, xyz, &c,
                        &bfr);

  // Now we are sending the requests to bring in the data for the extended
  // domain. Code doesn't distinguish between original and extended element
  // ids. It just asks for all the elemetns and sort them by element id.
  struct array extended;
  gather_extended_data(&extended, &e_to_p, &original, ndim, &cr, &bfr);
  array_free(&original);

  // We have all the data now. Let's sort them by original element ids and
  // set the Fortran array correctly.
  setup_output_arrays(nei, eids, ndim, vids, xyz, mask, mat, frontier, nw, wids,
                      &extended, &e_to_p, &c, &bfr);
  array_free(&extended);

  buffer_free(&bfr), crystal_free(&cr), comm_free(&c);
}

#undef comm_split
#undef check_error

#undef MAX_NE_IS_TOO_SMALL
#undef MAX_NBRS_IS_TOO_SMALL
#undef ELEMENT_NOT_FOUND

void gmgFindOverlappedSystem(VecLong_t &Aids, VecUInt_t &Ai, VecUInt_t &Aj,
                             VecDouble_t &Av, VecInt_t &frontier,
                             const MPI_Comm comm) {}
