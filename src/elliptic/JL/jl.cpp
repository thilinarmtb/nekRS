#include "platform.hpp"
#include "elliptic.h"
#include "gslib.h"

static int check_alloc_(void *ptr, std::string file, int line) {
  if (ptr == NULL) {
    printf("check_alloc failure: %s:%d\n", file.c_str(), line);
    return 1;
  }
  return 0;
}

#define check_alloc(ptr) check_alloc_(ptr, __FILE__, __LINE__)

static int gen_crs_basis(double *b, int j_, dfloat *z, int Nq, int Np) {
  int jj = j_ + 1;
  double *zr = (double *) calloc(Nq, sizeof(double));
  double *zs = (double *) calloc(Nq, sizeof(double));
  double *zt = (double *) calloc(Nq, sizeof(double));
  double *z0 = (double *) calloc(Nq, sizeof(double));
  double *z1 = (double *) calloc(Nq, sizeof(double));
  if (zr == NULL || zs == NULL || zt == NULL || z0 == NULL || z1 == NULL)
    return 1;

  for(int i = 0; i < Nq; i++) {
    z0[i] = 0.5 * (1 - z[i]);
    z1[i] = 0.5 * (1 + z[i]);
  }

  memcpy(zr, z0, Nq * sizeof(double));
  memcpy(zs, z0, Nq * sizeof(double));
  memcpy(zt, z0, Nq * sizeof(double));

  if (jj % 2 == 0)
    memcpy(zr, z1, Nq * sizeof(double));
  if (jj == 3 || jj == 4 || jj == 7 || jj == 8)
    memcpy(zs, z1, Nq * sizeof(double));
  if (jj > 4)
    memcpy(zt, z1, Nq * sizeof(double));

  for(int k = 0; k < Nq; k++)
    for(int j = 0; j < Nq; j++)
      for(int i = 0; i < Nq; i++) {
        int n = i + Nq * j + Nq * Nq * k + j_ * Np;
        b[n] = zr[i] * zs[j] * zt[k];
      }

  free(zr);
  free(zs);
  free(zt);
  free(z0);
  free(z1);

  return 0;
}

static int get_local_crs_galerkin(double *a, int nc, mesh_t *mf,
                                  elliptic_t * ef) {
  int nelt = mf->Nelements;
  int Np = mf->Np;
  size_t size = nelt * Np;

  double *b = tcalloc(double, nc * Np);
  check_alloc(b);
  int j;
  for (j = 0; j < nc; j++)
    gen_crs_basis(b, j, mf->gllz, mf->Nq, mf->Np);

  double *u = tcalloc(double, size);
  check_alloc(u);
  double *w = tcalloc(double, size);
  check_alloc(w);

  occa::memory o_u = platform->device.malloc(size * sizeof(dfloat), u);
  occa::memory o_w = platform->device.malloc(size * sizeof(dfloat), w);

  int i, e, k;
  for (j = 0; j < nc; j++) {
    for (e = 0; e < nelt; e++)
      memcpy(&u[e * Np], &b[j * Np], Np * sizeof(double));

    // call Ax
    o_u.copyFrom(u);
    ellipticAx(ef, mf->NglobalGatherElements, mf->o_globalGatherElementList, o_u, o_w, dfloatString);
    ellipticAx(ef, mf->NlocalGatherElements, mf->o_localGatherElementList, o_u, o_w, dfloatString);
    o_w.copyTo(w);

    for (e = 0; e < nelt; e++)
      for (i = 0; i < nc; i++) {
        a[i + j * nc + e * nc * nc] = 0.0;
        for (k = 0; k < Np; k++)
          a[i + j * nc + e * nc * nc] += b[k + i * Np] * w[k + e * Np];
      }
  }


  free(w);
  free(u);
  o_u.free();
  o_w.free();
  free(b);

  return 0;
}

static void set_mat_ij(uint *ia, uint *ja, int nc, int nelt) {
  int i, j, e;
  for (e = 0; e < nelt; e++)
    for (j = 0; j < nc; j++)
      for (i = 0; i < nc; i++) {
        ia[i + j * nc + nc * nc * e] = e * nc + i;
        ja[i + j * nc + nc * nc * e] = e * nc + j;
      }
}

int jl_setup_aux(uint *ntot, ulong **gids_, uint *nnz, uint **ia_, uint **ja_,
                 double **a_, elliptic_t *elliptic, elliptic_t *ellipticf) {
  mesh_t *mesh = elliptic->mesh;
  mesh_t *meshf = ellipticf->mesh;
  assert(mesh->Nelements == meshf->Nelements);
  int nelt = meshf->Nelements;

  int nc = mesh->Np;

  /* Set global ids: copy and apply the mask */
  *ntot = nelt * nc;
  ulong *gids = *gids_ = tcalloc(ulong, *ntot);
  check_alloc(gids);

  int j;
  for (j = 0; j < nelt * nc; j++)
    gids[j] = mesh->globalIds[j];

  if (elliptic->Nmasked) {
    dlong* maskIds = (dlong*) calloc(elliptic->Nmasked, sizeof(dlong));
    elliptic->o_maskIds.copyTo(maskIds, elliptic->Nmasked * sizeof(dlong));
    for (int n = 0; n < elliptic->Nmasked; n++)
      gids[maskIds[n]] = 0;
    free(maskIds);
  }

  /* Set coarse matrix */
  *nnz = nc * nc * nelt;
  double *a = *a_ = tcalloc(double, *nnz);
  check_alloc(a);

  get_local_crs_galerkin(a, nc, meshf, ellipticf);

  uint *ia = *ia_ = tcalloc(uint, *nnz);
  check_alloc(ia);
  uint *ja = *ja_ = tcalloc(uint, *nnz);
  check_alloc(ja);
  set_mat_ij(ia, ja, nc, nelt);

  return 0;
}
#undef check_alloc

extern int xxt_setup(parAlmond::solver_t* M, uint n, const ulong *id, uint nnz,
                     const uint* Ai, const uint* Aj, const double* A, uint null,
                     uint verbose);
extern int xxt_solve(occa::memory o_x, occa::memory o_rhs);
extern int xxt_free();

extern int amg_setup(parAlmond::solver_t* M, uint n, const ulong *id, uint nnz,
                     const uint* Ai, const uint* Aj, const double* A, uint null,
                     uint verbose);
extern int amg_solve(occa::memory o_x, occa::memory o_rhs);
extern int amg_free();

extern int nekamg_setup(const char *session, int coars_strat, int interp_strat,
                        double tol, MPI_Comm serial);

static uint solver;
int jl_setup(uint type, parAlmond::solver_t* M, uint n, const ulong *id,
             uint nnz, const uint* Ai, const uint* Aj, const double* A,
             uint null, uint verbose) {
  int rank;
  MPI_Comm_rank(M->comm, &rank);

  MPI_Comm serial;
  MPI_Comm_split(M->comm, rank == 0, rank, &serial);

  int coarse = 6;
  char *cval = getenv("NEKRS_JL_AMG_COARSENING");
  if (cval != NULL)
    coarse = atoi(cval);

  int interp = 6;
  char *ival = getenv("NEKRS_JL_AMG_INTERPOLATION");
  if (ival != NULL)
    interp = atoi(ival);

  double tol = 0.5;
  char *tval = getenv("NEKRS_JL_AMG_TOLERANCE");
  if (tval != NULL)
    tol = atof(tval);

  int err = 0, errg;
  solver = type;
  switch (solver) {
    case 0: // XXT
      err = xxt_setup(M, n, id, nnz, Ai, Aj, A, null, verbose);
      break;
    case 1:
      err = amg_setup(M, n, id, nnz, Ai, Aj, A, null, verbose);
      // if (errg) {
      //   if (rank == 0)
      //     nekamg_setup("crs_amg_data", coarse, interp, tol, serial);
      //   MPI_Barrier(M->comm);
      //   err = amg_setup(M, n, id, nnz, Ai, Aj, A, null, verbose);
      // }
      break;
    default:
      break;
  }

  MPI_Comm_free(&serial);
  MPI_Allreduce(&err, &errg, 1, MPI_INT, MPI_MAX, M->comm);

  return errg;
}

int jl_solve(occa::memory o_x, occa::memory o_rhs) {
  int err = 1;
  switch (solver) {
    case 0: // XXT
      err = xxt_solve(o_x, o_rhs);
      break;
    case 1: // AMG
      err = amg_solve(o_x, o_rhs);
      break;
    default:
      break;
  }
  return err;
}

int jl_free() {
  int err = 1;
  switch (solver) {
    case 0: // XXT
      err = xxt_free();
      break;
    case 1:
      err = amg_free();
      break;
    default:
      break;
  }
  return err;
}
