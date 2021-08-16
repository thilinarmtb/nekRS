#include <coarseXXT.h>
#include "platform.hpp"

int check_alloc_(void *ptr, std::string file, int line) {
  if (ptr == NULL) {
    printf("check_alloc failure: %s:%d\n", file.c_str(), line);
    return 1;
  }
  return 0;
}

#define check_alloc(ptr) check_alloc_(ptr, __FILE__, __LINE__)

int gen_crs_basis(double *b, int j_, dfloat *z, int Nq, int Np) {
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

int get_local_crs_galerkin(double *a, int nc, mesh_t *meshf,
                           occa::kernel &ax_kernel, dfloat lambda) {
  int nelt = meshf->Nelements;
  int Np = meshf->Np;
  size_t size = nelt * Np;

  double *b = tcalloc(double, nc * Np);
  check_alloc(b);
  int j;
  for (j = 0; j < nc; j++)
    gen_crs_basis(b, j, meshf->gllz, meshf->Nq, meshf->Np);

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

    /* call Ax */
    o_u.copyFrom(u);
    ax_kernel(nelt, meshf->o_ggeo, meshf->o_D, meshf->o_DT, lambda, o_u, o_w);
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

void set_mat_ij(uint *ia, uint *ja, int nc, int nelt) {
  int i, j, e;
  for (e = 0; e < nelt; e++)
    for (j = 0; j < nc; j++)
      for (i = 0; i < nc; i++) {
        ia[i + j * nc + nc * nc * e] = e * nc + i;
        ja[i + j * nc + nc * nc * e] = e * nc + j;
      }
}

int setup_h1_crs(uint *ntot, ulong **gids_, uint *nnz, uint **ia_, uint **ja_,
                 double **a_, elliptic_t *elliptic, elliptic_t *ellipticf,
                 dfloat lambda) {
  mesh_t *mesh = elliptic->mesh;
  mesh_t *meshf = ellipticf->mesh;
  assert(mesh->Nelements == meshf->Nelements);
  int nelt = meshf->Nelements;

  int nc = mesh->Np;
  int nf = meshf->Np;

  /* Set global ids: copy and apply the mask */
  *ntot = nelt * nc;
  ulong *gids = *gids_ = tcalloc(ulong, *ntot);
  check_alloc(gids);

  int j;
  for (j = 0; j < nelt * nc; j++)
    gids[j] = mesh->globalIds[j];

  int n;
  for (n = 0; n < elliptic->Nmasked; n++)
    gids[elliptic->maskIds[n]] = 0;

  /* Set coarse matrix */
  *nnz = nc * nc * nelt;
  double *a = *a_ = tcalloc(double, *nnz);
  check_alloc(a);

  get_local_crs_galerkin(a, nc, meshf, ellipticf->AxKernel, lambda);

  uint *ia = *ia_ = tcalloc(uint, *nnz);
  check_alloc(ia);
  uint *ja = *ja_ = tcalloc(uint, *nnz);
  check_alloc(ja);
  set_mat_ij(ia, ja, nc, nelt);

  return 0;
}

#undef check_alloc
