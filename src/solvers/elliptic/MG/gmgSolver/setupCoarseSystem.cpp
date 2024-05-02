#include <cassert>

#include "gmgSetup.hpp"

#include "platform.hpp"

static void generateCoarseBasis(VecDfloat_t &b, const VecDfloat_t &z,
                                const unsigned Nqf, const unsigned Nqc) {
  VecDouble_t z0(Nqf), z1(Nqf);
  for (size_t i = 0; i < Nqf; i++) {
    z0[i] = 0.5 * (1 - z[i]);
    z1[i] = 0.5 * (1 + z[i]);
  }

  for (unsigned l = 0; l < Nqc; l++) {
    const unsigned ll = l + 1;

    VecDouble_t zr(z0);
    if (ll % 2 == 0) zr = z1;

    VecDouble_t zs(z0);
    if (ll == 3 || ll == 4 || ll == 7 || ll == 8) zs = z1;

    VecDouble_t zt(z0);
    if (ll > 4) zt = z1;

    const size_t Nqf2 = Nqf * Nqf;
    const size_t Nqf3 = Nqf * Nqf2;
    for (size_t k = 0; k < Nqf; k++) {
      for (size_t j = 0; j < Nqf; j++) {
        for (size_t i = 0; j < Nqf; i++)
          b[i + Nqf * j + Nqf2 * k + Nqf3 * l] = zr[i] * zs[j] * zt[k];
      }
    }
  }
}

static void setupGalerkinCoarseSystem(VecUInt_t &Ai, VecUInt_t &Aj,
                                      VecDouble_t &Av, const size_t Nqc,
                                      const mesh_t *const mesh,
                                      elliptic_t *const   elliptic) {
  const size_t Nqf = mesh->Nq;
  const size_t Npf = mesh->Np;
  // Sanity check:
  assert(Npf == Nqf * Nqf * Nqf);

  VecDfloat_t b(Nqc * Npf), z(mesh->gllz, mesh->gllz + Nqf);
  generateCoarseBasis(b, z, Nqf, Nqc);

  const size_t Nelements = (size_t)(mesh->Nelements);
  const size_t Nlocal    = (size_t)(mesh->Nlocal);
  // Sanity check:
  assert(Nlocal == Nelements * Npf);

  VecDfloat_t u(Nlocal), w(Nlocal);
  for (size_t i = 0; i < Nlocal; i++) u[i] = w[i] = 0;

  auto o_ud = platform->device.malloc(Nlocal * sizeof(dfloat));
  auto o_wd = platform->device.malloc(Nlocal * sizeof(dfloat));
  auto o_up = platform->device.malloc(Nlocal * sizeof(pfloat));
  auto o_wp = platform->device.malloc(Nlocal * sizeof(pfloat));

  const size_t Nqc2   = Nqc * Nqc;
  auto         b_iter = b.begin();
  for (size_t j = 0; j < Nqc; j++) {
    auto b_next = std::next(b_iter, Npf);
    auto u_iter = u.begin();
    for (size_t e = 0; e < Nelements; e++)
      std::copy(b_iter, b_next, u_iter), u_iter = std::next(u_iter, Npf);

    o_ud.copyFrom(u.data());
    platform->copyDfloatToPfloatKernel(Nlocal, o_ud, o_up);
    ellipticAx(elliptic, Nelements, mesh->o_elementList, o_up, o_wp,
               pfloatString);
    platform->copyPfloatToDfloatKernel(Nlocal, o_wp, o_wd);
    o_wd.copyTo(w.data());

    auto w_iter = w.begin();
    for (size_t e = 0; e < Nelements; e++) {
      for (size_t i = 0; i < Nqc; i++) {
        auto bb = std::next(b.begin(), i * Npf), be = std::next(bb, Npf);
        Av[i + j * Nqc + e * Nqc2] = std::inner_product(bb, be, w_iter, 0.0);
      }
      w_iter = std::next(w_iter, Npf);
    }

    b_iter = b_next;
  }

  o_ud.free();
  o_wd.free();
  o_up.free();
  o_wp.free();

  for (size_t e = 0; e < Nelements; e++) {
    for (size_t j = 0; j < Nqc; j++) {
      for (size_t i = 0; i < Nqc; i++) {
        Ai[i + j * Nqc + e * Nqc2] = e * Nqc + i;
        Aj[i + j * Nqc + e * Nqc2] = e * Nqc + j;
      }
    }
  }
}

void setupCoarseSystem(VecLong_t &gIds, VecUInt_t &Ai, VecUInt_t &Aj,
                       VecDouble_t &Av, elliptic_t *const ecrs,
                       elliptic_t *const efine) {
  const mesh_t *const meshc = ecrs->mesh;
  const mesh_t *const meshf = efine->mesh;

  // Set global ids: copy ids from nekRS.
  const unsigned Nqc   = meshc->Np;
  const size_t   ndofs = (size_t)(meshf->Nelements) * (size_t)Nqc;
  gIds.reserve(ndofs);
  for (size_t j = 0; j < ndofs; j++) gIds[j] = meshc->globalIds[j];

  // Apply the mask for Dirichlet BCs.
  VecDlong_t maskIds(ecrs->Nmasked);
  ecrs->o_maskIds.copyTo(maskIds.data(), ecrs->Nmasked * sizeof(dlong));
  for (size_t n = 0; n < ecrs->Nmasked; n++) gIds[maskIds[n]] = 0;

  // Setup the coarse system.
  const size_t nnz = (size_t)Nqc * ndofs;
  Ai.reserve(nnz);
  Aj.reserve(nnz);
  Av.reserve(nnz);
  setupGalerkinCoarseSystem(Ai, Aj, Av, Nqc, meshf, efine);
}
