#if !defined(nek_interface_)
#define nek_interface_

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <dlfcn.h>
#include <mpi.h>

#include "setupAide.hpp"
#include "nekrs.hpp"
#include "nek.hpp"

typedef struct {
  double *param;

  int *istep;
  int *ifield;

  /* x,y and z co-ordinates */
  double *xm1, *ym1, *zm1;
  double *xc, *yc, *zc;

  double *unx, *uny, *unz;

  double *time;

  /* solution */
  double *vx, *vy, *vz;
  double *pr;
  double *t;

  int *ifgetu, *ifgetp;

  double *cbscnrs;

  /* global vertex ids */
  long long *glo_num, ngv;

  /* Boundary data */
  char *cbc;
  int *boundaryID;

  int NboundaryIDs;

  /* id to face mapping */
  int *eface1, *eface, *icface;

  /* dimension of the problem */
  int ndim;
  /* local problem size */
  int nelv, nelt;
  int lelt;
  /* polynomial order + 1*/
  int nx1;

  /* MPI communicator */
  MPI_Comm comm;
} nekdata_private;

extern nekdata_private nekData;

int buildNekInterface(const char *casename, int nFields, int N, int np);
void nek_copyFrom(ins_t *ins, dfloat time, int tstep);
void nek_ocopyFrom(ins_t *ins, dfloat time, int tstep);
void nek_copyFrom(ins_t *ins, dfloat time);
void nek_copyTo(ins_t *ins, dfloat &time);
void nek_ocopyTo(ins_t *ins, dfloat &time);
void nek_copyRestart(ins_t *ins);
void nek_setic(void);
int  nekInterfaceAdapterSetup(MPI_Comm c,setupAide &options_in);

#endif
