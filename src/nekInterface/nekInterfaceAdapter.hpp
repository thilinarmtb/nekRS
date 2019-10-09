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

int buildNekInterface(const char *casename, int nFields, int N, int np);
void nek_copyFrom(ins_t *ins, dfloat time, int tstep);
void nek_ocopyFrom(ins_t *ins, dfloat time, int tstep);
void nek_copyFrom(ins_t *ins, dfloat time);
void nek_copyTo(ins_t *ins, dfloat &time);
void nek_ocopyTo(ins_t *ins, dfloat &time);
void nek_copyRestart(ins_t *ins);

#endif
