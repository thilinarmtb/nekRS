#include "gslib.h"

#if defined(PARRSB)
#include "parRSB.h"
#endif

#define MAXNV 8 /* maximum number of vertices per element */
typedef struct{
  long long vtx[MAXNV];
  ulong eid;
  int proc;
} edata;

#ifdef PARMETIS

#include "parmetis.h"
#include "defs.h"

int parMETIS_partMesh(int *part, long long *vl, int nel, int nv, int *opt, comm_ext ce)
{
  int i, j;
  int ierrm;
  double time, time0;

  MPI_Comm comms;
  struct comm comm;
  int color;
  int ibuf;

  struct crystal cr;
  struct array A;
  edata *row;

  long long nell;
  long long *nelarray;
  idx_t *elmdist;
  idx_t *evlptr;
  idx_t *part_;
  real_t *tpwgts;
  idx_t edgecut;
  real_t ubvec;
  idx_t *elmwgt;
  idx_t wgtflag;
  idx_t numflag;
  idx_t ncon;
  idx_t ncommonnodes;
  idx_t nparts;
  idx_t nelsm;
  idx_t options[10];

  ierrm = METIS_OK;
  nell = nel;
  edgecut = 0;
  wgtflag = 0;
  numflag = 0;
  ncon = 1;
  ubvec = 1.02;
  elmwgt = NULL; /* no weights */
  ncommonnodes = 2;

  part_ = (idx_t*) malloc(nel*sizeof(idx_t));

  if (sizeof(idx_t) != sizeof(long long)){
    printf("ERROR: invalid sizeof(idx_t)!\n");
    goto err;
  }
  if (nv != 4 && nv != 8){
    printf("ERROR: nv is %d but only 4 and 8 are supported!\n", nv);
    goto err;
  }

  color = MPI_UNDEFINED;
  if (nel > 0) color = 1;
  MPI_Comm_split(ce, color, 0, &comms);
  if (color == MPI_UNDEFINED)
    goto end;

  comm_init(&comm,comms);
  if (comm.id == 0)
    printf("Running parMETIS ... "), fflush(stdout);

  nelarray = (long long*) malloc(comm.np*sizeof(long long));
  MPI_Allgather(&nell, 1, MPI_LONG_LONG_INT, nelarray, 1, MPI_LONG_LONG_INT, comm.c);
  elmdist = (idx_t*) malloc((comm.np+1)*sizeof(idx_t));
  elmdist[0] = 0;
  for (i=0; i<comm.np; ++i)
    elmdist[i+1] = elmdist[i] + (idx_t)nelarray[i];
  free(nelarray);

  evlptr = (idx_t*) malloc((nel+1)*sizeof(idx_t));
  evlptr[0] = 0;
  for (i=0; i<nel; ++i)
    evlptr[i+1] = evlptr[i] + nv;
  nelsm = elmdist[comm.id+1] - elmdist[comm.id];
  evlptr[nelsm]--;

  if (nv == 8) ncommonnodes = 4;
  nparts = comm.np;

  options[0] = 1;
  options[PMV3_OPTION_DBGLVL] = 0;
  options[PMV3_OPTION_SEED]   = 0;
  if (opt[0] != 0) {
    options[PMV3_OPTION_DBGLVL] = opt[1];
    if (opt[2] != 0) {
      options[3] = PARMETIS_PSR_UNCOUPLED;
      nparts = opt[2];
    }
  }

  tpwgts = (real_t*) malloc(ncon*nparts*sizeof(real_t));
  for (i=0; i<ncon*nparts; ++i)
    tpwgts[i] = 1./(real_t)nparts;

  if (options[3] == PARMETIS_PSR_UNCOUPLED)
    for (i=0; i<nel; ++i)
      part_[i] = comm.id;

  comm_barrier(&comm);
  time0 = comm_time();
  ierrm = ParMETIS_V3_PartMeshKway(elmdist,
                                   evlptr,
                                   (idx_t*)vl,
                                   elmwgt,
                                   &wgtflag,
                                   &numflag,
                                   &ncon,
                                   &ncommonnodes,
                                   &nparts,
                                   tpwgts,
                                   &ubvec,
                                   options,
                                   &edgecut,
                                   part_,
                                   &comm.c);

  time = comm_time() - time0;
  if (comm.id == 0)
    printf("%lf sec\n", time), fflush(stdout);

  for (i=0; i<nel; ++i)
    part[i] = part_[i];

  free(elmdist);
  free(evlptr);
  free(tpwgts);
  MPI_Comm_free(&comms);
  comm_free(&comm);

end:
  comm_init(&comm,ce);
  comm_allreduce(&comm, gs_int, gs_min, &ierrm, 1, &ibuf);
  if (ierrm != METIS_OK) goto err;
  return 0;

err:
  return 1;
}
#endif

void print_part_stat(long long *vtx, int nel, int nv, comm_ext ce)
{
  int i,j;

  struct comm comm;
  int np, id;

  int Nmsg;
  int *Ncomm;

  int nelMin, nelMax;
  int ncMin, ncMax, ncSum;
  int nsMin, nsMax, nsSum;
  int nssMin, nssMax;
  long long nssSum;

  struct gs_data *gsh;
  int b;
  long long b_long_long;

  int numPoints;
  long long *data;

  comm_init(&comm,ce);
  np = comm.np;
  id = comm.id;

  if (np == 1) return;

  numPoints = nel*nv;
  data = (long long*) malloc(numPoints*sizeof(long long));
  for(i = 0; i < numPoints; i++) data[i] = vtx[i];

  gsh = gs_setup(data, numPoints, &comm, 0, gs_pairwise, 0);

  pw_data_nmsg(gsh, &Nmsg);
  Ncomm = (int *) malloc(Nmsg*sizeof(int));
  pw_data_size(gsh, Ncomm);

  gs_free(gsh);
  free(data);

  ncMax = Nmsg;
  ncMin = Nmsg;
  ncSum = Nmsg;
  comm_allreduce(&comm, gs_int, gs_max, &ncMax , 1, &b);
  comm_allreduce(&comm, gs_int, gs_min, &ncMin , 1, &b);
  comm_allreduce(&comm, gs_int, gs_add, &ncSum , 1, &b);

  nsMax = Ncomm[0];
  nsMin = Ncomm[0];
  nsSum = Ncomm[0];
  for (i=1; i<Nmsg; ++i){
    nsMax = Ncomm[i] > Ncomm[i-1] ? Ncomm[i] : Ncomm[i-1];
    nsMin = Ncomm[i] < Ncomm[i-1] ? Ncomm[i] : Ncomm[i-1];
    nsSum += Ncomm[i];
  }
  comm_allreduce(&comm, gs_int, gs_max, &nsMax , 1, &b);
  comm_allreduce(&comm, gs_int, gs_min, &nsMin , 1, &b);

  nssMin = nsSum;
  nssMax = nsSum;
  nssSum = nsSum;
  comm_allreduce(&comm, gs_int, gs_max, &nssMax , 1, &b);
  comm_allreduce(&comm, gs_int, gs_min, &nssMin , 1, &b);
  comm_allreduce(&comm, gs_long_long, gs_add, &nssSum , 1, &b_long_long);

  nsSum = nsSum/Nmsg;
  comm_allreduce(&comm, gs_int, gs_add, &nsSum , 1, &b);

  nelMax = nel;
  nelMin = nel;
  comm_allreduce(&comm, gs_int, gs_max, &nelMax, 1, &b);
  comm_allreduce(&comm, gs_int, gs_min, &nelMin, 1, &b);

  if (id == 0) {
    printf(
      " nElements   max/min/bal: %d %d %.2f\n",
      nelMax, nelMin, (double)nelMax/nelMin);
    printf(
      " nMessages   max/min/avg: %d %d %.2f\n",
      ncMax, ncMin, (double)ncSum/np);
    printf(
      " msgSize     max/min/avg: %d %d %.2f\n",
      nsMax, nsMin, (double)nsSum/np);
    printf(
      " msgSizeSum  max/min/avg: %d %d %.2f\n",
      nssMax, nssMin, (double)nssSum/np);
    fflush(stdout);
  }

  comm_free(&comm);
}

int redistribute_data(int *nel_, long long *vl, long long *el, int *part,
    int nv, int lelt, struct comm *comm) {
  int nel=*nel_;

  struct crystal cr;
  struct array eList;
  edata *data;

  int count,e,n,ibuf;

  /* redistribute data */
  array_init(edata, &eList, nel), eList.n = nel;
  for(data = eList.ptr, e = 0; e < nel; ++e) {
    data[e].proc= part[e];
    data[e].eid = el[e];
    for(n = 0; n < nv; ++n) {
      data[e].vtx[n] = vl[e*nv + n];
    }
  }

  crystal_init(&cr,comm);
  sarray_transfer(edata, &eList, proc, 0, &cr);
  crystal_free(&cr);

  buffer bfr;
  buffer_init(&bfr, 1024);
  sarray_sort(edata, eList.ptr, eList.n, eid, 1, &bfr);
  buffer_free(&bfr);

  *nel_=nel=eList.n;

  count = 0;
  if (nel > lelt) count = 1;
  comm_allreduce(comm, gs_int, gs_add, &count, 1, &ibuf);
  if (count > 0) {
    count = nel;
    comm_allreduce(comm, gs_int, gs_max, &count, 1, &ibuf);
    if (comm->id == 0)
      printf("ERROR: resulting parition requires lelt=%d!\n", count);
    return 1;
  }

  for(data = eList.ptr, e = 0; e < nel; ++e) {
    el[e] = data[e].eid;
    for(n = 0; n < nv; ++n) {
      vl[e*nv + n] = data[e].vtx[n];
    }
  }

  array_free(&eList);

  return 0;
}

#define fpartmesh FORTRAN_UNPREFIXED(fpartmesh,FPARTMESH)
void fpartmesh(int *nell, long long *el, long long *vl, double *xyz,
    const int *const tag, const int *const lelm, const int *const nve,
    const int *const fcomm, const int *const fpartitioner,
    const int *const falgo, const int *const loglevel, int *rtval)
{
  struct comm comm;

  int nel, nv, lelt, partitioner, algo;
  int e, n;
  int count, ierr, ibuf;
  int *part;
  int opt[3];

  lelt = *lelm;
  nel = *nell;
  nv = *nve;
  // partitioner: 0 - Default, 1 - RSB, 2 - RCB
  partitioner = *fpartitioner;
  // algo: 0 - Lanczos, 1 - MG (Used only when partitioner = 1)
  algo = *falgo;

#if defined(MPI)
  comm_ext cext = MPI_Comm_f2c(*fcomm);
#else
  comm_ext cext = 0;
#endif
  comm_init(&comm, cext);

  ierr = 1;
  part = (int *)malloc(lelt * sizeof(int));
#if defined(PARRSB)
  parrsb_options options = parrsb_default_options;
  // parrsb_default_options:
  // - General options
  //   partitioner: Partition algo: 0 - RSB, 1 - RCB, 2 - RIB (Default: 0)
  //   verbose_level: Verbose level: 0, 1, 2, .. etc (Default: 1)
  //   profile_level: Profile level: 0, 1, 2, .. etc (Default: 1)
  //   level: Number of levles in partitioning: 1, 2 (Default: 2)
  //   repair: Repair disconnected components: 0 - No, 1 - Yes (Default: 0)
  // - RSB specific
  //   rsb_algo: RSB algo: 0 - Lanczos, 1 - RQI (Default: 0)
  //   rsb_pre: RSB pre-partition algo: 0 - None, 1 - RCB , 2 - RIB (Default: 1)
  //   rsb_max_iter: Maximum iterations in Lanczos or RQI (Default: 50)
  //   rsb_tol: Tolerance for Lanczos or RQI (Default: 1e-3)
  // - RSB-MG specific
  //   rsb_mg_grammian: MG Grammian: 0 or 1 (Default: 0)
  //   rsb_mg_factor: MG Coarsening factor (Default: 2, should be > 1)
  // - RSB-Lanczos specific
  //   rsb_lanczos_max_restarts: Maximum restarts in Lanczos (Default: 50)

  if (partitioner & 1) {
    options.partitioner = 0;
    options.rsb_algo = algo;
  } else if (partitioner & 2) {
    options.partitioner = 1;
  }

  if (*loglevel > 2)
    print_part_stat(vl, nel, nv, cext);

  ierr = parrsb_part_mesh(part, vl, xyz, tag, nel, nv, &options, comm.c);
  if (ierr != 0)
    goto err;

  ierr = redistribute_data(&nel, vl, el, part, nv, lelt, &comm);
  if (ierr != 0)
    goto err;

  // parrsb_check_tagged_partitions(el, vl, nel, nv, 10, &comm, 1);

  if (*loglevel > 2)
    print_part_stat(vl, nel, nv, cext);

#elif defined(PARMETIS)
  int metis = partitioner & 4;
  if (metis) {
    opt[0] = 1;
    opt[1] = 0; /* verbosity */
    opt[2] = comm.np;

    ierr = parMETIS_partMesh(part, vl, nel, nv, opt, comm.c);
    ierr = redistribute_data(&nel, vl, el, part, NULL, nv, lelt ,&comm);
    if (ierr != 0)
      goto err;
  }
#endif
  free(part);

  *nell = nel;
  *rtval = 0;
  return;

err:
  fflush(stdout);
  *rtval = 1;
}

#define fprintpartstat FORTRAN_UNPREFIXED(printpartstat,PRINTPARTSTAT)
void fprintpartstat(long long *vtx, int *nel, int *nv, int *comm)
{

#if defined(MPI)
  comm_ext c = MPI_Comm_f2c(*comm);
#else
  comm_ext c = 0;
#endif

  print_part_stat(vtx, *nel, *nv, c);
}
