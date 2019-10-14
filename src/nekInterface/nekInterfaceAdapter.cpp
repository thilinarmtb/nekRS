#include <unistd.h>
#include <fstream>
#include "nekrs.hpp"
#include "nekInterfaceAdapter.hpp"

nekdata_private nekData;
static int rank;
static setupAide *options; 

void nek_setic(void)
{
  int readRestartFile;
  options->getArgs("RESTART FROM FILE", readRestartFile);

  if (readRestartFile) {
    std::string str1;
    options->getArgs("RESTART FILE NAME", str1);
    std::string str2(str1.size(), '\0');
    std::replace_copy(str1.begin(), str1.end(), str2.begin(), '+', ' ');
    int len = str2.length();
    nek_restart((char *)str2.c_str(),len);
  }

  nek_setics();

  if (readRestartFile) {
    double startTime = *(nekData.time);
    options->setArgs("START TIME", to_string_f(startTime));
  }
}

void mkSIZE(int lx1, int lxd, int lelt, int lelg, int ldim, int lpmin, int ldimt) {
  //printf("generating SIZE file ... "); fflush(stdout);

  char line[BUFSIZ];
  char cmd[BUFSIZ];

  const char *cache_dir = getenv("NEKRS_CACHE_DIR");
  const char *nekrs_nek5000_dir = getenv("NEKRS_NEK5000_DIR");

  // Read and generate the new size file.
  sprintf(line,"%s/core/SIZE.template", nekrs_nek5000_dir);
  FILE *fp = fopen(line, "r");
  char *sizeFile, *curSizeFile;
  size_t result;

  if (fp) {
    fseek(fp, 0, SEEK_END);
    long length = ftell(fp);
    rewind(fp);
    // allocate actual length + some buffer
    sizeFile = (char *) calloc(length+500,sizeof(char));
    if(!sizeFile) {
      fprintf(stderr, "Error allocating space for SIZE file.\n");
      exit(EXIT_FAILURE);
    }
  } else {
    fprintf(stderr, "Error opening %s/core/SIZE.template!\n", nekrs_nek5000_dir);
    exit(EXIT_FAILURE);
  }

  int count = 0;
  while(fgets(line, BUFSIZ, fp) != NULL) {

    if(strstr(line, "parameter (lx1=") != NULL) {
      sprintf(line, "      parameter (lx1=%d)\n", lx1);
    } else if(strstr(line, "parameter (lxd=") != NULL) {
      sprintf(line, "      parameter (lxd=%d)\n", lxd);
    } else if(strstr(line, "parameter (lelt=") != NULL) {
      sprintf(line, "      parameter (lelt=%d)\n", lelt);
    } else if(strstr(line, "parameter (lelg=") != NULL) {
      sprintf(line, "      parameter (lelg=%ld)\n", lelg);
    } else if(strstr(line, "parameter (ldim=") != NULL) {
      sprintf(line, "      parameter (ldim=%d)\n", ldim);
    } else if(strstr(line, "parameter (lpmin=") != NULL) {
      sprintf(line, "      parameter (lpmin=%d)\n", lpmin);
    } else if(strstr(line, "parameter (ldimt=") != NULL) {
      sprintf(line, "      parameter (ldimt=%d)\n", ldimt);
    } else if(strstr(line, "parameter (mxprev=") != NULL) {
      sprintf(line, "      parameter (mxprev=%d)\n", 1);
    } else if(strstr(line, "parameter (lgmres=") != NULL) {
      sprintf(line, "      parameter (lgmres=%d)\n", 1);
    } else if(strstr(line, "parameter (lorder=") != NULL) {
      sprintf(line, "      parameter (lorder=%d)\n", 1);
    } else if(strstr(line, "parameter (lelr=") != NULL) {
      sprintf(line, "      parameter (lelr=%d)\n", 128*lelt);
    }

    strcpy(sizeFile + count, line);
    count += strlen(line);
  }
  fclose(fp);

  int writeSize = 1;

  // read size if exists
  std::ifstream osize;
  sprintf(line,"%s/SIZE", cache_dir);

  osize.open(line, std::ifstream::in); 
  if(osize.is_open()) {
    writeSize = 0;
    string line;
    int oldval;
    while(getline( osize, line )) {
      if(line.find( "lelg=") != string::npos ) {
          sscanf(line.c_str(), "%*[^=]=%d", &oldval);
          if(oldval < lelg) writeSize = 1; 
      }
      if(line.find( "lelt=") != string::npos ) {
          sscanf(line.c_str(), "%*[^=]=%d", &oldval);
          if(oldval < lelt) writeSize = 1; 
      }
      if(line.find( "lx1=") != string::npos ) {
          sscanf(line.c_str(), "%*[^=]=%d", &oldval);
          if(oldval != lx1) writeSize = 1;
      }
      if(line.find( "ldimt=") != string::npos ) {
          sscanf(line.c_str(), "%*[^=]=%d", &oldval);
          if(oldval < ldimt) writeSize = 1;
      }
    }
  }
  osize.close();

  if(writeSize) {
    fp = fopen(line, "w");
    fputs(sizeFile, fp);
    fclose(fp);
    free(sizeFile);
    //printf("done\n");
  } else {
    //printf("using existing SIZE file %s/SIZE\n", cache_dir);
  }

  fflush(stdout);
}

int buildNekInterface(const char *casename, int ldimt, int N, int np) {
  printf("building nek ... "); fflush(stdout);

  char buf[BUFSIZ];
  char fflags[BUFSIZ];
  char cflags[BUFSIZ];

  const char *cache_dir = getenv("NEKRS_CACHE_DIR");
  const char *nekInterface_dir = getenv("NEKRS_NEKINTERFACE_DIR");
  const char *nek5000_dir = getenv("NEKRS_NEK5000_DIR");

  FILE *fp;
  int retval;

  sprintf(buf, "%s.re2", casename);
  fp = fopen(buf, "r");
  if (!fp) {
    printf("\nERROR: Cannot find %s!\n", buf);
    exit(EXIT_FAILURE);;
  }
  fgets(buf, 80, fp);
  fclose(fp);

  char ver[10];
  int nelgv, nelgt, ndim;
  sscanf(buf, "%5s %9d %1d %9d", ver, &nelgt, &ndim, &nelgv);
  int lelt = nelgt/np + 2;
  mkSIZE(N+1, 1, lelt, nelgt, ndim, np, ldimt);

  // Copy case.usr file to cache_dir
  sprintf(buf,"%s.usr",casename);
  if(access(buf,F_OK)!=-1){
    sprintf(buf, "cp -pf %s.usr %s/ >build.log 2>&1",casename,cache_dir);
  } else {
    sprintf(buf, "cp -pf %s/core/zero.usr %s/%s.usr >>build.log 2>&1",nek5000_dir,cache_dir,casename);
  }
  retval=system(buf);
  if (retval) goto err;


  // Copy Nek5000/core from install_dir to cache_dir
  sprintf(buf, "cp -pr %s %s/", nek5000_dir, cache_dir);
  retval = system(buf);
  if (retval) goto err; 

  //TODO: Fix hardwired compiler flags 
  sprintf(fflags, "\"${NEKRS_FFLAGS} -mcmodel=medium -fPIC -fcray-pointer -I../ \"");
  sprintf(cflags, "\"${NEKRS_CXXFLAGS} -fPIC -I${NEKRS_NEKINTERFACE_DIR}\""); 

  sprintf(buf, "cd %s && FC=\"${NEKRS_FC}\" CC=\"${NEKRS_CC}\" FFLAGS=%s "
      "CFLAGS=%s PPLIST=\"${NEKRS_NEK5000_PPLIST}\" NEK_SOURCE_ROOT=%s/nek5000 "
      "%s/nek5000/bin/nekconfig %s >>build.log 2>&1", cache_dir, fflags,
      cflags, cache_dir, cache_dir, casename);
  retval = system(buf);
  if (retval) goto err; 
  sprintf(buf, "cd %s && NEKRS_WORKING_DIR=%s make -j4 -f %s/Makefile nekInterface "
      ">>build.log 2>&1", cache_dir, cache_dir, nekInterface_dir);
  retval = system(buf);
  if (retval) goto err; 

  printf("done\n\n"); 
  fflush(stdout);
  sync();
  return 0;

err:
  printf("\nAn ERROR occured, see %s/build.log for details!\n", cache_dir);
  exit(EXIT_FAILURE);
}

int nekInterfaceAdapterSetup(MPI_Comm c, setupAide &options_in) {
  options = &options_in;
  MPI_Comm_rank(c,&rank);
  MPI_Fint nek_comm = MPI_Comm_c2f(c);

  string casename;
  options->getArgs("CASENAME", casename);

  char buf[FILENAME_MAX];
  getcwd(buf, sizeof(buf));
  string cwd;
  cwd.assign(buf);

  int nscal = 0;
  options->getArgs("NUMBER OF SCALARS", nscal);

  nek_setup(nek_comm,(char *)cwd.c_str(),(char *)casename.c_str(),nscal);

  nekData.param = (double *) nek_ptr("param");
  nekData.ifield = (int *) nek_ptr("ifield");
  nekData.istep = (int *) nek_ptr("istep");
  nekData.time = (double *) nek_ptr("time");

  nekData.ndim = *(int *) nek_ptr("ndim");
  nekData.nelt = *(int *) nek_ptr("nelt");
  nekData.nelv = *(int *) nek_ptr("nelv");
  nekData.lelt = *(int *) nek_ptr("lelt");
  nekData.nx1 =  *(int *) nek_ptr("nx1");

  nekData.vx = (double *) nek_ptr("vx");
  nekData.vy = (double *) nek_ptr("vy"); 
  nekData.vz = (double *) nek_ptr("vz");
  nekData.pr = (double *) nek_ptr("pr");
  nekData.t  = (double *) nek_ptr("t");

  nekData.ifgetu = (int *) nek_ptr("ifgetu");
  nekData.ifgetp = (int *) nek_ptr("ifgetp");

  nekData.unx = (double *) nek_ptr("unx");
  nekData.uny = (double *) nek_ptr("uny"); 
  nekData.unz = (double *) nek_ptr("unz");
 
  nekData.xm1 = (double *) nek_ptr("xm1");
  nekData.ym1 = (double *) nek_ptr("ym1"); 
  nekData.zm1 = (double *) nek_ptr("zm1");
  nekData.xc = (double *) nek_ptr("xc");
  nekData.yc = (double *) nek_ptr("yc"); 
  nekData.zc = (double *) nek_ptr("zc");

  nekData.ngv = *(long long *) nek_ptr("ngv");
  nekData.glo_num = (long long *) nek_ptr("glo_num");
  nekData.cbscnrs = (double *) nek_ptr("cb_scnrs");
  nekData.cbc = (char *) nek_ptr("cbc");
  nekData.boundaryID = (int *) nek_ptr("boundaryID");
  nekData.eface1 = (int *) nek_ptr("eface1");
  nekData.eface = (int *) nek_ptr("eface");
  nekData.icface = (int *) nek_ptr("icface");
  nekData.comm = MPI_Comm_f2c(*(int *) nek_ptr("nekcomm"));

  nekData.NboundaryIDs = nek_nbid(); 

  dfloat nu;
  options->getArgs("VISCOSITY", nu);
  nekData.param[1] = nu;

  options->getArgs("SCALAR01 DIFFUSIVITY", nu);
  nekData.param[7] = nu;

  return 0;
}

void nek_copyFrom(ins_t *ins, dfloat time) {

  mesh_t *mesh = ins->mesh;

  dlong Nlocal = mesh->Nelements*mesh->Np;

  dfloat *vx = ins->U + 0*ins->fieldOffset;
  dfloat *vy = ins->U + 1*ins->fieldOffset;
  dfloat *vz = ins->U + 2*ins->fieldOffset;

  *(nekData.time) = time;

  memcpy(nekData.vx, vx, sizeof(dfloat)*Nlocal);
  memcpy(nekData.vy, vy, sizeof(dfloat)*Nlocal);
  memcpy(nekData.vz, vz, sizeof(dfloat)*Nlocal);
  memcpy(nekData.pr, ins->P, sizeof(dfloat)*Nlocal);
  if(ins->Nscalar) memcpy(nekData.t, ins->cds->S, sizeof(dfloat)*Nlocal);
}

void nek_ocopyFrom(ins_t *ins, dfloat time, int tstep) {

  ins->o_U.copyTo(ins->U);
  ins->o_P.copyTo(ins->P); 
  if(ins->Nscalar) ins->cds->o_S.copyTo(ins->cds->S);
  nek_copyFrom(ins, time, tstep);
}

void nek_copyFrom(ins_t *ins, dfloat time, int tstep) {

  if(rank==0) {
    printf("copying solution to nek\n");
    fflush(stdout);
  }

  mesh_t *mesh = ins->mesh;

  dlong Nlocal = mesh->Nelements*mesh->Np;

  dfloat *vx = ins->U + 0*ins->fieldOffset;
  dfloat *vy = ins->U + 1*ins->fieldOffset;
  dfloat *vz = ins->U + 2*ins->fieldOffset;

  *(nekData.istep) = tstep;
  *(nekData.time) = time;

  memcpy(nekData.vx, vx, sizeof(dfloat)*Nlocal);
  memcpy(nekData.vy, vy, sizeof(dfloat)*Nlocal);
  memcpy(nekData.vz, vz, sizeof(dfloat)*Nlocal);
  memcpy(nekData.pr, ins->P, sizeof(dfloat)*Nlocal);
  if(ins->Nscalar)  memcpy(nekData.t, ins->cds->S, sizeof(dfloat)*Nlocal);
 
}

void nek_ocopyTo(ins_t *ins, dfloat &time) {

  nek_copyTo(ins, time);
  ins->o_P.copyFrom(ins->P);
  ins->o_U.copyFrom(ins->U);
  if(ins->Nscalar) ins->cds->o_S.copyFrom(ins->cds->S); 
}

void nek_copyTo(ins_t *ins, dfloat &time) {

  if(rank==0) {
    printf("copying solution from nek\n");
    fflush(stdout);
  }

  mesh_t *mesh = ins->mesh;

  time = *(nekData.time);

  dfloat *vx = ins->U + 0*ins->fieldOffset;
  dfloat *vy = ins->U + 1*ins->fieldOffset;
  dfloat *vz = ins->U + 2*ins->fieldOffset;

  dlong Nlocal = mesh->Nelements*mesh->Np;
  memcpy(vx, nekData.vx, sizeof(dfloat)*Nlocal);
  memcpy(vy, nekData.vy, sizeof(dfloat)*Nlocal);
  memcpy(vz, nekData.vz, sizeof(dfloat)*Nlocal);
  memcpy(ins->P, nekData.pr, sizeof(dfloat)*Nlocal);
  if(ins->Nscalar)  memcpy(ins->cds->S, nekData.t, sizeof(dfloat)*Nlocal);
}

void nek_copyRestart(ins_t *ins) {
  mesh_t *mesh = ins->mesh;
  dlong Nlocal = mesh->Nelements*mesh->Np;
  if (*(nekData.ifgetu)) {
    dfloat *vx = ins->U + 0*ins->fieldOffset;
    dfloat *vy = ins->U + 1*ins->fieldOffset;
    dfloat *vz = ins->U + 2*ins->fieldOffset;
    memcpy(vx, nekData.vx, sizeof(dfloat)*Nlocal);
    memcpy(vy, nekData.vy, sizeof(dfloat)*Nlocal);
    memcpy(vz, nekData.vz, sizeof(dfloat)*Nlocal);
  }
  if(ins->Nscalar) memcpy(ins->cds->S, nekData.t, sizeof(dfloat)*Nlocal);
  if (*(nekData.ifgetp)) memcpy(ins->P, nekData.pr, sizeof(dfloat)*Nlocal);
}

