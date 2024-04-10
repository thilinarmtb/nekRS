#include "backends/gemv-backend-cuda.h"

#include <cublas_v2.h>

static inline void check_cublas_(cublasStatus_t status, const char *file,
                                 const unsigned line) {
#define add_case(A)                                                            \
  case A: error = #A; break

  char *error = NULL;
  // clang-format off
  switch (status) {
  case CUBLAS_STATUS_SUCCESS: return; break;
  add_case(CUBLAS_STATUS_NOT_INITIALIZED);
  add_case(CUBLAS_STATUS_ALLOC_FAILED);
  add_case(CUBLAS_STATUS_INVALID_VALUE);
  add_case(CUBLAS_STATUS_ARCH_MISMATCH);
  add_case(CUBLAS_STATUS_MAPPING_ERROR);
  add_case(CUBLAS_STATUS_EXECUTION_FAILED);
  add_case(CUBLAS_STATUS_INTERNAL_ERROR);
  add_case(CUBLAS_STATUS_NOT_SUPPORTED);
  add_case(CUBLAS_STATUS_LICENSE_ERROR);
  default: break;
  }
  // clang-format on

#undef add_case

  gemv_log(GEMV_ERROR, "cuBLAS error: %s in file: \"%s\" line: %u", error, file,
           line);
}

#define check_cublas(call) check_cublas_(call, __FILE__, __LINE__)

static cublasHandle_t handle = NULL;
static void *d_A = NULL;
static int initialized = 0;

static void cublas_run(void *d_y, const void *d_x, const struct gemv_t *gemv) {
  if (!initialized)
    gemv_log(GEMV_ERROR, "cublas_run: cuBLAS backend is not initialized !");

  gemv_log(GEMV_INFO, "y = %p, x = %p, m = %u, n = %u", d_y, d_x, gemv->m,
           gemv->n);

  float alpha_f = 1.0f, beta_f = 0.0f;
  double alpha_d = 1.0, beta_d = 0.0;
  switch (gemv->precision) {
  case GEMV_FP32:
    check_cublas(cublasSgemv(handle, CUBLAS_OP_T, gemv->m, gemv->n, &alpha_f,
                             d_A, gemv->m, d_x, 1, &beta_f, d_y, 1));
    gemv_log(GEMV_INFO, "cublas_run: cublasSgemv, done.");
    break;
  case GEMV_FP64:
    check_cublas(cublasDgemv(handle, CUBLAS_OP_T, gemv->m, gemv->n, &alpha_d,
                             d_A, gemv->m, d_x, 1, &beta_d, d_y, 1));
    gemv_log(GEMV_INFO, "cublas_run: cublasDgemv, done.");
    break;
  default: break;
  }
}

static void cublas_finalize(void) {
  gemv_log(GEMV_INFO, "cublas_finalize: ...");
  if (!initialized) return;

  check_cuda_runtime(cudaFree(d_A)), d_A = NULL;
  check_cublas(cublasDestroy(handle)), handle = NULL;
  initialized = 0;

  gemv_log(GEMV_INFO, "cublas_finalize: done.");
}

static void cublas_init_aux(const struct gemv_t *gemv) {
  check_cuda_runtime(cudaSetDevice(gemv->device));

  const unsigned m = gemv->m, n = gemv->n;
  check_cuda_runtime(cudaMalloc((void **)&d_A, m * n * sizeof(double)));

  const size_t unit_size = gemv_unit_size(gemv->precision);
  gemv_log(GEMV_INFO, "cublas_init_aux: unit_size = %zu", unit_size);

  void *const A = gemv_malloc(m * n * unit_size);
  gemv_convert(A, gemv->A, m * n, gemv->precision);

  check_cuda_runtime(
      cudaMemcpy(d_A, A, m * n * unit_size, cudaMemcpyHostToDevice));
  gemv_free(&A);

  check_cublas(cublasCreate(&handle));
}

static void cublas_init(struct gemv_backend_t *backend,
                        const struct gemv_t *gemv) {
  gemv_log(GEMV_INFO, "cublas_init: ...", initialized);
  if (initialized) return;

  backend->malloc = cuda_malloc;
  backend->free = cuda_free;
  backend->copy = cuda_copy;
  backend->run = cublas_run;
  backend->finalize = cublas_finalize;

  cublas_init_aux(gemv);

  initialized = 1;
  gemv_log(GEMV_INFO, "cublas_init: done.");
}

void gemv_register_cublas(void) {
  gemv_backend_register("cublas", cublas_init);
}
