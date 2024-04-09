#include "backends/gemv-backend-cuda.h"

#include <nvrtc.h>

static inline void check_cuda_rtc_(nvrtcResult status, const char *file,
                                   const unsigned line) {
  if (status == NVRTC_SUCCESS) return;
  const char *error = nvrtcGetErrorString(status);
  gemv_log(GEMV_ERROR, "NVRTC error: \"%s\" in file: %s line: %u", error, file,
           line);
}

#define check_cuda_rtc(call) check_cuda_rtc_((call), __FILE__, __LINE__)

static void *d_A = NULL;
static int initialized = 0;

static void cuda_gemv(void *d_y, const void *d_x, const struct gemv_t *gemv) {}

static void cuda_finalize(void) {
  check_cuda_runtime(cudaFree(d_A)), d_A = NULL;
}

static void cuda_init_aux(const struct gemv_t *gemv) {
  check_cuda_runtime(cudaSetDevice(gemv->device));

  const unsigned n = gemv->n, m = gemv->m;
  check_cuda_runtime(cudaMalloc((void **)&d_A, n * m * sizeof(double)));

  const size_t unit_size = gemv_unit_size(gemv->precision);
  void *const A = gemv_malloc(m * n * unit_size);
  gemv_convert(A, gemv->A, m * n, gemv->precision);

  check_cuda_runtime(
      cudaMemcpy(d_A, A, m * n * unit_size, cudaMemcpyHostToDevice));
  gemv_free(&A);

  // char source[BUFSIZ];
  // const char *precision = gemv_precision_to_str(gemv->precision);
}

static void cuda_init(struct gemv_backend_t *backend,
                      const struct gemv_t *gemv) {
  gemv_log(GEMV_INFO, "cuda_init: ...");
  if (initialized) return;

  // TODO: Set backend functions.

  cuda_init_aux(gemv);

  initialized = 1;
  gemv_log(GEMV_INFO, "hip_init: done.");
}

void gemv_register_cuda(void) { gemv_backend_register("cuda", cuda_init); }
