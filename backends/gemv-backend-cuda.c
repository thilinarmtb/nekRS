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

static CUmodule module = NULL;
static CUfunction kernel = NULL;

static int block_size = 32;
static int kernel_id = 1;

#include "gemv-backend-unified-cuda-hip-kernels.h"

static void cuda_run(void *d_y, const void *d_x, const struct gemv_t *gemv) {
  if (!initialized)
    gemv_log(GEMV_ERROR, "cuda_run: CUDA backend is not initialized !");

  gemv_log(GEMV_INFO, "y = %p, x = %p, m = %u, n = %u", d_y, d_x, gemv->m,
           gemv->n);

  void *arguments[] = {&d_y, &d_A, &d_x, (void *)&gemv->m, (void *)&gemv->n};
  check_cuda_runtime(cuLaunchKernel(kernel, (gemv->m + 31) / 32, 1, 1, 32, 1, 1,
                                    0, 0, arguments, NULL));
}

static void cuda_finalize(void) {
  gemv_log(GEMV_INFO, "cuda_finalize: ...");
  if (!initialized) return;

  check_cuda_runtime(cudaFree(d_A)), d_A = NULL;
  check_cuda_runtime(cuModuleUnload(module)), module = NULL, kernel = NULL;
  initialized = 0;

  gemv_log(GEMV_INFO, "cuda_finalize: done.");
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

  char source[BUFSIZ];
  const char *precision = gemv_precision_to_str(gemv->precision);

  switch (kernel_id) {
  case 0:
    snprintf(source, BUFSIZ, gemv_kernels[0], precision, precision, precision,
             precision);
    break;
  case 1:
    snprintf(source, BUFSIZ, gemv_kernels[1], block_size, precision, precision,
             precision, precision);
    break;
  default:
    gemv_log(GEMV_ERROR, "cuda_init_aux: invalid kernel id = %d", kernel_id);
    break;
  }
  gemv_log(GEMV_INFO, "cuda_init_aux: kernel id = %d, source = \n%s", kernel_id,
           source);

  nvrtcProgram program = NULL;
  check_cuda_rtc(nvrtcCreateProgram(&program, source, NULL, 0, NULL, NULL));
  nvrtcResult status = nvrtcCompileProgram(program, 0, NULL);
  if (status == NVRTC_SUCCESS) goto generate_kernel;

  size_t log_size;
  check_cuda_rtc(nvrtcGetProgramLogSize(program, &log_size));
  char *log = gemv_calloc(char, log_size + 1);
  check_cuda_rtc(nvrtcGetProgramLog(program, log));
  fprintf(stderr, "[ERROR] cuda_init_aux: kernel compilation error:\n%s\n",
          log);
  fflush(stderr);
  gemv_free(&log);
  gemv_log(GEMV_ERROR, "cuda_init_aux: kernel compilation failed !");

  size_t size;
generate_kernel:
  check_cuda_rtc(nvrtcGetPTXSize(program, &size));
  char *binary_data = gemv_calloc(char, size + 1);
  check_cuda_rtc(nvrtcGetPTX(program, binary_data));
  check_cuda_rtc(nvrtcDestroyProgram(&program));

  check_cuda_runtime(cuModuleLoadData(&module, binary_data));
  gemv_free(&binary_data);
  check_cuda_runtime(cuModuleGetFunction(&kernel, module, "gemv"));
}

static void cuda_init(struct gemv_backend_t *backend,
                      const struct gemv_t *gemv) {
  gemv_log(GEMV_INFO, "cuda_init: ...");
  if (initialized) return;

  backend->malloc = cuda_malloc;
  backend->free = cuda_free;
  backend->copy = cuda_copy;
  backend->run = cuda_run;
  backend->finalize = cuda_finalize;

  cuda_init_aux(gemv);

  initialized = 1;
  gemv_log(GEMV_INFO, "cuda_init: done.");
}

void gemv_register_cuda(void) { gemv_backend_register("cuda", cuda_init); }
