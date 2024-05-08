find_package(CUDAToolkit)

if (TARGET CUDA::toolkit)
  add_library(gemv::CUDA INTERFACE IMPORTED)
  target_link_libraries(gemv::CUDA INTERFACE CUDA::nvrtc CUDA::cudart
    CUDA::cuda_driver)

  if (TARGET CUDA::cublas)
    target_link_libraries(gemv::CUDA INTERFACE CUDA::cublas)
    if ("${GEMV_DEFAULT_BACKEND}" STREQUAL "")
      set(GEMV_DEFAULT_BACKEND "cublas")
    endif()
  endif()

  if ("${GEMV_DEFAULT_BACKEND}" STREQUAL "")
    set(GEMV_DEFAULT_BACKEND "cuda")
  endif()
endif()
