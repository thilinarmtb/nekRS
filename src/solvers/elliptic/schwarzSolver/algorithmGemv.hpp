#include <cassert>

#include "gemv.h"
#include "lapacke.h"

#include "localSolver.hpp"

template <typename val_t>
class AlgorithmGemv_t : public AlgorithmInterface_t<val_t> {
public:
  AlgorithmGemv_t();

  void Setup(const uint num_rows, uint *row_offsets, uint *col_indices,
             double *values, const gs_dom dom, const std::string &backend,
             const int device_id) override;

  void Solve(val_t *x, const val_t *rhs) override;

  ~AlgorithmGemv_t();

private:
  struct gemv_t *gemv;
  gs_dom         dom;
  size_t         size;
  val_t         *h_r, *h_x;
  void          *d_r, *d_x;
};

template <typename val_t>
AlgorithmGemv_t<val_t>::AlgorithmGemv_t() { gemv = nullptr; }

template <typename val_t>
void AlgorithmGemv_t<val_t>::Setup(const uint num_rows, uint *row_offsets,
                                   uint *col_indices, double *values,
                                   const gs_dom       dom_,
                                   const std::string &backend,
                                   const int          device_id) {
  dom  = dom_;
  size = sizeof(val_t) * num_rows;

  double *A = new double[num_rows * num_rows];
  for (uint i = 0; i < num_rows; i++) A[i] = 0;

  for (uint i = 0; i < num_rows; i++) {
    for (uint j = row_offsets[i], je = row_offsets[i + 1]; j < je; j++)
      A[i * num_rows + col_indices[j]] = values[j];
  }

  lapack_int *pivots = new lapack_int[num_rows];
  lapack_int  info =
      LAPACKE_dgetrf(LAPACK_ROW_MAJOR, num_rows, num_rows, A, num_rows, pivots);
  assert(info == 0);

  info = LAPACKE_dgetri(LAPACK_ROW_MAJOR, num_rows, A, num_rows, pivots);
  assert(info == 0);

  gemv = gemv_init(NULL, NULL);
  gemv_set_device(gemv, device_id);
  gemv_set_backend(gemv, backend.c_str());
  gemv_set_matrix(gemv, num_rows, num_rows, A);

  h_r = new val_t[num_rows];
  h_x = new val_t[num_rows];
  gemv_device_malloc(&d_r, size);
  gemv_device_malloc(&d_x, size);

  delete[] A, pivots;
}

template <typename val_t>
void AlgorithmGemv_t<val_t>::Solve(val_t *x, const val_t *rhs) {
  gemv_copy(d_r, (void *)rhs, size, GEMV_H2D);
  gemv_run(d_x, d_r, gemv);
  gemv_copy(d_x, (void *)rhs, size, GEMV_H2D);
}

template <typename val_t>
AlgorithmGemv_t<val_t>::~AlgorithmGemv_t() {
  delete[] h_r, h_x;
  gemv_device_free(&d_r);
  gemv_device_free(&d_x);
  gemv_finalize(&gemv);
}
