#include <cassert>

#include "gemv.h"
#include "lapacke.h"

#include "localSolver.hpp"

template <typename val_t>
class AlgorithmGemv_t : public AlgorithmInterface_t<val_t> {
  using Vec_t = std::vector<val_t>;

public:
  AlgorithmGemv_t();

  void Setup(const VecIdx_t &row_offsets, const VecIdx_t &col_indices,
             const VecDouble_t &values, const std::string &backend,
             const int device_id) override;

  void Solve(Vec_t &x, const Vec_t &rhs) override;

  ~AlgorithmGemv_t();

private:
  struct gemv_t *gemv;
  size_t         size;
  void          *d_r, *d_x;
};

template <typename val_t> AlgorithmGemv_t<val_t>::AlgorithmGemv_t() {
  gemv = nullptr;
}

template <typename val_t>
void AlgorithmGemv_t<val_t>::Setup(const VecIdx_t    &row_offsets,
                                   const VecIdx_t    &col_indices,
                                   const VecDouble_t &values,
                                   const std::string &backend,
                                   const int          device_id) {

  const size_t num_rows = row_offsets.size() - 1;
  size                  = sizeof(val_t) * num_rows;

  VecDouble_t A(num_rows * num_rows);
  for (uint i = 0; i < num_rows; i++) A[i] = 0;

  for (uint i = 0; i < num_rows; i++) {
    for (uint j = row_offsets[i], je = row_offsets[i + 1]; j < je; j++)
      A[i * num_rows + col_indices[j]] = values[j];
  }

  std::vector<lapack_int> pivots(num_rows);
  lapack_int info = LAPACKE_dgetrf(LAPACK_ROW_MAJOR, num_rows, num_rows,
                                   A.data(), num_rows, pivots.data());
  assert(info == 0);

  info = LAPACKE_dgetri(LAPACK_ROW_MAJOR, num_rows, A.data(), num_rows,
                        pivots.data());
  assert(info == 0);

  gemv = gemv_init(NULL, NULL);
  gemv_set_device(gemv, device_id);
  gemv_set_backend(gemv, backend.c_str());
  gemv_set_matrix(gemv, num_rows, num_rows, A.data());

  gemv_device_malloc(&d_r, size);
  gemv_device_malloc(&d_x, size);
}

template <typename val_t>
void AlgorithmGemv_t<val_t>::Solve(Vec_t &x, const Vec_t &rhs) {
  gemv_copy(d_r, rhs.data(), size, GEMV_H2D);
  gemv_run(d_x, d_r, gemv);
  gemv_copy(x.data(), d_x, size, GEMV_D2H);
}

template <typename val_t> AlgorithmGemv_t<val_t>::~AlgorithmGemv_t() {
  gemv_device_free(&d_r);
  gemv_device_free(&d_x);
  gemv_finalize(&gemv);
}
