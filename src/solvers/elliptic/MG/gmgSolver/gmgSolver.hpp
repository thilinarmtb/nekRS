#if !defined(__SCHWARZ_SOLVER_HPP__)
#define __SCHWARZ_SOLVER_HPP__

#include "gmgTypes.hpp"

#include "gmgLocalSolver.hpp"

template <typename val_t> class GMGSolver_t {
  using Vec_t = std::vector<val_t>;

public:
  GMGSolver_t(const VecLong_t &Aids, const VecUInt_t &Ai, const VecUInt_t &Aj,
              const VecDouble_t &Av);

  void Solve(occa::memory &o_x, const occa::memory &o_rhs);

  ~GMGSolver_t();

private:
  void SetupCoarseAverage(const VecLong_t &vtx, const MPI_Comm comm);

  void CoarseAverage(Vec_t &vec);

private:
  size_t                   crs_size, user_size, shared_size;
  gs_dom                   dom;
  Vec_t                    A, x, rhs, inv_mul;
  buffer                   bfr;
  struct gs_data          *gsh;
  GMGLocalSolver_t<val_t> *solver;
};

template class GMGSolver_t<float>;
template class GMGSolver_t<double>;

#endif // __SCHWARZ_SOLVER_HPP__
