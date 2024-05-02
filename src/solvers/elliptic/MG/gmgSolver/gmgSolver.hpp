#if !defined(__SCHWARZ_SOLVER_HPP__)
#define __SCHWARZ_SOLVER_HPP__

#include "gmgTypes.hpp"

#include "localSolver.hpp"

template <typename val_t> class GMGSolver_t {
  using Vec_t = std::vector<val_t>;

public:
  GMGSolver_t();

  void Setup(const Long_t &vtx, const Double_t &amat, const Double_t &mask,
             const Int_t &frontier, const Algorithm_t &algo);

  void Solve(occa::memory &o_x, const occa::memory &o_rhs);

  ~GMGSolver_t();

private:
  void SetupCoarseAverage(const Long_t &vtx, const MPI_Comm comm);

  void SetupLocalSolver(const Long_t &vtx, const Double_t &va,
                        const Algorithm_t &algo, const std::string &backend,
                        const int device_id);

  void CoarseAverage(Vec_t &vec);

private:
  size_t                crs_size, user_size, shared_size;
  gs_dom                dom;
  Vec_t                 A, x, rhs, inv_mul;
  buffer                bfr;
  struct gs_data       *gsh;
  LocalSolver_t<val_t> *solver;
};

template class GMGSolver_t<float>;
template class GMGSolver_t<double>;

#endif // __SCHWARZ_SOLVER_HPP__
