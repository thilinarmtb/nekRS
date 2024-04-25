#if !defined(__SCHWARZ_SOLVER_IMPL_HPP__)
#define __SCHWARZ_SOLVER_IMPL_HPP__

#include "localSolver.hpp"
#include "schwarzSolver.hpp"

class SchwarzSolverInterface_t {
public:
  virtual void Setup(const Long_t &vtx, const Double_t &amat,
                     const Double_t &mask, const Int_t &frontier,
                     const size_t num_elements, const MPI_Comm comm,
                     const std::string &backend, const int device_id) = 0;

  virtual void Solve(occa::memory &o_x, const occa::memory &o_rhs) = 0;

  virtual ~SchwarzSolverInterface_t() = default;
};

template <typename val_t>
class SchwarzSolverImpl_t : public SchwarzSolverInterface_t {
  using Vec_t = std::vector<val_t>;

public:
  SchwarzSolverImpl_t(const size_t user_size, const size_t shared_size,
                      const size_t crs_size);

  void Setup(const Long_t &vtx, const Double_t &amat, const Double_t &mask,
             const Int_t &frontier, const size_t num_elements,
             const MPI_Comm comm, const std::string &backend,
             const int device_id) override;

  void Solve(occa::memory &o_x, const occa::memory &o_rhs) override;

  ~SchwarzSolverImpl_t();

private:
  void SetupLocalSolver(const Long_t &vtx, const Double_t &va,
                        const std::string &backend, const int device_id);

  void SetupCoarseAverage(const Long_t &vtx, const MPI_Comm comm);

  void CoarseAverage(Vec_t &vec);

private:
  size_t                user_size, shared_size, crs_size;
  gs_dom                dom;
  Vec_t                 A, x, rhs, inv_mul;
  buffer                bfr;
  struct gs_data       *gsh;
  LocalSolver_t<val_t> *local_solver;
};

template class SchwarzSolverImpl_t<float>;
template class SchwarzSolverImpl_t<double>;

#endif // __SCHWARZ_SOLVER_IMPL_HPP__
