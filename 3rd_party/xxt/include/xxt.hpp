#if !defined(_XXT_HPP_)
#define _XXT_HPP_

#if defined(__cplusplus)
extern "C" {
#endif

struct xxt;
struct xxt *crs_xxt_setup(uint n, const ulong *id, uint nz, const uint *Ai,
                          const uint *Aj, const double *A, uint null_space,
                          const struct comm *comm, gs_dom dom);
void crs_xxt_solve(void *x, struct xxt *data, const void *b);
void crs_xxt_stats(struct xxt *data);
void crs_xxt_times(double *cholesky, double *local, double *xxt, double *qqt);
void crs_xxt_free(struct xxt *data);

#if defined(__cplusplus)
}
#endif

#endif // _XXT_HPP_
