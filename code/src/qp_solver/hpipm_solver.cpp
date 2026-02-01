#include <qp_solver/hpipm_solver.h>

#include <cstdlib>

extern "C" {
#include <hpipm_d_ocp_qp.h>
#include <hpipm_d_ocp_qp_dim.h>
#include <hpipm_d_ocp_qp_ipm.h>
#include <hpipm_d_ocp_qp_sol.h>
}

/**
 * @file hpipm_solver.cpp
 * @brief Minimal implementation (only memory management)
 *
 * All performance-critical methods are inline in the header.
 * This file only contains constructor, destructor, and initialization.
 */

HPIPMSolver::HPIPMSolver()
    : _N(0),
      _nx(0),
      _nu(0),
      _ng(0),
      _initialized(false),
      _dim_mem(nullptr),
      _qp_mem(nullptr),
      _sol_mem(nullptr),
      _arg_mem(nullptr),
      _ws_mem(nullptr) {}

void HPIPMSolver::initialize(int N, int nx, int nu, int ng, int nbx, int nbu) {
    _N = N;
    _nx = nx;
    _nu = nu;
    _ng = ng;

    // Allocate dimension structure
    int dim_size = d_ocp_qp_dim_memsize(N);
    _dim_mem = malloc(dim_size);
    d_ocp_qp_dim_create(N, &_qp_dim, _dim_mem);

    // Set dimensions for all stages
    // Note: HPIPM API is (stage, value, dim_struct)
    for (int k = 0; k <= N; k++) {
        d_ocp_qp_dim_set_nx(k, nx, &_qp_dim);
        if (k < N) {
            d_ocp_qp_dim_set_nu(k, nu, &_qp_dim);
        }
        if (ng > 0) {
            d_ocp_qp_dim_set_ng(k, ng, &_qp_dim);
        }
        if (nbx > 0) {
            d_ocp_qp_dim_set_nbx(k, nbx, &_qp_dim);
        }
        if (nbu > 0 && k < N) {
            d_ocp_qp_dim_set_nbu(k, nbu, &_qp_dim);
        }
    }

    // Allocate QP problem structure
    int qp_size = d_ocp_qp_memsize(&_qp_dim);
    _qp_mem = malloc(qp_size);
    d_ocp_qp_create(&_qp_dim, &_qp, _qp_mem);

    // Allocate solution structure
    int sol_size = d_ocp_qp_sol_memsize(&_qp_dim);
    _sol_mem = malloc(sol_size);
    d_ocp_qp_sol_create(&_qp_dim, &_qp_sol, _sol_mem);

    // Allocate IPM arguments
    int arg_size = d_ocp_qp_ipm_arg_memsize(&_qp_dim);
    _arg_mem = malloc(arg_size);
    d_ocp_qp_ipm_arg_create(&_qp_dim, &_qp_arg, _arg_mem);

    // Allocate workspace
    int ws_size = d_ocp_qp_ipm_ws_memsize(&_qp_dim, &_qp_arg);
    _ws_mem = malloc(ws_size);
    d_ocp_qp_ipm_ws_create(&_qp_dim, &_qp_arg, &_qp_ws, _ws_mem);

    _initialized = true;
}

HPIPMSolver::~HPIPMSolver() {
    // Free in reverse order of allocation
    if (_ws_mem) free(_ws_mem);
    if (_arg_mem) free(_arg_mem);
    if (_sol_mem) free(_sol_mem);
    if (_qp_mem) free(_qp_mem);
    if (_dim_mem) free(_dim_mem);
}
