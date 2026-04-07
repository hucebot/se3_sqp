#include <sqp_solver/sqp_solver.h>
#include <common/profiling.h>

#include <cmath>
#include <iostream>

SQPSolver::SQPSolver(OCP& ocp): _ocproblem(ocp) { init(); }

SQPSolver::~SQPSolver() {
    // HPIPM memory automatically freed by HPIPMSolver destructor
}

void SQPSolver::set_options(const SQPoptions& opts) {
    _opts = opts;
    switch (_opts.ls_type) {
        case LSType::MERIT:  _ls_function = &SQPSolver::ls_merit;  break;
        case LSType::FILTER: _ls_function = &SQPSolver::ls_filter; break;
        case LSType::NONE:   _ls_function = nullptr;               break;
    }
    _qp_solver.set_iter_max(_opts.hpipm_iter_max);
    _qp_solver.set_tol(_opts.hpipm_tol_stat, _opts.hpipm_tol_eq,
                       _opts.hpipm_tol_ineq, _opts.hpipm_tol_comp);
    _qp_solver.set_warm_start(_opts.hpipm_warm_start);
    _opts.print();
}

void SQPSolver::solve(const Eigen::VectorXd& x0) {
    _stats.reset();
    _stats.start_timer();
    _current_reg = _opts.regularization;

    Eigen::VectorXd dx(2*_ocproblem.get_node(0).model().nv);
    dx.setZero();

    for (int i = 0; i < _opts.max_sqp_iters; i++) {
        PROFILE_DECLARE(iter_linearize_ms);
        PROFILE_DECLARE(iter_linesearch_ms);

        { PROFILE_SCOPE(iter_linearize_ms);
            linearize();

            if(x0.size() > 0)
            {
                pinocchio::difference(_ocproblem.get_node(0).model(), _ocproblem.x_traj()[0].segment(0, _ocproblem.get_node(0).model().nq), x0.segment(0, _ocproblem.get_node(0).model().nq), dx.segment(0, _ocproblem.get_node(0).model().nv));
                dx.segment(_ocproblem.get_node(0).model().nv, _ocproblem.get_node(0).model().nv) = x0.segment(_ocproblem.get_node(0).model().nq, _ocproblem.get_node(0).model().nv) - _ocproblem.x_traj()[0].segment(_ocproblem.get_node(0).model().nq, _ocproblem.get_node(0).model().nv);

                _b[0] = _A[0] * dx + _b[0];
                _r[0] = _S[0] * dx + _r[0];
            }
        }
        populate_qp();
        _qp_solver.solve();
        _stats.update_qp_info(_qp_solver.get_status(), _qp_solver.get_iter());

        if(x0.size() > 0)
        {
            _dx[0] = dx;
        }

        // Backtracking line search
        {
            PROFILE_SCOPE(iter_linesearch_ms);
            _ls_alpha = 1.;
            step();
            for (int ls_iter = 1; (ls_iter <= _opts.max_ls_iters) && _ls_function; ++ls_iter) {
                _stats.update_linesearch_iterations(ls_iter);
                if ((this->*_ls_function)()) break;
                _ls_alpha *= _opts.ls_scale_factor;
                step();
            }
        }

        // Adaptive regularization: scale up when line search took minimum step, reset otherwise
        if (_ls_function && _ls_alpha < std::pow(_opts.ls_scale_factor, _opts.max_ls_iters - 1) + 1e-12) {
            _current_reg *= _opts.regularization_scale;
            DEBUG_PRINT("reg: "<< _current_reg);
        } else {
            accept_step();
            _current_reg = _opts.regularization;
        }



        // When no line search ran,  _candidate_* were not populated by ls_filter().
        // Compute once here so stats and  _nominal_* are always up to date.
        if (!_ls_function) {
             _candidate_cost   = _ocproblem.cost();
             _candidate_viol   = _ocproblem.constraint_violation();
             _candidate_defect = _ocproblem.dynamics_defect();
        }

        _stats.update();
        _stats.update_cost( _candidate_cost);
        _stats.update_constraint_violation( _candidate_viol);
        _stats.update_dynamics_defect( _candidate_defect);
        

        PROFILE_PRINT("  Lin", iter_linearize_ms);
        PROFILE_PRINT("  LS", iter_linesearch_ms);
        if (break_criteria()) break;
        if (_opts.verbose == 1) _stats.print(0);
    }
    if (_opts.verbose == 1) _stats.print(2);
    else if (_opts.verbose == 2) _stats.print(1);
}

void SQPSolver::init() {
    _N = _ocproblem.num_nodes();
    _ls_alpha = 1.0;
    _current_reg = _opts.regularization;

    _nominal_cost = 0.;
    _nominal_defect = 0.;
    _nominal_viol = 0.;

    // Set line search function pointer based on options
    switch (_opts.ls_type) {
        case LSType::MERIT:  _ls_function = &SQPSolver::ls_merit;  break;
        case LSType::FILTER: _ls_function = &SQPSolver::ls_filter; break;
        case LSType::NONE:   _ls_function = nullptr;               break;
    }

    // _opts = set_options(_opts.default());

    _Nx = _N;
    _Nu = _N-1;

    // Linearization storage - resize vectors and pre-allocate Eigen matrices
    // Dynamics: N-1 transitions (stages 0 to N-2)
    _A.resize(_Nu);
    _B.resize(_Nu);
    _b.resize(_Nu);
    // Costs:
    _Q.resize(_Nx);   //state
    _q.resize(_Nx);
    _R.resize(_Nu); //control
    _r.resize(_Nu);
    _S.resize(_Nu); //cross term
    // Constraints:
    _C.resize(_Nx);   //state
    _D.resize(_Nu); //control
    _lg.resize(_Nx);
    _ug.resize(_Nx);
    // Step storage (reused every iteration)
    _dx.resize(_Nx);
    _du.resize(_Nu);
    // Lagrange multipliers
    _pi.resize(_Nu);
    _lam_lg.resize(_Nx);
    _lam_ug.resize(_Nx);
    _pi_qp.resize(_Nu);
    _lam_lg_qp.resize(_Nx);
    _lam_ug_qp.resize(_Nx);
    _pi_candidate.resize(_Nu);
    _lam_lg_candidate.resize(_Nx);
    _lam_ug_candidate.resize(_Nx);
    _scaled_dx.resize(_Nx);
    _scaled_du.resize(_Nu);
    // Trajectory storage
    _x_candidate.resize(_Nx);
    _u_candidate.resize(_Nu);

    // std::cout<<"initing hpipm"<<std::endl;
    _qp_solver.initialize(_N);    
    // std::cout<<"inited hpipm"<<std::endl;

    // std::cout<<"initing hpipm stages"<<std::endl;
    for (int k = 0; k<_N; ++k )
    {
        // std::cout<<"hpipm_stage:"<<k<<std::endl;
        _nx = _ocproblem.get_node(k).nx();
        _nu = _ocproblem.get_node(k).nu();
        _ndx = _ocproblem.get_node(k).ndx();
        _ndu = _ocproblem.get_node(k).ndu();

        _ng = 0;
        for (auto& con : _ocproblem.get_node(k).get_constraints())
            _ng += con->get_output_dim();

        // Pre-allocate and zero dynamics matrices (N-1 transitions)
        if (k<_Nu){
            _A[k].setZero(_ndx, _ndx);
            _B[k].setZero(_ndx, _ndu);
            _b[k].setZero(_ndx);
        }

        // Pre-allocate and zero cost matrices (N stages)
        _Q[k].setIdentity(_ndx, _ndx);  // Default: identity cost on state
        _q[k].setZero(_ndx);
        if (k<_Nu){
            _R[k].setIdentity(_ndu, _ndu);  // Default: identity cost on control
            _r[k].setZero(_ndu);
            _S[k].setZero(_ndu, _ndx);      // Cross term
        }

        // Pre-allocate constraint matrices (ng x ndx/ndu) per stage
        _C[k].setZero(_ng, _ndx);
        if (k<_Nu) _D[k].setZero(_ng, _ndu);
        _lg[k].setZero(_ng);
        _ug[k].setZero(_ng);

        _dx[k].setZero(_ndx);
        if (k<_Nu) _du[k].setZero(_ndu);
        if (k<_Nu) _pi[k].setZero(_ndx);
        _lam_lg[k].setZero(_ng);
        _lam_ug[k].setZero(_ng);
        if (k<_Nu) _pi_qp[k].setZero(_ndx);
        _lam_lg_qp[k].setZero(_ng);
        _lam_ug_qp[k].setZero(_ng);
        if (k<_Nu) _pi_candidate[k].setZero(_ndx);
        _lam_lg_candidate[k].setZero(_ng);
        _lam_ug_candidate[k].setZero(_ng);
        _scaled_dx[k].setZero(_ndx);
        if (k<_Nu) _scaled_du[k].setZero(_ndu);

        _x_candidate[k].setZero(_nx);
        if (k<_Nu) _u_candidate[k].setZero(_nu);

        if (k==0) //
        {
            _qp_solver.init_stage(k, _ndx, _ndu, _ng, _ndx);
            // Initial state constraint: dx_0 = 0 (all state variables constrained)
            _idxbx.resize(_ndx);
            for (int i = 0; i < _ndx; ++i) _idxbx[i] = i;
            _lbx.setZero(_ndx);
            _ubx.setZero(_ndx);
        }
        else if (k<_Nu)
            _qp_solver.init_stage(k, _ndx, _ndu, _ng);
        else
            _qp_solver.init_stage(k, _ndx, 0, _ng);
    } 


    // std::cout<<"allocating hpipm"<<std::endl;
    _qp_solver.allocate();
    // std::cout<<"allocated hpipm"<<std::endl;


    // Initial state constraint: dx_0 = 0 (box constraint at stage 0)
    _qp_solver.set_idxbx(0, _idxbx.data());
    _qp_solver.set_lbx(0, _lbx.data());
    _qp_solver.set_ubx(0, _ubx.data());


    _qp_solver.set_iter_max(_opts.hpipm_iter_max);
    _qp_solver.set_tol(_opts.hpipm_tol_stat, _opts.hpipm_tol_eq,
                       _opts.hpipm_tol_ineq, _opts.hpipm_tol_comp);
    _qp_solver.set_warm_start(_opts.hpipm_warm_start);

    // std::cout<<"inited_sqp"<<std::endl;
}

void SQPSolver::populate_qp() {
    // Transfer linearization data to HPIPM

    for (int k = 0; k < _N; ++k) {
        if (k<_Nu){
            _qp_solver.set_A(k, _A[k].data());
            _qp_solver.set_B(k, _B[k].data());
            _qp_solver.set_b(k, _b[k].data());    
        }

        _qp_solver.set_Q(k, _Q[k].data());
        _qp_solver.set_q(k, _q[k].data());

        if (k<_Nu){
            _qp_solver.set_R(k, _R[k].data());
            _qp_solver.set_r(k, _r[k].data());
            _qp_solver.set_S(k, _S[k].data());
        }

        _qp_solver.set_C(k, _C[k].data());
        if (k< _Nu){
            _qp_solver.set_D(k, _D[k].data());
        }

        _qp_solver.set_lg(k, _lg[k].data());
        _qp_solver.set_ug(k, _ug[k].data());

    }
}

void SQPSolver::step() {
    // Compute candidate trajectory from QP solution at current _ls_alpha.
    // After this call, nodes are bound to _x_candidate/_u_candidate.
    _step_norm = 0.0;

    auto& x_nom = _ocproblem.x_traj();
    auto& u_nom = _ocproblem.u_traj();

    for (int k = 0; k < _N; ++k) {
        // DEBUG_PRINT(k<<" x");
        _qp_solver.get_x(k, _dx[k].data());
        _scaled_dx[k].noalias() = _ls_alpha * _dx[k];
        _step_norm = std::max(_step_norm, _scaled_dx[k].cwiseAbs().maxCoeff());
        _ocproblem.get_node(k).x_oplus(x_nom[k], _scaled_dx[k], _x_candidate[k]);

        if (k < _Nu) {
            _qp_solver.get_u(k, _du[k].data());
            _scaled_du[k].noalias() = _ls_alpha * _du[k];
            _step_norm = std::max(_step_norm, _scaled_du[k].cwiseAbs().maxCoeff());
            _ocproblem.get_node(k).u_oplus(u_nom[k], _scaled_du[k], _u_candidate[k]);
        }

        // Multiplier update: λ_candidate = λ + α·(λ_QP - λ)
        _qp_solver.get_lam_lg(k, _lam_lg_qp[k].data());
        _qp_solver.get_lam_ug(k, _lam_ug_qp[k].data());
        _lam_lg_candidate[k] = _lam_lg[k] + _ls_alpha * (_lam_lg_qp[k] - _lam_lg[k]);
        _lam_ug_candidate[k] = _lam_ug[k] + _ls_alpha * (_lam_ug_qp[k] - _lam_ug[k]);
        if (k < _Nu) {
            _qp_solver.get_pi(k, _pi_qp[k].data());
            _pi_candidate[k] = _pi[k] + _ls_alpha * (_pi_qp[k] - _pi[k]);
        }
    }

    // Bind nodes to candidate trajectory and refresh _value for LS checks
    _ocproblem.bind_trajectory(_x_candidate, _u_candidate);
    _ocproblem.evaluate_all();
    _stats.update_step_norm(_step_norm);
}

void SQPSolver::accept_step() {
    auto& x_nom = _ocproblem.x_traj();
    auto& u_nom = _ocproblem.u_traj();

    // O(1) swap — no element copies
    std::swap(x_nom, _x_candidate);
    std::swap(u_nom, _u_candidate);
    std::swap(_pi, _pi_candidate);
    std::swap(_lam_lg, _lam_lg_candidate);
    std::swap(_lam_ug, _lam_ug_candidate);

    // Rebind nodes to the (now swapped) nominal trajectory
    _ocproblem.bind_trajectory(x_nom, u_nom);
}

void SQPSolver::linearize() {
    // Rebind nodes to nominal trajectory (safe reset after step/linesearch)
    _ocproblem.bind_trajectory(_ocproblem.x_traj(), _ocproblem.u_traj());

    for (int i = 0; i < _N; i++)
    {
        // Dynamics for N-1 transitions (nodes 0 to N-2)
        if (i < _Nu)
        {
            auto dyn = _ocproblem.get_node(i).get_dynamics();
            dyn->evaluate();
            dyn->jacobian();
            _b[i] = dyn->get_value();
            _A[i] = dyn->get_jac_x();
            _B[i] = dyn->get_jac_u();
        }

        // Cost linearization for all N nodes (Gauss-Newton Hessian approximation)
        // For each cost f(x, u) with residual r = f(x, u):
        //   Q += w * Jx' * Jx,  q += w * Jx' * r
        //   R += w * Ju' * Ju,  r += w * Ju' * r
        //   S += w * Ju' * Jx
        _Q[i].setZero();
        _q[i].setZero();
        if (i<_Nu){
            _R[i].setZero();
            _r[i].setZero();
            _S[i].setZero();
        }

        for (auto& cost : _ocproblem.get_node(i).get_costs()) {
            cost->evaluate();
            cost->jacobian();

            const VectorXd& cost_val = cost->get_value();
            MatrixXdConstRef Jx = cost->get_jac_x();
            MatrixXdConstRef Ju = cost->get_jac_u();
            const MatrixXd& W = cost->get_weight();

            _Q[i].noalias() += Jx.transpose() * W * Jx;
            _q[i].noalias() += Jx.transpose() * W * cost_val;

            if (i < _Nu) {
                if (Ju.cols() > 0 && i < _Nu) {
                    _R[i].noalias() += Ju.transpose() * W * Ju;
                    _r[i].noalias() += Ju.transpose() * W * cost_val;
                    _S[i].noalias() += Ju.transpose() * W * Jx;
                }
            }
        }

        //TODO - move this so we don't relinearise
        // Hessian regularization for positive definiteness
        _Q[i].diagonal().array() += _current_reg;
        if (i < _Nu)
            _R[i].diagonal().array() += _current_reg;


        // Constraint linearization: lg <= C*dx + D*du <= ug
        // Linearized around current x: lg = lb - g(x),  ug = ub - g(x)
        // C = dg/dx,  D = dg/du
        // std::cout<<"Linearizing constraints"<<std::endl;
        int row = 0;
        for (auto& con : _ocproblem.get_node(i).get_constraints()) {
            // DEBUG_PRINT(con->get_name());
            int nc = con->get_output_dim();
            con->evaluate();
            con->jacobian();

            const VectorXd& residual = con->get_value();
            _C[i].middleRows(row, nc) = con->get_jac_x();
            _lg[i].segment(row, nc) = con->get_lower_bound() - residual;
            _ug[i].segment(row, nc) = con->get_upper_bound() - residual;

            _lg[i].array() -= _opts.eps_inequality;
            _ug[i].array() += _opts.eps_inequality;

            if (i < _Nu) {
                MatrixXdConstRef Ju = con->get_jac_u();  // No copy!
                if (Ju.cols() > 0)
                    _D[i].middleRows(row, nc) = Ju;
                else
                    _D[i].middleRows(row, nc).setZero();
            }

            row += nc;
        }

    }

    _nominal_cost   = _ocproblem.cost();
    _nominal_defect = _ocproblem.dynamics_defect();
    _nominal_viol   = _ocproblem.constraint_violation();

    // hold the merit values at the current nominal trajectory, not the previous iterate.
    // Compute nominal merit values and subgradients for ls_merit() Armijo check.
    // Must run after all evaluate()/jacobian() calls above so _value and _jac are fresh.
    if (_opts.ls_type == LSType::MERIT) {
        for (int k = 0; k < _N; ++k) {
            auto& node = _ocproblem.get_node(k);
            node.calc_cost_gradient();
            node.calc_defect_gradient();
            node.calc_violation_gradient();
        }
    }

    // DEBUG_PRINT("Linearized successfully");
}

double SQPSolver::compute_kkt_residual() {
    double kkt = -1e12;

    for (int k = 0; k < _N; ++k) {
        // Stationarity w.r.t. state: ∇_{x_k} L = q_k + C_k^T(λ_ug - λ_lg) + A_{k-1}^T π_{k-1} - π_k
        // See Problem.md for derivation and HPIPM sign convention
        if (k > 0) {
            VectorXd grad_x = _q[k];
            grad_x.noalias() += _C[k].transpose() * (_lam_ug[k] - _lam_lg[k]);
            if (k < _Nu) grad_x.noalias() += _A[k].transpose() * _pi[k];
            if (k < _Nu) grad_x.noalias() -= _pi[k-1];
            kkt = std::max(kkt, grad_x.lpNorm<Eigen::Infinity>());
        }
        // Stationarity w.r.t. control: ∇_{u_k} L = r_k + D_k^T(λ_ug - λ_lg) + B_k^T π_k
        if (k < _Nu) {
            VectorXd grad_u = _r[k];
            grad_u.noalias() += _D[k].transpose() * (_lam_ug[k] - _lam_lg[k]);
            grad_u.noalias() += _B[k].transpose() * _pi[k];
            kkt = std::max(kkt, grad_u.lpNorm<Eigen::Infinity>());
        }
    }

    return kkt;
}

bool SQPSolver::break_criteria() {
    double dual_infeas = compute_kkt_residual();
    _stats.update_dual_infeasibility(dual_infeas);

    bool dual_feasible = (dual_infeas < _opts.tolerance);
    bool primal_feasible = (_candidate_defect < _opts.tolerance &&
                            _candidate_viol < _opts.tolerance);

    if (dual_feasible && primal_feasible) return true;

    // Fallback: step norm (backward-compatible safety)
    if ((_step_norm < _opts.tolerance) && primal_feasible) return true;

    return false;
}
