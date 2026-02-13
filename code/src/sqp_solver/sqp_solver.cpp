#include <sqp_solver/sqp_solver.h>

#include <cmath>
#include <iostream>

SQPSolver::SQPSolver(OCP& ocp): _ocproblem(ocp) { init(); }

SQPSolver::~SQPSolver() {
    // HPIPM memory automatically freed by HPIPMSolver destructor
}

void SQPSolver::solve() {
    _stats.start_timer();
    for (int i = 0; i < _opts.max_sqp_iters; i++) {
        _ls_alpha = 1.;
        linearize();
        populate_qp();
        _qp_solver.solve();

        // Backtracking line search
        step();
        // TODO write tests for the pipeline
        if (_ls_function) {
            DEBUG_PRINT("LS");
            for (int ls_iter = 1; ls_iter < _opts.max_ls_iters; ++ls_iter) {
                _stats.update_linesearch_iterations(ls_iter);
                if ((this->*_ls_function)()) break;
                _ls_alpha *= _opts.ls_scale_factor;
                step();
            }
        }
        accept_step();

        _stats.update();
        _stats.update_cost(_ocproblem.cost());
        _stats.update_constraint_violation(_ocproblem.constraint_violation());
        _stats.update_dynamics_defect(_ocproblem.dynamics_defect());
        _stats.print();
        if (break_criteria()) break;
    }
    _stats.print(1);
}

void SQPSolver::init() {
    _N = _ocproblem.num_nodes();
    _ls_alpha = 1.0;

    _prev_cost = 0.;
    _prev_defect = 0.;
    _prev_viol = 0.;

    _opts.print();

    // Set line search function pointer based on options
    switch (_opts.ls_type) {
        case LSType::MERIT:  _ls_function = &SQPSolver::ls_merit;  break;
        case LSType::FILTER: _ls_function = &SQPSolver::ls_filter; break;
        case LSType::NONE:   _ls_function = nullptr;               break;
    }
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
    // Constraints: 
    _C.resize(_Nx);   //state
    _D.resize(_Nu); //control 
    _lg.resize(_Nx); 
    _ug.resize(_Nx);
    // Step storage (reused every iteration)
    _dx.resize(_Nx);
    _du.resize(_Nu);
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
        }

        // Pre-allocate constraint matrices (ng x ndx/ndu) per stage
        _C[k].setZero(_ng, _ndx);
        if (k<_Nu) _D[k].setZero(_ng, _ndu);
        _lg[k].setZero(_ng);
        _ug[k].setZero(_ng);

        _dx[k].setZero(_ndx);
        if (k<_Nu) _du[k].setZero(_ndu);

        _x_candidate[k].setZero(_ndx);
        if (k<_Nu) _u_candidate[k].setZero(_ndu);

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


    // Set solver options for maximum performance
    _qp_solver.set_iter_max(200);                 // Reasonable default
    _qp_solver.set_tol(1e-3, 1e-3, 1e-3, 1e-3);  // Tolerances
    _qp_solver.set_warm_start(false);             // CRITICAL: Enable warm-starting (huge speedup)

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
        }

        _qp_solver.set_C(k, _C[k].data());
        if (k< _Nu) _qp_solver.set_D(k, _D[k].data());
        _qp_solver.set_lg(k, _lg[k].data());
        _qp_solver.set_ug(k, _ug[k].data());

    }

    // std::cout<<"populated_qp"<<std::endl;
}

void SQPSolver::step() {
    // Compute candidate trajectory from QP solution at current _ls_alpha.
    // After this call, nodes are bound to _x_candidate/_u_candidate.
    _step_norm = 0.0;

    auto& x_nom = _ocproblem.x_traj();
    auto& u_nom = _ocproblem.u_traj();

    for (int k = 0; k < _N; ++k) {
        _qp_solver.get_x(k, _dx[k].data());
        _step_norm = std::max(_step_norm, (_ls_alpha * _dx[k]).cwiseAbs().maxCoeff());

        VectorXd scaled_dx = _ls_alpha * _dx[k];
        _ocproblem.get_node(k).x_oplus(x_nom[k], scaled_dx, _x_candidate[k]);

        if (k < _Nu) {
            _qp_solver.get_u(k, _du[k].data());
            _step_norm = std::max(_step_norm, (_ls_alpha * _du[k]).cwiseAbs().maxCoeff());

            VectorXd scaled_du = _ls_alpha * _du[k];
            _ocproblem.get_node(k).u_oplus(u_nom[k], scaled_du, _u_candidate[k]);
        }
    }

    // Bind nodes to candidate trajectory for evaluation by ls_merit()/ls_filter()
    _ocproblem.bind_trajectory(_x_candidate, _u_candidate);
    _stats.update_step_norm(_step_norm);
}

void SQPSolver::accept_step() {
    auto& x_nom = _ocproblem.x_traj();
    auto& u_nom = _ocproblem.u_traj();

    // O(1) swap — no element copies
    std::swap(x_nom, _x_candidate);
    std::swap(u_nom, _u_candidate);

    // Rebind nodes to the (now swapped) nominal trajectory
    _ocproblem.bind_trajectory(x_nom, u_nom);
}

void SQPSolver::linearize() {
    // Rebind nodes to nominal trajectory (safe reset after step/linesearch)
    _ocproblem.bind_trajectory(_ocproblem.x_traj(), _ocproblem.u_traj());

    for (int i = 0; i < _N; i++)
    {
        // std::cout<<i<<std::endl;
        // Dynamics for N-1 transitions (nodes 0 to N-2)
        if (i < _Nu)
        {
            // std::cout<<"Linearizing dynamics"<<std::endl;
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
        }
        

        // std::cout<<"Linearizing costs"<<std::endl;
        for (auto& cost : _ocproblem.get_node(i).get_costs()) {
            std::cout<<"Linearizing :"<<cost->get_name()<< std::endl;
            cost->evaluate();
            cost->jacobian();

            const VectorXd& cost_val = cost->get_value();
            MatrixXdConstRef Jx = cost->get_jac_x();
            MatrixXd Ju = cost->get_jac_u();
            double w = cost->get_weight();

            _Q[i].noalias() += w * Jx.transpose() * Jx;
            _q[i].noalias() += w * Jx.transpose() * cost_val;

            if (Ju.cols() > 0 && i < _Nu) {
                _R[i].noalias() += w * Ju.transpose() * Ju;
                _r[i].noalias() += w * Ju.transpose() * cost_val;
            }
        }


        // Constraint linearization: lg <= C*dx + D*du <= ug
        // Linearized around current x: lg = lb - g(x),  ug = ub - g(x)
        // C = dg/dx,  D = dg/du
        // std::cout<<"Linearizing constraints"<<std::endl;
        int row = 0;
        for (auto& con : _ocproblem.get_node(i).get_constraints()) {
            int nc = con->get_output_dim();
            con->evaluate();
            con->jacobian();

            const VectorXd& residual = con->get_value();
            _C[i].middleRows(row, nc) = con->get_jac_x();
            _lg[i].segment(row, nc) = con->get_lower_bound() - residual;
            _ug[i].segment(row, nc) = con->get_upper_bound() - residual;

            MatrixXd Ju = con->get_jac_u();
            if (Ju.cols() > 0 && i < _Nu)
                _D[i].middleRows(row, nc) = Ju;
            else
                _D[i].middleRows(row, nc).setZero();

            row += nc;
        }

    }
}

bool SQPSolver::break_criteria() {
    // Check if the step norm is below tolerance (converged)
    // This is the 2-norm of the accepted step: ||alpha * [dx; du]||
    if (_step_norm < _opts.tolerance) {
        return true;  // Converged
    }

    //TODO copy converge criteria from opensot

    return false;  // Not converged yet
}
