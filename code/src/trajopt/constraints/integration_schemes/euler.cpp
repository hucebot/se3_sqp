#include <trajopt/constraints/integration_schemes/euler.h>
#include <trajopt/node.h>

EulerIntegration::EulerIntegration(double dt) : AbstractConstraint(), _dt(dt) {
    _name = "euler_integration";
    // Note: Allocation deferred to allocate_slices() since _node is not set yet
}

void EulerIntegration::allocate_slices() {
    // Now _node is set, we can allocate based on dimensions
    int nq = _node->nq();
    int nv = _node->nv();

    _q_integrated.resize(nq);
    _res.resize(2*nv);
    _J_q.resize(nv, nv);
    _J_v.resize(nv, nv);
    _J_diff_qnext.resize(nv, nv);

    // Pre-allocate Jacobian: (2*nv × 5*nv)
    // Columns: [q_k(nv), v_k(nv), u_k(nv), q_{k+1}(nv), v_{k+1}(nv)]
    _output_dim = 2 * nv;
    _input_dim = 3 * nv;  // [x_k, u_k, x_{k+1}] = [2*nv, nv, 2*nv]
    _jacobian.resize(_output_dim, _input_dim);
    _jacobian.setZero();

    // Set bounds to zero (equality constraint)
    set_equality_to_zero();
}

void EulerIntegration::evaluate(VectorXdRef output) {
    _q       = _node->q();
    _q_next  = _node->next_node->q();
    _vq      = _node->v();
    _vq_next = _node->next_node->v();
    _aq      = _node->u();

    output.resize(2 * _node->nv());

    pinocchio::integrate(_node->model(), _q, _vq * _dt, _q_integrated);

    // Position residual: q_next ⊖ q_integrated
    pinocchio::difference(_node->model(), _q_next, _q_integrated,
                          _res.head(_node->nv()));

    // Velocity defect: f_v(x_k, u_k) - v_{k+1} = (v + dt*a) - v_next
    _res.tail(_node->nv()) = (_vq + _dt * _aq) - _vq_next;

    output = _res;

}

void EulerIntegration::jacobian(MatrixXdRef jac) {
    _q = _node->q();
    _q_next = _node->next_node->q();
    _vq = _node->v();
    _vq_next = _node->next_node->v();
    _aq = _node->u();

    auto nv = _node->nv();

    jac.setZero();
    _J_q.setZero();
    _J_v.setZero();
    _J_diff_qnext.setZero();

    // Jacobians of integrate(q, dt*v) w.r.t. q and dt*v
    pinocchio::dIntegrate(_node->model(), _q, _vq * _dt, _J_q,
                          pinocchio::ArgumentPosition::ARG0);
    pinocchio::dIntegrate(_node->model(), _q, _vq * _dt, _J_v,
                          pinocchio::ArgumentPosition::ARG1);

    // Jacobian of difference w.r.t. q_next (first argument)
    pinocchio::dDifference(_node->model(), _q_next, _q_integrated, _J_diff_qnext,
                           pinocchio::ArgumentPosition::ARG1);

    // Position constraint Jacobian: ∂(q_next ⊖ q_integrated)/∂[q_k, v_k, u_k]
    // ∂/∂q_k = -dDiff/dq_int * dInt/dq
    jac.block(0, 0, nv, nv) = _J_diff_qnext * _J_q;
    // ∂/∂v_k = -dDiff/dq_int * dInt/d(dt*v) * dt
    jac.block(0, nv, nv, nv) = _J_diff_qnext * _J_v * _dt;

    // Velocity dynamics Jacobian: ∂f_v/∂[...] where f_v = v_k + dt*u_k
    // ∂f_v/∂v_k = I
    jac.block(nv, nv, nv, nv) = MatrixXd::Identity(nv, nv);
    // ∂f_v/∂u_k = dt*I
    jac.block(nv, 2*nv, nv, nv) = _dt * MatrixXd::Identity(nv, nv);
}

void EulerIntegration::jacobian() {
    // Compute into internal storage
    jacobian(_jacobian);
}

MatrixXdConstRef EulerIntegration::get_jac_x() const {
    // Return ∂g/∂x_k = [∂g/∂q_k | ∂g/∂v_k]
    return _jacobian.leftCols(2 * _node->nv());
}

MatrixXd EulerIntegration::get_jac_u() const {
    // Return ∂g/∂u_k (columns 2*nv to 3*nv-1)
    return _jacobian.rightCols(_node->nv(););
}
