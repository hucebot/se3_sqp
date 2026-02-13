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
    _J_q.resize(nv, nv);
    _J_v.resize(nv, nv);
    _J_diff_qnext.resize(nv, nv);

    // Pre-allocate Jacobian: (2*nv × 5*nv)
    // Columns: [q_k(nv), v_k(nv), u_k(nv), q_{k+1}(nv), v_{k+1}(nv)]
    _output_dim = 2 * nv;
    _input_dim = 3 * nv;  // [x_k, u_k, x_{k+1}] = [2*nv, nv, 2*nv]
    _value.resize(_output_dim);
    _jacobian.resize(_output_dim, _input_dim);
    _jacobian.setZero();

    // Set bounds to zero (equality constraint)
    set_equality_to_zero();
}


void EulerIntegration::evaluate_impl() {
    _q       = _node->q();
    _q_next  = _node->next_node->q();
    _vq      = _node->v();
    _vq_next = _node->next_node->v();
    _aq      = _node->u();

    pinocchio::integrate(_node->model(), _q, _vq * _dt, _q_integrated);
    pinocchio::difference(_node->model(), _q_next, _q_integrated, _value.head(_node->nv()));
    _value.tail(_node->nv()) = (_vq + _dt * _aq) - _vq_next;
}

void EulerIntegration::jacobian_impl() {
    _q       = _node->q();
    _q_next  = _node->next_node->q();
    _vq      = _node->v();
    _vq_next = _node->next_node->v();
    _aq      = _node->u();

    auto nv = _node->nv();

    _jacobian.setZero();
    _J_q.setZero();
    _J_v.setZero();
    _J_diff_qnext.setZero();

    pinocchio::dIntegrate(_node->model(), _q, _vq * _dt, _J_q,
                          pinocchio::ArgumentPosition::ARG0);
    pinocchio::dIntegrate(_node->model(), _q, _vq * _dt, _J_v,
                          pinocchio::ArgumentPosition::ARG1);
    pinocchio::dDifference(_node->model(), _q_next, _q_integrated, _J_diff_qnext,
                           pinocchio::ArgumentPosition::ARG1);

    _jacobian.block(0, 0,    nv, nv) = _J_diff_qnext * _J_q;
    _jacobian.block(0, nv,   nv, nv) = _J_diff_qnext * _J_v * _dt;
    _jacobian.block(nv, nv,  nv, nv) = MatrixXd::Identity(nv, nv);
    _jacobian.block(nv, 2*nv, nv, nv) = _dt * MatrixXd::Identity(nv, nv);
}



MatrixXdConstRef EulerIntegration::get_jac_x() const {
    // Return ∂g/∂x_k = [∂g/∂q_k | ∂g/∂v_k]
    return _jacobian.leftCols(2 * _node->nv());
}

MatrixXd EulerIntegration::get_jac_u() const {
    // Return ∂g/∂u_k (columns 2*nv to 3*nv-1)
    return _jacobian.rightCols(_node->nv());
}
