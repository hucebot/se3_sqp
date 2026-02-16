#include <trajopt/constraints/inverse_dynamics.h>

InvDynamics::InvDynamics() : AbstractConstraint() {
    _name = "inverse_dynamics";
    // Note: Allocation deferred to allocate_slices() since _node is not set yet
}

void InvDynamics::allocate_slices() {
    // Now _node is set, we can allocate based on dimensions
    int nq = _node->nq();
    int nv = _node->nv();

    _output_dim = nv;
    _input_dim = 3 * nv;  // [x_k, u_k]

    _dtau_dq.resize(nv, nv);
    _dtau_dv.resize(nv, nv);
    _dtau_da.resize(nv, nv);

    // Columns: [q_k(nv), v_k(nv), u_k(nv)]

    _value.resize(_output_dim);
    _jacobian.resize(_output_dim, _input_dim);
    _jacobian.setZero();

    // Set bounds from model effort limits
    // effortLimit is nv-dimensional; joints with limit 0 are unactuated (tau = 0)
    const VectorXd& effort = _node->model().effortLimit;
    _lower_bound = -effort;
    _upper_bound =  effort;
}

void InvDynamics::evaluate_impl() {
    _q  = _node->q();
    _vq = _node->v();
    _aq = _node->u();
    pinocchio::rnea(_node->model(), _node->data(), _q, _vq, _aq);
    _value = _node->data().tau;
}

void InvDynamics::jacobian_impl() {
    _q  = _node->q();
    _vq = _node->v();
    _aq = _node->u();
    _jacobian.setZero();
    _dtau_dq.setZero();
    _dtau_dv.setZero();
    _dtau_da.setZero();
    pinocchio::computeRNEADerivatives(_node->model(), _node->data(), _q, _vq, _aq, _dtau_dq, _dtau_dv, _dtau_da);
    _dtau_da.template triangularView<Eigen::StrictlyLower>() = _dtau_da.transpose().template triangularView<Eigen::StrictlyLower>();
    _jacobian.block(0, 0,           _node->nv(), _node->nv()) = _dtau_dq;
    _jacobian.block(0, _node->nv(), _node->nv(), _node->nv()) = _dtau_dv;
    _jacobian.block(0, 2*_node->nv(), _node->nv(), _node->nv()) = _dtau_da;
}


MatrixXdConstRef InvDynamics::get_jac_x() const {
    // Return ∂g/∂x_k = [∂g/∂q_k | ∂g/∂v_k]
    return _jacobian.leftCols(2 * _node->nv());
}

MatrixXd InvDynamics::get_jac_u() const {
    // Return ∂g/∂u_k (columns 2*nv to 3*nv-1)
    return _jacobian.rightCols(_node->nv());
}
