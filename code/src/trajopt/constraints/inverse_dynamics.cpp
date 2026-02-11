#include <trajopt/constraints/inverse_dynamics.h>

InvDynamics::InvDynamics() : AbstractConstraint() {
    _name = "inverse_dynamics";
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

    // Columns: [q_k(nv), v_k(nv), u_k(nv)]
    _output_dim = nv;
    _input_dim = 3 * nv;  // [x_k, u_k]
    _jacobian.resize(_output_dim, _input_dim);
    _jacobian.setZero();

    // TODO: Set bounds from effort limimits
}

void EulerIntegration::evaluate(VectorXdRef output) {
    _q       = _node->q();
    _vq      = _node->v();
    _aq      = _node->u();
    output.resize(_node->nv());

    
    pinocchio::rnea(_node->model(), _node->data(), _q, _vq, _aq, _res);

    output = _res;
}

void EulerIntegration::jacobian(MatrixXdRef jac) {
    _q = _node->q();
    _vq = _node->v();
    _aq = _node->u();

    jac.setZero();
    _dtau_dq.setZero();
    _dtau_dv.setZero();
    _dtau_da.setZero();

    pinocchio::computeRNEADerivatives(_node->model(), _node->data(), _q, _vq, _aq, _dtau_dq, _dtau_dv, _dtau_da);

    jac.block(0, 0, nv, nv) = _dtau_dq;
    jac.block(0, nv, nv, nv) = _dtau_dv;
    jac.block(0, 2*nv, nv, nv) = _dtau_da;
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
    return _jacobian.rightCols(_node->nv());
}
