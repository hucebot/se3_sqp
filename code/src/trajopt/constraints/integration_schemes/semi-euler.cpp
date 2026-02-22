#include <trajopt/constraints/integration_schemes/semi-euler.h>
#include <trajopt/node.h>

SemiEulerIntegration::SemiEulerIntegration(double dt) : AbstractConstraint(), _dt(dt) {
    _name = "semi_euler_integration";
    // Note: Allocation deferred to allocate_slices() since _node is not set yet
}

void SemiEulerIntegration::allocate_dims() {
    int nq = _node->nq();
    int nv = _node->nv();

    _q_integrated.resize(nq);
    _J_q.resize(nv, nv);
    _J_v.resize(nv, nv);
    _J_diff_qnext.resize(nv, nv);

    _output_dim = 2 * nv;
    _input_dim = _node->ndx() + _node->ndu();  // [x_k, u_k] where u_k = [a; forces]

    // Set bounds to zero (equality constraint)
    set_equality_to_zero();
}


void SemiEulerIntegration::evaluate_impl() {
    _q       = _node->q();
    _q_next  = _node->next_node->q();
    _vq      = _node->v();
    _vq_next = _node->next_node->v();
    _aq      = _node->a();

    _value.tail(_node->nv()) = (_vq + _dt * _aq) - _vq_next;

    pinocchio::integrate(_node->model(), _q, (_vq + _dt * _aq) * _dt, _q_integrated);
    pinocchio::difference(_node->model(), _q_next, _q_integrated, _value.head(_node->nv()));
    
}

void SemiEulerIntegration::jacobian_impl() {
    _q       = _node->q();
    _q_next  = _node->next_node->q();
    _vq      = _node->v();
    _vq_next = _node->next_node->v();
    _aq      = _node->a();

    auto nv = _node->nv();

    _jacobian.setZero();
    _J_q.setZero();
    _J_v.setZero();
    _J_diff_qnext.setZero();

    pinocchio::dIntegrate(_node->model(), _q, (_vq + _dt * _aq) * _dt, _J_q,
                          pinocchio::ArgumentPosition::ARG0);
    pinocchio::dIntegrate(_node->model(), _q, (_vq + _dt * _aq) * _dt, _J_v,
                          pinocchio::ArgumentPosition::ARG1);
    pinocchio::dDifference(_node->model(), _q_next, _q_integrated, _J_diff_qnext,
                           pinocchio::ArgumentPosition::ARG1);

    _jacobian.block(0, 0,    nv, nv) = _J_diff_qnext * _J_q;
    _jacobian.block(0, nv,   nv, nv) = _J_diff_qnext * _J_v * _dt;
    _jacobian.block(0, 2*nv,   nv, nv) = _J_diff_qnext * _J_v * _dt*_dt;
    _jacobian.block(nv, nv,  nv, nv) = MatrixXd::Identity(nv, nv);
    _jacobian.block(nv, 2*nv, nv, nv) = _dt * MatrixXd::Identity(nv, nv);
}



MatrixXdConstRef SemiEulerIntegration::get_jac_x() const {
    // Return ∂g/∂x_k = [∂g/∂q_k | ∂g/∂v_k]
    return _jacobian.leftCols(_node->ndx());
}

MatrixXdConstRef SemiEulerIntegration::get_jac_u() const {
    // Return ∂g/∂u_k — acceleration + force columns (forces are zero)
    return _jacobian.rightCols(_node->ndu());
}
