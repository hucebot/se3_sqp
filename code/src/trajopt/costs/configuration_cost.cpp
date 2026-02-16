#include <trajopt/costs/configuration_cost.h>
#include <trajopt/node.h>

ConfigurationCost::ConfigurationCost(const VectorXd& q_ref)
    : AbstractCost(), _q_ref(q_ref) {
    _name = "configuration_cost";
}

void ConfigurationCost::allocate_slices() {
    int nv = _node->nv();

    // Residual lives in tangent space: q ⊖ q_ref (dimension nv)
    _output_dim = nv;
    // Input is full state [q; v], dimension 2*nv (no control dependency)
    _input_dim = 2 * nv;

    _J_dq.resize(nv, nv);

    // Jacobian: (nv × 2*nv), right half (∂r/∂v) stays zero
    _value.resize(_output_dim);
    _jacobian.resize(_output_dim, _input_dim);
    _jacobian.setZero();
}

void ConfigurationCost::evaluate_impl() {
    pinocchio::difference(_node->model(), _q_ref, _node->q(), _value);
}

void ConfigurationCost::jacobian_impl() {
    _jacobian.setZero();
    _J_dq.setZero();
    pinocchio::dDifference(_node->model(), _q_ref, _node->q(), _J_dq,
                           pinocchio::ArgumentPosition::ARG1);
    _jacobian.leftCols(_node->nv()) = _J_dq;
}

MatrixXdConstRef ConfigurationCost::get_jac_x() const {
    return _jacobian;
}

MatrixXd ConfigurationCost::get_jac_u() const {
    return MatrixXd::Zero(_output_dim, 0);
}
