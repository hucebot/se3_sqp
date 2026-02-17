#include <trajopt/costs/configuration_cost.h>
#include <trajopt/node.h>

ConfigurationCost::ConfigurationCost(const VectorXd& q_ref)
    : AbstractCost(), _q_ref(q_ref) {
    _name = "configuration_cost";
}

void ConfigurationCost::allocate_slices() {
    int nv = _node->nv();

    _output_dim = nv;
    _input_dim =_node->ndx() + _node->ndu();

    _J_dq.resize(nv, nv);

    _weight.resize(_output_dim, _output_dim);
    _weight.setIdentity();

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
    return _jacobian.leftCols(_node->ndx());
}

MatrixXd ConfigurationCost::get_jac_u() const {
    return MatrixXd::Zero(_output_dim, 0);
}
