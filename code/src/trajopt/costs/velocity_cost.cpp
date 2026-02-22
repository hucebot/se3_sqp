#include <trajopt/costs/velocity_cost.h>

VelocityCost::VelocityCost(const VectorXd& v_ref, double weight)
    : AbstractCost(weight), _v_ref(v_ref) {
    _name = "velocity_cost";
}
VelocityCost::VelocityCost(const VectorXd& v_ref, const MatrixXd& weight)
    : AbstractCost(weight), _v_ref(v_ref) {
    _name = "velocity_cost";
}
VelocityCost::VelocityCost(double weight)
    : AbstractCost(weight) {
    _name = "velocity_cost";
}

void VelocityCost::allocate_dims() {
    int nv = _node->nv();
    _output_dim = nv;
    _input_dim = _node->ndx() + _node->ndu();

    if (_v_ref.size() == 0) {
        _v_ref.resize(_output_dim);
        _v_ref.setZero();
    }
}

void VelocityCost::evaluate_impl() {
    _value = _node->v() - _v_ref;
}

void VelocityCost::jacobian_impl() {
    _jacobian.middleCols(_node->nv(), _node->nv()).setIdentity();
}

MatrixXdConstRef VelocityCost::get_jac_x() const {
    return _jacobian.leftCols(_node->ndx());
}

MatrixXdConstRef VelocityCost::get_jac_u() const {
    return _jacobian.rightCols(_node->ndu());
}
