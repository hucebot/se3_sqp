#include <trajopt/costs/acceleration_cost.h>

AccelerationCost::AccelerationCost(const VectorXd& a_ref, double weight)
    : AbstractCost(weight), _a_ref(a_ref) {
    _name = "acceleration_cost";
}
AccelerationCost::AccelerationCost(const VectorXd& a_ref, const MatrixXd& weight)
    : AbstractCost(weight), _a_ref(a_ref) {
    _name = "acceleration_cost";
}
AccelerationCost::AccelerationCost(double weight)
    : AbstractCost(weight) {
    _name = "acceleration_cost";
}

void AccelerationCost::allocate_slices_impl() {
    int nv = _node->nv();
    _output_dim = nv;
    _input_dim = _node->ndx() + _node->ndu();

    if (_a_ref.size() == 0) {
        _a_ref.resize(_output_dim);
        _a_ref.setZero();
    }
}

void AccelerationCost::evaluate_impl() {
    _value = _node->a() - _a_ref;
}

void AccelerationCost::jacobian_impl() {
    // Jacobian w.r.t acceleration is identity, w.r.t forces is zero
    _jacobian.middleCols(2*_node->nv(), _node->nv()).setIdentity();
    // Force columns are already zero from _jacobian.setZero() in base class
}

MatrixXdConstRef AccelerationCost::get_jac_x() const {
    return _jacobian.leftCols(_node->ndx());
}

MatrixXd AccelerationCost::get_jac_u() const {
    // Return all control columns (acceleration + forces)
    return _jacobian.rightCols(_node->ndu());
}
