#include <trajopt/costs/acceleration_cost.h>

AccelerationCost::AccelerationCost(const VectorXd& a_ref)
    : AbstractCost(), _a_ref(a_ref) {
    _name = "acceleration_cost";
}
AccelerationCost::AccelerationCost()
    : AbstractCost() {
    _name = "acceleration_cost";
}

void AccelerationCost::allocate_slices() {
    int nv = _node->nv();

    _output_dim = nv;
    _input_dim = _node->ndx() + _node->ndu();

    if (_a_ref.size()==0){
        _a_ref.resize(_output_dim);
        _a_ref.setZero();
    }

    _weight.resize(_output_dim, _output_dim);
    _weight.setIdentity();

    // Jacobian: (nv × 2*nv), right half (∂r/∂v) stays zero
    _value.resize(_output_dim);
    _jacobian.resize(_output_dim, _input_dim);
    _jacobian.setZero();
}

void AccelerationCost::evaluate_impl() {
    _value = _node->u() - _a_ref; //TODO: change with forces
}

void AccelerationCost::jacobian_impl() {
    _jacobian.middleCols(2*_node->nv(), _node->nv()).setIdentity();
}

MatrixXdConstRef AccelerationCost::get_jac_x() const {
    return _jacobian.leftCols(_node->ndx());
}

MatrixXd AccelerationCost::get_jac_u() const {
    return _jacobian.middleCols(2*_node->nv(), _node->nv());
}
