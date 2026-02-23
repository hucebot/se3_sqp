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

void AccelerationCost::allocate_dims() {
    int nv = _node->nv();

    // Detect floating base (check once and cache)
    _fb_nv = 0;
    if (_node->model().njoints > 1 &&
        _node->model().joints[1].shortname() == "JointModelFreeFlyer") {
        _fb_nv = _node->model().joints[1].nv();  // 6
    }

    _output_dim = nv - _fb_nv;
    _input_dim = _node->ndx() + _node->ndu();

    if (_a_ref.size() == 0) {
        _a_ref.resize(_output_dim);
        _a_ref.setZero();
    }
}

void AccelerationCost::evaluate_impl() {
    // Exclude floating base accelerations
    _value = _node->a().tail(_output_dim) - _a_ref;
}

void AccelerationCost::jacobian_impl() {
    _jacobian.setZero();
    // Jacobian w.r.t. joint accelerations (exclude floating base): ∂(a_joints - a_ref)/∂a_joints = I
    // Acceleration starts at column offset 2*nv in the input [q, v, a, f]
    _jacobian.block(0, 2 * _node->nv() + _fb_nv, _output_dim, _output_dim).setIdentity();
}
