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

    // Detect floating base (check once and cache)
    _fb_nv = 0;
    if (_node->model().njoints > 1 &&
        _node->model().joints[1].shortname() == "JointModelFreeFlyer") {
        _fb_nv = _node->model().joints[1].nv();  // 6
    }

    _output_dim = nv - _fb_nv;
    _input_dim = _node->ndx() + _node->ndu();

    if (_v_ref.size() == 0) {
        _v_ref.resize(_output_dim);
        _v_ref.setZero();
    }

    _jacobian.setZero(_input_dim, _output_dim);
}

void VelocityCost::evaluate_impl() {
    // Exclude floating base velocities
    _value = _node->v().tail(_output_dim) - _v_ref;
}

void VelocityCost::jacobian_impl() {
    _jacobian.setZero();
    // Jacobian w.r.t. joint velocities (exclude floating base): ∂(v_joints - v_ref)/∂v_joints = I
    _jacobian.block(0, _node->nv() + _fb_nv, _output_dim, _output_dim).setIdentity();
}
