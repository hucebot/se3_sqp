#include <trajopt/constraints/inverse_dynamics.h>

InvDynamics::InvDynamics() : AbstractConstraint() {
    _name = "inverse_dynamics";
    // Note: Allocation deferred to allocate_slices() since _node is not set yet
}

void InvDynamics::allocate_dims() {
    _id.allocate_slices();

    _output_dim = _id.get_output_dim();
    _input_dim = _id.get_input_dim();  // [q_k, v_k, a_k, f_1, ..., f_nc]

    // Set bounds from model effort limits
    const VectorXd& effort = _node->model().effortLimit;
    _lower_bound = -effort;
    _upper_bound =  effort;

    // Floating base: unactuated DOFs → tau must be zero
    if (_node->model().njoints > 1 &&
        _node->model().joints[1].shortname() == "JointModelFreeFlyer") {
        int fb_nv = _node->model().joints[1].nv();  // 6 for free-flyer
        _lower_bound.head(fb_nv).setZero();
        _upper_bound.head(fb_nv).setZero();
    }
}

void InvDynamics::evaluate_impl() {
    _id.evaluate_impl();
    _value = _id.get_value();
}

void InvDynamics::jacobian_impl() {
    _id.jacobian();

    _jacobian.setZero();
    _jacobian.leftCols(_node->ndx()) = _id.get_jac_x();
    _jacobian.rightCols(_node->ndu()) = _id.get_jac_u();
}
