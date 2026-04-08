#include <trajopt/constraints/frame_velocity_constraint.h>
#include <trajopt/node.h>


FrameVelocityConstraint::FrameVelocityConstraint(const std::string& frame_name, const Vector6d& v_ref):
    AbstractConstraint(),
    _fv(frame_name, Vector6d::Zero()),
    _v_ref(v_ref)
{
    _name = "frame_velocity_constraint(" + frame_name + ")";
}

void FrameVelocityConstraint::allocate_dims() {

    _fv.allocate_slices();

    _output_dim = _fv.get_output_dim();
    _input_dim = _fv.get_input_dim();

    // Initialize bounds (will be updated in evaluate_impl based on contact state)
    _lower_bound.setZero(_output_dim);
    _upper_bound.setZero(_output_dim);
}

void FrameVelocityConstraint::evaluate_impl() {
    _fv.evaluate();
    _value = _fv.get_value();
    _lower_bound = -_v_ref;
    _upper_bound = _v_ref;
}

void FrameVelocityConstraint::jacobian_impl() {
    _fv.jacobian();
    _jacobian.leftCols(_node->ndx()) = _fv.get_jac_x();
    _jacobian.rightCols(_node->ndu()) = _fv.get_jac_u();
}

