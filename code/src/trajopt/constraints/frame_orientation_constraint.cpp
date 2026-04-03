#include <trajopt/constraints/frame_orientation_constraint.h>

FrameOrientationConstraint::FrameOrientationConstraint(
    const std::string& frame_name, const Eigen::Matrix3d& R_ref)
    : AbstractConstraint(), _fo(frame_name, R_ref) {
    _name = "frame_orientation_constraint(" + frame_name + ")";
}

void FrameOrientationConstraint::allocate_dims() {
    _fo.allocate_slices();

    _output_dim = _fo.get_output_dim();
    _input_dim = _fo.get_input_dim();

    // set_equality_to_zero();
    _lower_bound.setZero(3);
    _upper_bound.setZero(3);
}

void FrameOrientationConstraint::evaluate_impl() {
    _fo.evaluate();
    _value = _fo.get_value();
}

void FrameOrientationConstraint::jacobian_impl() {
    _fo.jacobian();
    _jacobian.leftCols(_node->ndx()) = _fo.get_jac_x();
}
