#include <trajopt/costs/frame_orientation_cost.h>

FrameOrientationCost::FrameOrientationCost(
    const std::string& frame_name, const Eigen::Matrix3d& R_ref, double weight)
    : AbstractCost(weight), _fo(frame_name, R_ref) {
    _name = "frame_orientation_cost(" + frame_name + ")";
}

FrameOrientationCost::FrameOrientationCost(
    const std::string& frame_name, const Eigen::Matrix3d& R_ref, const Matrix3d& weight)
    : AbstractCost(weight), _fo(frame_name, R_ref) {
    _name = "frame_orientation_cost(" + frame_name + ")";
}

void FrameOrientationCost::allocate_dims() {
    _fo.allocate_slices();

    _output_dim = _fo.get_output_dim();
    _input_dim = _fo.get_input_dim();
}

void FrameOrientationCost::evaluate_impl() {
    _fo.evaluate();
    _value = _fo.get_value();
}

void FrameOrientationCost::jacobian_impl() {
    _fo.jacobian();
    // Copy x-block into our jacobian (u-block stays zero from allocate_slices)
    _jacobian.leftCols(_node->ndx()) = _fo.get_jac_x();
}
