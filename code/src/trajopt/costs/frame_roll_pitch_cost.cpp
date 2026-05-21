#include <trajopt/costs/frame_roll_pitch_cost.h>

FrameRollPitchCost::FrameRollPitchCost(
    const std::string& frame_name, const Eigen::Matrix3d& R_ref, double weight)
    : AbstractCost(weight), _foc(frame_name, R_ref) {
    _name = "frame_roll_pitch_cost(" + frame_name + ")";
}

FrameRollPitchCost::FrameRollPitchCost(
    const std::string& frame_name, const Eigen::Matrix3d& R_ref, const Matrix3d& weight)
    : AbstractCost(weight), _foc(frame_name, R_ref) {
    _name = "frame_roll_pitch_cost(" + frame_name + ")";
}

void FrameRollPitchCost::allocate_dims() {
    _foc.allocate_slices();

    _output_dim = 2;
    _input_dim = _foc.get_input_dim();
}

void FrameRollPitchCost::evaluate_impl() {
    _foc.evaluate();
    _value = _foc.get_value().segment(0, 2);
}

void FrameRollPitchCost::jacobian_impl() {
    _foc.jacobian();
    // Copy x-block into our jacobian (u-block stays zero from allocate_slices)
    _jacobian.leftCols(_node->ndx()) = _foc.get_jac_x().topRows(2);
}
