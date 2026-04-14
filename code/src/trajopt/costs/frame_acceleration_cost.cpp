#include <trajopt/costs/frame_acceleration_cost.h>
#include <trajopt/node.h>


FrameAccelerationCost::FrameAccelerationCost(const std::string& frame_name, const Vector6d& a_ref, double weight):
    AbstractCost(weight),
    _fa(frame_name, a_ref)
{
    _name = "frame_acceleration_cost(" + frame_name + ")";
}

FrameAccelerationCost::FrameAccelerationCost(const std::string& frame_name, const Vector6d& a_ref, const Matrix6d& weight):
    AbstractCost(weight),
    _fa(frame_name, a_ref)
{
    _name = "frame_acceleration_cost(" + frame_name + ")";
}

void FrameAccelerationCost::allocate_dims() {

    _fa.allocate_slices();

    _output_dim = _fa.get_output_dim();
    _input_dim = _fa.get_input_dim();
}

void FrameAccelerationCost::evaluate_impl() {
    _fa.evaluate();
    _value = _fa.get_value();
}

void FrameAccelerationCost::jacobian_impl() {
    _fa.jacobian();
    _jacobian.leftCols(_node->ndx()) = _fa.get_jac_x();
    _jacobian.rightCols(_node->ndu()) = _fa.get_jac_u();
}

