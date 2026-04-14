#include <trajopt/costs/frame_velocity_cost.h>
#include <trajopt/node.h>


FrameVelocityCost::FrameVelocityCost(const std::string& frame_name, const Vector6d& v_ref, double weight):
    AbstractCost(weight),
    _fv(frame_name, v_ref)
{
    _name = "frame_velocity_cost(" + frame_name + ")";
}

FrameVelocityCost::FrameVelocityCost(const std::string& frame_name, const Vector6d& v_ref, const Matrix6d& weight):
    AbstractCost(weight),
    _fv(frame_name, v_ref)
{
    _name = "frame_velocity_cost(" + frame_name + ")";
}

void FrameVelocityCost::allocate_dims() {

    _fv.allocate_slices();

    _output_dim = _fv.get_output_dim();
    _input_dim = _fv.get_input_dim();
}

void FrameVelocityCost::evaluate_impl() {
    _fv.evaluate();
    _value = _fv.get_value();
}

void FrameVelocityCost::jacobian_impl() {
    _fv.jacobian();
    _jacobian.leftCols(_node->ndx()) = _fv.get_jac_x();
    _jacobian.rightCols(_node->ndu()) = _fv.get_jac_u();
}

