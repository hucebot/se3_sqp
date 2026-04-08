#include <trajopt/functions/frame_velocity.h>
#include <pinocchio/algorithm/frames-derivatives.hpp>

FrameVelocity::FrameVelocity(const std::string& frame_name, const Vector6d& v_ref)
    : AbstractFunction(),     _v_ref(v_ref),
    _frame_name(frame_name),
    _re_ref_frame(pinocchio::LOCAL),
    _base_frame_name("base_link")
{
    _name = "frame_velocity(" + frame_name + ")";
    _frame_id = -1;
}


void FrameVelocity::allocate_dims() {
    if (!_node->model().existFrame(_frame_name)) {
        throw std::invalid_argument(
            _name + ": frame '" + _frame_name + "' not found in model");
    }

    _frame_id = _node->model().getFrameId(_frame_name);

    _output_dim = 6;
    _input_dim = _node->ndx() + _node->ndu();

    _dv_dq.setZero(_output_dim, _node->nv());
    _dv_dqdot.setZero(_output_dim, _node->nv());
}

void FrameVelocity::evaluate_impl() {
    // Ensure FK and frame placements are computed
    _node->require_frame_placements();

    pinocchio::Motion vel;
    if(_frame_name == _base_frame_name)
        vel = _node->data().v[1];
    else
        vel = pinocchio::getFrameVelocity(_node->model(), _node->data(), _frame_id, _re_ref_frame);

    _value.setZero(6);
    _value.segment(0, 3) = vel.linear() - _v_ref.segment(0,3);
    _value.segment(3, 3) = vel.angular() - _v_ref.segment(3,3);
}

void FrameVelocity::jacobian_impl() {
    _node->require_fk_derivatives();
    if(_frame_name == _base_frame_name)
        pinocchio::getFrameVelocityDerivatives(_node->model(), _node->data(), _frame_id, pinocchio::LOCAL, _dv_dq, _dv_dqdot);
    else
        pinocchio::getFrameVelocityDerivatives(_node->model(), _node->data(), _frame_id, _re_ref_frame, _dv_dq, _dv_dqdot);

    _jacobian.setZero();
    _jacobian.block(0, 0,             6, _node->nv()) = _dv_dq;
    _jacobian.block(0, _node->nv(),   6, _node->nv()) = _dv_dqdot;
}