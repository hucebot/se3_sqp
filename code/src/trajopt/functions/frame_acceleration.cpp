#include <trajopt/functions/frame_acceleration.h>
#include <pinocchio/algorithm/frames-derivatives.hpp>

FrameAcceleration::FrameAcceleration(const std::string& frame_name, const Vector6d& a_ref)
    : AbstractFunction(),     _a_ref(a_ref),
    _frame_name(frame_name),
    _re_ref_frame(pinocchio::LOCAL),
    _base_frame_name("base_link")
{
    _name = "frame_acceleration(" + frame_name + ")";
    _frame_id = -1;
}


void FrameAcceleration::allocate_dims() {
    if (!_node->model().existFrame(_frame_name)) {
        throw std::invalid_argument(
            _name + ": frame '" + _frame_name + "' not found in model");
    }

    _frame_id = _node->model().getFrameId(_frame_name);

    _output_dim = 6;
    _input_dim = _node->ndx() + _node->ndu();

    _dv_dq.setZero(_output_dim, _node->nv());
    _da_dq.setZero(_output_dim, _node->nv());
    _da_dv.setZero(_output_dim, _node->nv());
    _da_da.setZero(_output_dim, _node->nv());
}

void FrameAcceleration::evaluate_impl() {
    // Ensure FK and frame placements are computed
    _node->require_frame_placements();

    pinocchio::Motion acc;
    if(_frame_name == _base_frame_name)
        acc = _node->data().a[1];
    else
        acc = pinocchio::getFrameAcceleration(_node->model(), _node->data(), _frame_id, _re_ref_frame);

    _value.setZero(6);
    _value.segment(0, 3) = acc.linear() - _a_ref.segment(0,3);
    _value.segment(3, 3) = acc.angular() - _a_ref.segment(3,3);
}

void FrameAcceleration::jacobian_impl() {
    _node->require_fk_derivatives();
    if(_frame_name == _base_frame_name)
        pinocchio::getFrameAccelerationDerivatives(_node->model(), _node->data(), _frame_id, pinocchio::LOCAL, _dv_dq, _da_dq, _da_dv, _da_da);
    else
        pinocchio::getFrameAccelerationDerivatives(_node->model(), _node->data(), _frame_id, _re_ref_frame, _dv_dq, _da_dq, _da_dv, _da_da);

    _jacobian.setZero();
    _jacobian.block(0, 0,             6, _node->nv()) = _da_dq;
    _jacobian.block(0, _node->nv(),   6, _node->nv()) = _da_dv;
    _jacobian.block(0, 2 * _node->nv(),   6, _node->nv()) = _da_da;
}