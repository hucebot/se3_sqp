#include <trajopt/functions/frame_orientation.h>

FrameOrientation::FrameOrientation(const std::string& frame_name, const Eigen::Matrix3d& R_ref)
    : AbstractFunction(), _frame_name(frame_name), _R_ref(R_ref) {
    _name = "frame_orientation(" + frame_name + ")";
    _frame_id = -1;
}

void FrameOrientation::allocate_dims() {
    if (!_node->model().existFrame(_frame_name)) {
        throw std::invalid_argument(
            _name + ": frame '" + _frame_name + "' not found in model");
    }
    _frame_id = _node->model().getFrameId(_frame_name);

    _output_dim = 3;
    _input_dim = _node->ndx() + _node->ndu();

    _Jframe.resize(6, _node->nv());
    _Jlog.resize(3, 3);
}

void FrameOrientation::evaluate_impl() {
    _node->require_frame_placements();

    const Eigen::Matrix3d& R_frame = _node->data().oMf[_frame_id].rotation();
    Eigen::Matrix3d R_err = R_frame.transpose() * _R_ref;

    _value = pinocchio::log3(R_err);
}

void FrameOrientation::jacobian_impl() {
    _node->require_fk_derivatives();

    const Eigen::Matrix3d& R_frame = _node->data().oMf[_frame_id].rotation();
    _R_err = R_frame.transpose() * _R_ref;

    pinocchio::Jlog3(_R_err, _Jlog);

    _Jframe.setZero();
    pinocchio::getFrameJacobian(
        _node->model(), _node->data(),
        _frame_id, pinocchio::LOCAL, _Jframe);

    _jacobian.setZero();
    // LOCAL Jacobian angular part already expresses angular velocity in frame's body frame
    _jacobian.leftCols(_node->nv()).noalias() = -_Jlog.transpose() * _Jframe.bottomRows(3);
}
