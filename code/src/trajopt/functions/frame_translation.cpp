#include <trajopt/functions/frame_translation.h>

FrameTranslation::FrameTranslation(const std::string& frame_name, const Eigen::Vector3d& p_ref)
    : AbstractFunction(), _frame_name(frame_name), _p_ref(p_ref) {
    _name = "frame_translation(" + frame_name + ")";
    _frame_id = -1;
}

void FrameTranslation::allocate_dims() {
    if (!_node->model().existFrame(_frame_name)) {
        throw std::invalid_argument(
            _name + ": frame '" + _frame_name + "' not found in model");
    }
    _frame_id = _node->model().getFrameId(_frame_name);

    _output_dim = 3;
    _input_dim = _node->ndx() + _node->ndu();

    _Jframe.resize(6, _node->nv());
}

void FrameTranslation::evaluate_impl() {
    // Ensure FK and frame placements are computed (cached on node)
    _node->require_frame_placements();

    _value = _node->data().oMf[_frame_id].translation() - _p_ref;
}

void FrameTranslation::jacobian_impl() {
    _Jframe.setZero();
    //TODO - getFrameJacobian for faster
    pinocchio::computeFrameJacobian(
        _node->model(), _node->data(), _node->q(),
        _frame_id, pinocchio::LOCAL_WORLD_ALIGNED, _Jframe);

    _jacobian.setZero();
    // Linear part of the frame Jacobian (top 3 rows) goes into the q-columns
    _jacobian.leftCols(_node->nv()) = _Jframe.topRows(3);
}

MatrixXdConstRef FrameTranslation::get_jac_x() const {
    return _jacobian.leftCols(_node->ndx());
}

MatrixXd FrameTranslation::get_jac_u() const {
    return MatrixXd::Zero(_output_dim, _node->ndu());
}
