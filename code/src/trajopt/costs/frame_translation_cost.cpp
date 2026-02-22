#include <trajopt/costs/frame_translation_cost.h>

FrameTranslationCost::FrameTranslationCost(
    const std::string& frame_name, const Eigen::Vector3d& p_ref, double weight)
    : AbstractCost(weight), _ft(frame_name, p_ref) {
    _name = "frame_translation_cost(" + frame_name + ")";
}

FrameTranslationCost::FrameTranslationCost(
    const std::string& frame_name, const Eigen::Vector3d& p_ref, const MatrixXd& weight)
    : AbstractCost(weight), _ft(frame_name, p_ref) {
    _name = "frame_translation_cost(" + frame_name + ")";
}

void FrameTranslationCost::allocate_dims() {
    _ft.allocate_slices();

    _output_dim = _ft.get_output_dim();
    _input_dim = _ft.get_input_dim();
}

void FrameTranslationCost::evaluate_impl() {
    _ft.evaluate();
    _value = _ft.get_value();
}

void FrameTranslationCost::jacobian_impl() {
    _ft.jacobian();
    // Copy x-block into our jacobian (u-block stays zero from allocate_slices)
    _jacobian.leftCols(_node->ndx()) = _ft.get_jac_x();
}

MatrixXdConstRef FrameTranslationCost::get_jac_x() const {
    return _jacobian.leftCols(_node->ndx());
}

MatrixXdConstRef FrameTranslationCost::get_jac_u() const {
    return MatrixXd::Zero(_output_dim, _node->ndu());
}
