#include <trajopt/constraints/frame_translation_constraint.h>

FrameTranslationConstraint::FrameTranslationConstraint(
    const std::string& frame_name, const Eigen::Vector3d& p_ref)
    : AbstractConstraint(), _ft(frame_name, p_ref) {
    _name = "frame_translation_constraint(" + frame_name + ")";
}

void FrameTranslationConstraint::allocate_dims() {
    _ft.set_node(_node);
    _ft.allocate_slices();

    _output_dim = _ft.get_output_dim();
    _input_dim = _ft.get_input_dim();

    set_equality_to_zero();
}

void FrameTranslationConstraint::evaluate_impl() {
    _ft.evaluate();
    _value = _ft.get_value();
}

void FrameTranslationConstraint::jacobian_impl() {
    _ft.jacobian();
    _jacobian.leftCols(_node->ndx()) = _ft.get_jac_x();
}

MatrixXdConstRef FrameTranslationConstraint::get_jac_x() const {
    return _jacobian.leftCols(_node->ndx());
}

MatrixXd FrameTranslationConstraint::get_jac_u() const {
    return MatrixXd::Zero(_output_dim, _node->ndu());
}
