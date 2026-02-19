#include <trajopt/constraints/joint_limits_constraint.h>
#include <stdexcept>

JointLimitsConstraint::JointLimitsConstraint()
    : AbstractConstraint() {
    _name = "joint_limits_constraint";
}

void JointLimitsConstraint::allocate_dims() {
    const int nq = _node->nq();
    const int nv = _node->nv();

    // Constraint dimension: 2*nv (both position and velocity in tangent space)
    _output_dim = 2 * nv;
    _input_dim = _node->ndx() + _node->ndu();  // [q; v; u]

    // Validate position limits
    if (_node->model().lowerPositionLimit.size() != nq ||
        _node->model().upperPositionLimit.size() != nq) {
        throw std::runtime_error(
            "JointLimitsConstraint: model position limits dimension mismatch. "
            "Expected " + std::to_string(nq) + " limits.");
    }

    // Validate velocity limits
    if (_node->model().velocityLimit.size() != nv) {
        throw std::runtime_error(
            "JointLimitsConstraint: model velocity limits dimension mismatch. "
            "Expected " + std::to_string(nv) + " limits.");
    }

    // Check for invalid position limits (lower > upper)
    for (int i = 0; i < nq; ++i) {
        if (_node->model().lowerPositionLimit(i) > _node->model().upperPositionLimit(i)) {
            throw std::runtime_error(
                "JointLimitsConstraint: invalid position limits at index " +
                std::to_string(i) + ": lower (" +
                std::to_string(_node->model().lowerPositionLimit(i)) +
                ") > upper (" + std::to_string(_node->model().upperPositionLimit(i)) + ")");
        }
    }

    // Check for non-positive velocity limits
    for (int i = 0; i < nv; ++i) {
        if (_node->model().velocityLimit(i) <= 0.0) {
            throw std::runtime_error(
                "JointLimitsConstraint: velocity limit at index " +
                std::to_string(i) + " is non-positive (" +
                std::to_string(_node->model().velocityLimit(i)) + "). "
                "Velocity limits should be positive (symmetric).");
        }
    }

    // Allocate bounds vectors
    _lower_bound.resize(_output_dim);
    _upper_bound.resize(_output_dim);

    // Position bounds: project configuration limits onto tangent space
    // We compute the tangent vector from neutral config to the limits
    VectorXd q_lower_tangent(nv), q_upper_tangent(nv);
    pinocchio::difference(_node->model(), pinocchio::neutral(_node->model()),
                          _node->model().lowerPositionLimit, q_lower_tangent);
    pinocchio::difference(_node->model(), pinocchio::neutral(_node->model()),
                          _node->model().upperPositionLimit, q_upper_tangent);

    _lower_bound.head(nv) = q_lower_tangent;
    _upper_bound.head(nv) = q_upper_tangent;

    // Velocity limits (symmetric: -v_max to +v_max)
    _lower_bound.tail(nv) = -_node->model().velocityLimit;
    _upper_bound.tail(nv) = _node->model().velocityLimit;


}

void JointLimitsConstraint::evaluate_impl() {
    const int nv = _node->nv();

    // The constraint g(q) should be bounded: g_lower <= g(q) <= g_upper
    // We use the difference representation: the current q in tangent space
    // relative to the neutral configuration
    VectorXd q_tangent(nv);
    pinocchio::difference(_node->model(), pinocchio::neutral(_node->model()),
                          _node->q(), q_tangent);

    _value.head(nv) = q_tangent;
    _value.tail(nv) = _node->v();
}

void JointLimitsConstraint::jacobian_impl() {
    const int nv = _node->nv();

    _jacobian.setZero();

    MatrixXd dq_dv(nv, nv);
    dq_dv.setZero();
    pinocchio::dDifference(_node->model(), pinocchio::neutral(_node->model()), _node->q(), dq_dv,
                          pinocchio::ArgumentPosition::ARG1);


    // Position constraints: ∂(q_tangent)/∂q_tangent = I
    // Since we're already in tangent space, the Jacobian is identity
    _jacobian.block(0, 0, nv, nv) = dq_dv;

    // Velocity constraints: ∂v/∂v = I
    _jacobian.block(nv, nv, nv, nv).setIdentity();

    // ∂g/∂u = 0 (already set by setZero)
}

MatrixXdConstRef JointLimitsConstraint::get_jac_x() const {
    return _jacobian.leftCols(_node->ndx());
}

MatrixXd JointLimitsConstraint::get_jac_u() const {
    return _jacobian.rightCols(_node->ndu());
}
