#include <trajopt/constraints/friction_cone_constraint.h>
#include <stdexcept>

#include <pinocchio/algorithm/frames-derivatives.hpp>
#include <pinocchio/spatial/skew.hpp>
#include <pinocchio/spatial/se3.hpp>

FrictionConeConstraint::FrictionConeConstraint(const std::string& frame_name, double mu)
    : AbstractConstraint(), _frame_name(frame_name), _mu(mu),
      _frame_id(-1), _contact_idx(-1) {
    _name = "friction_cone_constraint(" + frame_name + ")";
}

void FrictionConeConstraint::allocate_dims() {
    // Look up frame ID from model
    if (!_node->model().existFrame(_frame_name)) {
        throw std::runtime_error("FrictionConeConstraint: frame '" + _frame_name +
                               "' does not exist in model");
    }
    _frame_id = _node->model().getFrameId(_frame_name);

    // Find contact index in node's contact list
    _contact_idx = -1;
    const auto& contacts = _node->contacts();
    for (int i = 0; i < contacts.size(); ++i) {
        if (contacts[i].name == _frame_name) {
            _contact_idx = i;
            break;
        }
    }

    if (_contact_idx == -1) {
        throw std::runtime_error("FrictionConeConstraint: frame '" + _frame_name +
                               "' not registered as contact in node. Call node->add_contact() first.");
    }

    const int nv = _node->nv();

    // Fixed dimension: 5 constraints (pyramid approximation)
    // Constraint vector: g = [f_world_z; μ*f_world_z - f_world_x; μ*f_world_z + f_world_x;
    //                         μ*f_world_z - f_world_y; μ*f_world_z + f_world_y]
    _output_dim = 5;
    _input_dim = _node->ndx() + _node->ndu();  // [q; v]

    // Allocate scratch buffers
    _Jframe.resize(6, nv);
    _R_world.setIdentity();
    _Adj_rot.setIdentity();
    _f_local.setZero();
    _f_world.setZero();

    // Initialize bounds (will be updated in evaluate_impl based on contact state)
    _lower_bound.resize(5);
    _upper_bound.resize(5);
}

void FrictionConeConstraint::evaluate_impl() {
    // Ensure FK and frame placements are computed
    _node->require_frame_placements();

    // Get force in local frame from control vector
    _f_local = _node->fc(_contact_idx);


    _R_world = _node->data().oMf[_frame_id].rotation();

    // Transform force to world frame using rotation matrix
    // For linear forces (no moment), f_world = R * f_local
    _f_world = _R_world * _f_local;

    // Constraint values (pyramid approximation)
    _value(0) = _f_world(2);                        // fz >= 0
    _value(1) = _mu * _f_world(2) - _f_world(0);    // μ*fz - fx >= 0
    _value(2) = _mu * _f_world(2) + _f_world(0);    // μ*fz + fx >= 0
    _value(3) = _mu * _f_world(2) - _f_world(1);    // μ*fz - fy >= 0
    _value(4) = _mu * _f_world(2) + _f_world(1);    // μ*fz + fy >= 0

    // Update bounds based on contact state
    if (_node->contacts()[_contact_idx].active) {
        // Active contact: enforce friction cone
        _lower_bound.setZero();
        _upper_bound.setConstant(1e8);
    } else {
        // Inactive contact: relax constraints
        _lower_bound.setConstant(-1e8);
        _upper_bound.setConstant(1e8);
    }
}

void FrictionConeConstraint::jacobian_impl() {
    const int nv = _node->nv();
    const int ndx = _node->ndx();
    const int ndu = _node->ndu();
    const int force_idx = _node->nv() + 3 * _contact_idx;

    // Ensure FK derivatives computed (includes FK + joint Jacobians)
    _node->require_fk_derivatives();

    _jacobian.setZero();

    // Use fast GET API (joint Jacobians already computed)
    pinocchio::getFrameJacobian(
        _node->model(), _node->data(),
        _frame_id,
        pinocchio::LOCAL,
        _Jframe
    );

    // Derivative of world force w.r.t. q:
    // ∂f_world/∂q = ∂(R * f_local)/∂q = ∂R/∂q * f_local
    // Using the relationship: ∂(R * f)/∂q = -R * skew(f) * J_angular
    MatrixXd dwf_dq = -_R_world * pinocchio::skew(_f_local) * _Jframe.bottomRows(3);


    _jacobian.block(0, 0, 1, nv) = dwf_dq.row(2);
    _jacobian.block(0, ndx + force_idx, 1, 3) = _R_world.row(2);

    _jacobian.block(1, 0, 1, nv) = _mu * dwf_dq.row(2) - dwf_dq.row(0);
    _jacobian.block(1, ndx + force_idx, 1, 3) = _mu * _R_world.row(2) - _R_world.row(0);

    _jacobian.block(2, 0, 1, nv) = _mu * dwf_dq.row(2) + dwf_dq.row(0);
    _jacobian.block(2, ndx + force_idx, 1, 3) = _mu * _R_world.row(2) + _R_world.row(0);

    _jacobian.block(3, 0, 1, nv) = _mu * dwf_dq.row(2) - dwf_dq.row(1);
    _jacobian.block(3, ndx + force_idx, 1, 3) = _mu * _R_world.row(2) - _R_world.row(1);

    _jacobian.block(4, 0, 1, nv) = _mu * dwf_dq.row(2) + dwf_dq.row(1);
    _jacobian.block(4, ndx + force_idx, 1, 3) = _mu * _R_world.row(2) + _R_world.row(1);
}

MatrixXdConstRef FrictionConeConstraint::get_jac_x() const {
    return _jacobian.leftCols(_node->ndx());
}

MatrixXdConstRef FrictionConeConstraint::get_jac_u() const {
    return _jacobian.rightCols(_node->ndu());
}
