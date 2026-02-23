#include <trajopt/constraints/contact_constraint.h>
#include <stdexcept>

#include <pinocchio/algorithm/frames-derivatives.hpp>

ContactConstraint::ContactConstraint(const std::string& frame_name,
                                     double ground_height)
    : AbstractConstraint(), _frame_name(frame_name), _ground_height(ground_height),
      _frame_id(-1), _contact_idx(-1) {
    _name = "contact_constraint(" + frame_name + ")";
}

void ContactConstraint::allocate_dims() {
    // Look up frame ID from model
    if (!_node->model().existFrame(_frame_name)) {
        throw std::runtime_error("ContactConstraint: frame '" + _frame_name +
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
        throw std::runtime_error("ContactConstraint: frame '" + _frame_name +
                               "' not registered as contact in node. Call node->add_contact() first.");
    }

    const int nv = _node->nv();

    // Fixed dimension: always 4 constraints
    // Constraint vector: g = [p_z - ground_height; v_x; v_y; v_z]
    _output_dim = 4;
    _input_dim = _node->ndx() + _node->ndu();  // [q; v; u]

    // Allocate scratch buffers
    _Jframe.resize(6, nv);
    _Jvf_dq.resize(6, nv);
    _Jvf_dv.resize(6, nv);

    // Initialize bounds (will be updated in evaluate_impl based on contact state)
    _lower_bound.resize(4);
    _upper_bound.resize(4);
}

void ContactConstraint::evaluate_impl() {
    // Ensure FK and frame placements are computed
    _node->require_frame_placements();

    // Constraint value: [p_z - ground_height; v_frame]
    _value(0) = _node->data().oMf[_frame_id].translation()(2) - _ground_height;  // z position error
    _value.segment<3>(1) = pinocchio::getFrameVelocity(_node->model(), _node->data(), _frame_id, pinocchio::ReferenceFrame::LOCAL).toVector().head(3);            // 3D velocity


    
    // Update bounds based on contact state
    if (_node->contacts()[_contact_idx].active) {
        _lower_bound.setZero();//* (-1e-2);
        _upper_bound.setZero();// = VectorXd::setOnes(_output_dim)* 1e-2;
    } else {
        _lower_bound(0) = 0.0;
        _upper_bound(0) = 1e8;

        _lower_bound.segment<3>(1).setConstant(-1e8);
        _upper_bound.segment<3>(1).setConstant(1e8);
    }
}

void ContactConstraint::jacobian_impl() {
    // Ensure FK derivatives computed (includes FK + joint Jacobians)
    _node->require_fk_derivatives();

    _jacobian.setZero();
    _Jvf_dq.setZero();
    _Jvf_dv.setZero();

    // Use fast GET API (joint Jacobians already computed)
    pinocchio::getFrameJacobian(
        _node->model(), _node->data(),
        _frame_id,
        pinocchio::LOCAL_WORLD_ALIGNED,
        _Jframe
    );

    // Get velocity derivatives (FK derivatives already computed)
    pinocchio::getFrameVelocityDerivatives(
        _node->model(), _node->data(), _frame_id,
        pinocchio::ReferenceFrame::LOCAL, _Jvf_dq, _Jvf_dv);

    _jacobian.block(0, 0, 1, _node->nv()) = _Jframe.row(2);
    _jacobian.block(1, 0, 3, _node->nv()) = _Jvf_dq.topRows(3);
    _jacobian.block(1, _node->nv(), 3, _node->nv()) = _Jvf_dv.topRows(3);
}
