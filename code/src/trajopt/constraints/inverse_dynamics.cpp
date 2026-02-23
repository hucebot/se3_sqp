#include <trajopt/constraints/inverse_dynamics.h>

InvDynamics::InvDynamics() : AbstractConstraint() {
    _name = "inverse_dynamics";
    // Note: Allocation deferred to allocate_slices() since _node is not set yet
}

void InvDynamics::allocate_dims() {
    int nv = _node->nv();
    int nc = _node->n_contacts();

    _output_dim = nv;
    _input_dim = _node->ndx() + _node->ndu();  // [q_k, v_k, a_k, f_1, ..., f_nc]

    _dtau_dq.resize(nv, nv);
    _dtau_dv.resize(nv, nv);
    _dtau_da.resize(nv, nv);

    // Allocate external forces vector (size = model.njoints, all zero)
    _fext.resize(_node->model().njoints, pinocchio::Force::Zero());

    // Allocate frame Jacobian scratch buffer
    _Jframe.resize(6, nv);

    // Set bounds from model effort limits
    const VectorXd& effort = _node->model().effortLimit;
    _lower_bound = -effort;
    _upper_bound =  effort;

    // Floating base: unactuated DOFs → tau must be zero
    if (_node->model().njoints > 1 &&
        _node->model().joints[1].shortname() == "JointModelFreeFlyer") {
        int fb_nv = _node->model().joints[1].nv();  // 6 for free-flyer
        _lower_bound.head(fb_nv).setZero();
        _upper_bound.head(fb_nv).setZero();
    }

    _node->link_tau(&_value);

}

void InvDynamics::build_fext() {
    // Zero all forces first
    for (auto& force : _fext) {
        force.setZero();
    }

    // Add active contact forces
    // Forces are expressed in the contact frame's local coordinates (3D linear force, no torque)
    const auto& contacts = _node->contacts();
    for (int i = 0; i < _node->n_contacts(); ++i) {
        if (!contacts[i].active) continue;

        // Get 3D force in frame's local coordinates
        Vector3d f_local = _node->fc(i);

        // Create spatial force (linear force only, zero torque)
        pinocchio::Force f_contact(f_local, Vector3d::Zero());

        // Get frame placement relative to parent joint
        const auto& frame = _node->model().frames[contacts[i].frame_id];
        const pinocchio::SE3& jMf = frame.placement;

        // Transform force from frame coordinates to parent joint coordinates using SE3 action
        // This properly accounts for both rotation and translation (lever arm)
        _fext[contacts[i].parent_joint_id] += jMf.act(f_contact);
    }
}

void InvDynamics::evaluate_impl() {
    _q  = _node->q();
    _vq = _node->v();
    _aq = _node->a();

    // Ensure FK and frame placements are computed (cached on node)
    _node->require_frame_placements();

    // Build external forces from active contacts
    build_fext();

    // Compute inverse dynamics with external forces
    pinocchio::rnea(_node->model(), _node->data(), _q, _vq, _aq, _fext);
    _value = _node->data().tau;
}

void InvDynamics::jacobian_impl() {
    _q  = _node->q();
    _vq = _node->v();
    _aq = _node->a();

    _jacobian.setZero();
    _dtau_dq.setZero();
    _dtau_dv.setZero();
    _dtau_da.setZero();

    // Compute RNEA derivatives with external forces (fext already built in evaluate_impl)
    pinocchio::computeRNEADerivatives(_node->model(), _node->data(), _q, _vq, _aq, _fext, _dtau_dq, _dtau_dv, _dtau_da);
    _dtau_da.template triangularView<Eigen::StrictlyLower>() = _dtau_da.transpose().template triangularView<Eigen::StrictlyLower>();

    // Fill state and acceleration Jacobian blocks
    _jacobian.block(0, 0,             _node->nv(), _node->nv()) = _dtau_dq;
    _jacobian.block(0, _node->nv(),   _node->nv(), _node->nv()) = _dtau_dv;
    _jacobian.block(0, 2*_node->nv(), _node->nv(), _node->nv()) = _dtau_da;

    // Ensure FK derivatives computed (includes FK + joint Jacobians)
    _node->require_fk_derivatives();

    // Compute ∂τ/∂f for each active contact
    // Forces are expressed in frame local coordinates, so we need Jacobian in LOCAL frame
    const auto& contacts = _node->contacts();
    for (int i = 0; i < _node->n_contacts(); ++i) {
        if (!contacts[i].active) {
            // Inactive contact: Jacobian columns are zero (already set by setZero above)
            continue;
        }

        // Use fast GET API (joint Jacobians already computed)
        _Jframe.setZero();
        pinocchio::getFrameJacobian(
            _node->model(), _node->data(),
            contacts[i].frame_id,
            pinocchio::LOCAL,  // LOCAL frame because forces are in frame coordinates
            _Jframe
        );

        // Extract linear part (top 3 rows) and transpose to get (nv × 3)
        // ∂τ/∂f = -J^T because external forces reduce required torques
        _jacobian.block(0, 2*_node->nv() + _node->nv() + 3*i, _node->nv(), 3) = -_Jframe.topRows(3).transpose();
    }
}
