#include <trajopt/constraints/inverse_dynamics.h>

InvDynamics::InvDynamics() : AbstractConstraint() {
    _name = "inverse_dynamics";
    // Note: Allocation deferred to allocate_slices() since _node is not set yet
}

void InvDynamics::allocate_dims() {
    int nv = _node->nv();
    int nc = _node->n_contacts();

    _output_dim = nv;
    _input_dim = 2*nv + nv + 3*nc;  // [q_k(nv), v_k(nv), a_k(nv), f_1(3), ..., f_nc(3)]

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

    // Forward kinematics to get frame poses
    pinocchio::forwardKinematics(_node->model(), _node->data(), _q);
    pinocchio::updateFramePlacements(_node->model(), _node->data());

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

    // Compute ∂τ/∂f for each active contact
    // Forces are expressed in frame local coordinates, so we need Jacobian in LOCAL frame
    const auto& contacts = _node->contacts();
    for (int i = 0; i < _node->n_contacts(); ++i) {
        if (!contacts[i].active) {
            // Inactive contact: Jacobian columns are zero (already set by setZero above)
            continue;
        }

        // Compute frame Jacobian in LOCAL frame (frame's local coordinates)
        _Jframe.setZero();
        pinocchio::computeFrameJacobian(
            _node->model(), _node->data(), _q,
            contacts[i].frame_id,
            pinocchio::LOCAL,  // LOCAL frame because forces are in frame coordinates
            _Jframe
        );

        // Extract linear part (top 3 rows) and transpose to get (nv × 3)
        // ∂τ/∂f = -J^T because external forces reduce required torques
        _jacobian.block(0, 2*_node->nv() + _node->nv() + 3*i, _node->nv(), 3) = -_Jframe.topRows(3).transpose();
    }
}


MatrixXdConstRef InvDynamics::get_jac_x() const {
    // Return ∂g/∂x_k = [∂g/∂q_k | ∂g/∂v_k]
    return _jacobian.leftCols(2 * _node->nv());
}

MatrixXd InvDynamics::get_jac_u() const {
    // Return ∂g/∂u_k = [∂g/∂a_k | ∂g/∂f_1 | ... | ∂g/∂f_nc]
    // Columns from 2*nv onwards (acceleration + all contact forces)
    return _jacobian.rightCols(_node->nv() + 3 * _node->n_contacts());
}
