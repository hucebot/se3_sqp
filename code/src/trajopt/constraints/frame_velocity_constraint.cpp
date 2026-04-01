#include <trajopt/constraints/frame_velocity_constraint.h>
#include <trajopt/node.h>
#include <pinocchio/algorithm/frames-derivatives.hpp>

FrameVelocityConstraint::FrameVelocityConstraint(const std::string& frame_name, const Vector6d& v_ref):
    AbstractConstraint(),
    _v_ref(v_ref),
    _frame_name(frame_name),
    _re_ref_frame(pinocchio::LOCAL_WORLD_ALIGNED),
    _base_frame_name("base_link")
{
    _name = "frame_velocity_constraint(" + frame_name + ")";
    _frame_id = -1;
}

void FrameVelocityConstraint::allocate_dims() {

    _frame_id = _node->model().getFrameId(_frame_name);

    _output_dim = 6;
    _input_dim = _node->ndx() + _node->ndu();

    // Initialize bounds (will be updated in evaluate_impl based on contact state)
    _lower_bound.setZero(_output_dim);
    _upper_bound.setZero(_output_dim);

}

void FrameVelocityConstraint::evaluate_impl() {
    // Ensure FK and frame placements are computed
    _node->require_frame_placements();

    pinocchio::Motion vel;
    if(_frame_name == _base_frame_name)
        vel = _node->data().v[1];
    else
        vel = pinocchio::getFrameVelocity(_node->model(), _node->data(), _frame_id, _re_ref_frame);

    _value.setZero(6);
    _value.segment(0, 3) = vel.linear();
    _value.segment(3, 3) = vel.angular();
}

void FrameVelocityConstraint::jacobian_impl() {
    Eigen::MatrixXd dv_dq;
    dv_dq.setZero(_output_dim, _node->nv());
    Eigen::MatrixXd dv_dqdot;
    dv_dqdot.setZero(_output_dim, _node->nv());

    if(_frame_name == _base_frame_name)
        pinocchio::getFrameVelocityDerivatives(_node->model(), _node->data(), _frame_id, pinocchio::LOCAL, dv_dq, dv_dqdot);
    else
        pinocchio::getFrameVelocityDerivatives(_node->model(), _node->data(), _frame_id, _re_ref_frame, dv_dq, dv_dqdot);

    _jacobian.setZero();
    _jacobian.block(0, 0,             6, _node->nv()) = dv_dq;
    _jacobian.block(0, _node->nv(),   6, _node->nv()) = dv_dqdot;
}

