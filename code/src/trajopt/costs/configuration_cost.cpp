#include <trajopt/costs/configuration_cost.h>
#include <trajopt/node.h>

#include <pinocchio/algorithm/joint-configuration.hpp>

ConfigurationCost::ConfigurationCost(const VectorXd& q_ref, double weight)
    : AbstractCost(weight), _q_ref(q_ref) {
    _name = "configuration_cost";
}

ConfigurationCost::ConfigurationCost(const VectorXd& q_ref, const MatrixXd& weight)
    : AbstractCost(weight), _q_ref(q_ref) {
    _name = "configuration_cost";
}

void ConfigurationCost::allocate_dims() {
    int nv = _node->nv();

    // Detect floating base (check once and cache)
    _fb_nv = 0;
    if (_node->model().njoints > 1 &&
        _node->model().joints[1].shortname() == "JointModelFreeFlyer") {
        _fb_nv = _node->model().joints[1].nv();  // 6
    }

    _output_dim = nv - _fb_nv;
    _input_dim = _node->ndx() + _node->ndu();
}

void ConfigurationCost::evaluate_impl() {
    // Simple vector difference for joint positions (no SE3 operations needed)
    _value = _node->q().tail(_output_dim) - _q_ref.tail(_output_dim);
}

void ConfigurationCost::jacobian_impl() {
    _jacobian.setZero();
    // Jacobian is identity for joint positions: ∂(q_joints - q_ref_joints)/∂q_joints = I
    _jacobian.block(0, _fb_nv, _output_dim, _output_dim).setIdentity();
}

MatrixXdConstRef ConfigurationCost::get_jac_x() const {
    return _jacobian.leftCols(_node->ndx());
}

MatrixXdConstRef ConfigurationCost::get_jac_u() const {
    return _jacobian.rightCols(_node->ndu());
}
