#include <trajopt/costs/torque_cost.h>

TorqueCost::TorqueCost(const VectorXd& tau_ref, double weight)
    : AbstractCost(weight), _tau_ref(tau_ref) {
    _name = "torque_cost";
}
TorqueCost::TorqueCost(const VectorXd& tau_ref, const MatrixXd& weight)
    : AbstractCost(weight), _tau_ref(tau_ref) {
    _name = "torque_cost";
}
TorqueCost::TorqueCost(double weight)
    : AbstractCost(weight) {
    _name = "torque_cost";
}

void TorqueCost::allocate_dims() {
    _id.allocate_slices();

    int nv = _node->nv();

    // Detect floating base (check once and cache)
    _fb_nv = 0;
    if (_node->model().njoints > 1 &&
        _node->model().joints[1].shortname() == "JointModelFreeFlyer") {
        _fb_nv = _node->model().joints[1].nv();  // 6
    }

    _output_dim = nv - _fb_nv;
    _input_dim = _node->ndx() + _node->ndu();

    if (_tau_ref.size() == 0) {
        _tau_ref.resize(_output_dim);
        _tau_ref.setZero();
    }
}

void TorqueCost::evaluate_impl() {
    _id.evaluate_impl();

    // Exclude floating base accelerations
    _value = _id.get_value().segment(_fb_nv, _node->nv() - _fb_nv) - _tau_ref;
}

void TorqueCost::jacobian_impl() {
    _id.jacobian();

    _jacobian.setZero();
    // Jacobian (exclude floating base):
    _jacobian.leftCols(_node->ndx()) = _id.get_jac_x().bottomRows(_output_dim);
    _jacobian.rightCols(_node->ndu()) = _id.get_jac_u().bottomRows(_output_dim);
}
