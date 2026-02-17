#include <trajopt/costs/velocity_cost.h>
#include <trajopt/node.h>

VelocityCost::VelocityCost(const VectorXd& v_ref)
    : AbstractCost(), _v_ref(v_ref) {
    _name = "velocity_cost";
}
VelocityCost::VelocityCost()
    : AbstractCost() {
    _name = "velocity_cost";
}

void VelocityCost::allocate_slices() {
    int nv = _node->nv();

    // Residual lives in tangent space: q ⊖ q_ref (dimension nv)
    _output_dim = nv;
    // Input is full state [q; v], dimension 2*nv (no control dependency)
    _input_dim = 2 * nv;

    if (_v_ref.size()==0){
        _v_ref.resize(_output_dim);
        _v_ref.setZero();
    }


    // Jacobian: (nv × 2*nv), right half (∂r/∂v) stays zero
    _value.resize(_output_dim);
    _jacobian.resize(_output_dim, _input_dim);
    _jacobian.setZero();
}

void VelocityCost::evaluate_impl() {
    _value = _node->v() - _v_ref;
}

void VelocityCost::jacobian_impl() {
    _jacobian.middleCols(_node->nv(), _node->nv()).setIdentity();
}

MatrixXdConstRef VelocityCost::get_jac_x() const {
    return _jacobian;
}

MatrixXd VelocityCost::get_jac_u() const {
    return MatrixXd::Zero(_output_dim, 0);
}
