#include <trajopt/costs/configuration_cost.h>
#include <trajopt/node.h>

ConfigurationCost::ConfigurationCost(const VectorXd& q_ref)
    : AbstractCost(), _q_ref(q_ref) {
    _name = "configuration_cost";
}

void ConfigurationCost::allocate_slices() {
    int nv = _node->nv();

    // Residual lives in tangent space: q ⊖ q_ref (dimension nv)
    _output_dim = nv;
    // Input is full state [q; v], dimension 2*nv (no control dependency)
    _input_dim = 2 * nv;

    _J_dq.resize(nv, nv);

    // Jacobian: (nv × 2*nv), right half (∂r/∂v) stays zero
    _jacobian.resize(_output_dim, _input_dim);
    _jacobian.setZero();
}

void ConfigurationCost::evaluate_impl(VectorXdRef output) {
    // r = q ⊖ q_ref  (tangent vector from q_ref to q)
    pinocchio::difference(_node->model(), _q_ref, _node->q(), output);
}

void ConfigurationCost::jacobian_impl(MatrixXdRef jac) {
    jac.setZero();
    _J_dq.setZero();

    // ∂(difference(q_ref, q))/∂q  →  derivative w.r.t. second argument (ARG1)
    pinocchio::dDifference(_node->model(), _q_ref, _node->q(), _J_dq,
                           pinocchio::ArgumentPosition::ARG1);

    // Fill left block: ∂r/∂q
    jac.leftCols(_node->nv()) = _J_dq;
    // Right block: ∂r/∂v = 0 (already zero from setZero)
}

MatrixXdConstRef ConfigurationCost::get_jac_x() const {
    return _jacobian;
}

MatrixXd ConfigurationCost::get_jac_u() const {
    return MatrixXd::Zero(_output_dim, 0);
}
