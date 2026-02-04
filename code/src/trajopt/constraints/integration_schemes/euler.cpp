#include <trajopt/constraints/integration_schemes/euler.h>
#include <trajopt/node.h>

EulerIntegration::EulerIntegration(double dt) : AbstractConstraint(), _dt(dt) {
    _name = "euler_integration";

    set_equality_to_zero();

}

// void EulerIntegration::allocate_slices() {
//     auto& reg = _node->get_registry();

//     // Get dimensions from node
//     int nq = _node->get_nq();
//     int nv = _node->get_nv();

//     // Allocate/retrieve variable slices (deduplicates if already allocated)
//     _conf_slice = &reg.allocate_var(VarType::CONF, nq);
//     _vel_slice = &reg.allocate_var(VarType::VEL, nv);
//     _acc_slice = &reg.allocate_var(VarType::ACC, nv);

//     // Allocate constraint slice
//     _output_dim = nq + nv;  // Residual: [q_error; v_error]
//     reg.allocate_constraint(_name, _output_dim);

//     // Pre-allocate temporaries
//     _q_integrated.resize(nq);
//     _v_dt.resize(nv);
//     _J_q.resize(nv, nq);
//     _J_v.resize(nv, nv);
// }

void EulerIntegration::evaluate(VectorXdRef output) {
    _q       = _node->q();
    _vq      = _node->v();
    _aq      = _node->u();

    output.resize(2 * _node->nv());

    pinocchio::integrate(_node->model(), _q, _vq * _dt,
                         output.head(_node->nv()));

    // Position residual: q_next ⊖ q_integrated
    // pinocchio::difference(_node->model(), _q_integrated, _q_next,
    //                       output.head(_node->nv()));

    // Velocity residual: v_next - v - dt * a
    output.tail(_node->nv()) = _vq + _dt * _aq;
}

void EulerIntegration::jacobian(MatrixXdRef jac) {
    _q = _node->q();
    _vq = _node->v();
    _aq = _node->u();

    auto nv = _node->nv();

    jac.setZero();

    // Pre-allocate Jacobian matrices if needed
    if (_J_q.rows() != nv || _J_q.cols() != nv) {
        _J_q.resize(nv, nv);
    }
    if (_J_v.rows() != nv || _J_v.cols() != nv) {
        _J_v.resize(nv, nv);
    }
    _J_q.setZero();
    _J_v.setZero();

    // Jacobians of integrate(q, dt*v)
    pinocchio::dIntegrate(_node->model(), _q, _vq*_dt, _J_q,
                          pinocchio::ArgumentPosition::ARG0);
    pinocchio::dIntegrate(_node->model(), _q, _vq*_dt, _J_v,
                          pinocchio::ArgumentPosition::ARG1);

    jac.block(0,0,nv, nv) = _J_q;
    jac.block(0, nv, nv, nv) = _J_v * _dt;
    jac.block(nv, nv, nv,nv) = MatrixXd::Identity(nv, nv);
    jac.block(nv, 2*nv, nv, nv)= _dt * MatrixXd::Identity(nv, nv);
}
