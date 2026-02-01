#include <trajopt/constraints/integration_schemes/euler.h>
#include <trajopt/node.h>

EulerIntegration::EulerIntegration(double dt) : AbstractConstraint(), _dt(dt) {
    _name = "euler_integration";
    set_equality_to_zero();
}

void EulerIntegration::allocate_slices() {
    auto& reg = _node->get_registry();

    // Get dimensions from node
    int nq = _node->get_nq();
    int nv = _node->get_nv();

    // Allocate/retrieve variable slices (deduplicates if already allocated)
    _conf_slice = &reg.allocate_var(VarType::CONF, nq);
    _vel_slice = &reg.allocate_var(VarType::VEL, nv);
    _acc_slice = &reg.allocate_var(VarType::ACC, nv);

    // Allocate constraint slice
    _output_dim = nq + nv;  // Residual: [q_error; v_error]
    reg.allocate_constraint(_name, _output_dim);

    // Pre-allocate temporaries
    _q_integrated.resize(nq);
    _v_dt.resize(nv);
    _J_q.resize(nv, nq);
    _J_v.resize(nv, nv);
}

void EulerIntegration::evaluate(VectorXdConstRef x, VectorXdConstRef u,
                                VectorXdRef residual) {
    // Get dimensions from cached slices
    int nq = _conf_slice->size;
    int nv = _vel_slice->size;

    // Extract variables (using knowledge of x = [q; v] layout)
    auto q = x.head(nq);
    auto v = x.segment(nq, nv);
    auto a = u.head(nv);

    // TODO: Get x_next from the linked node
    auto x_next = x.segment(nq+nv, nq+nv);
    auto q_next = x_next.head(nq);
    auto v_next = x_next.segment(nq, nv);

    // // q_integrated = q ⊕ (dt * v)
    pinocchio::integrate(*_node->_model_ptr, q, v*_dt, _q_integrated);

    // // Position residual: q_next ⊖ q_integrated
    pinocchio::difference(*_node->_model_ptr, _q_integrated, q_next,
                          residual.head(nv));

    // Velocity residual: v_next - v - dt * a
    residual.tail(nv) = v_next - v - _dt * a;
}

void EulerIntegration::jacobian(VectorXdConstRef x, VectorXdConstRef u,
                                MatrixXdRef jac_x, MatrixXdRef jac_u) {
    // Get dimensions from cached slices
    int nq = _conf_slice->size;
    int nv = _vel_slice->size;

    auto q = x.head(nq);
    auto v = x.segment(nq, nv);

    _v_dt = _dt * v;

    // Jacobians of integrate(q, dt*v)
    pinocchio::dIntegrate(*_node->_model_ptr, q, _v_dt, _J_q,
                          pinocchio::ArgumentPosition::ARG0);
    pinocchio::dIntegrate(*_node->_model_ptr, q, _v_dt, _J_v,
                          pinocchio::ArgumentPosition::ARG1);

    jac_x.setZero();
    // d(res_q)/dq = -J_q
    jac_x.topLeftCorner(nv, nq) = -_J_q;
    // d(res_q)/dv = -J_v * dt
    jac_x.block(0, nq, nv, nv) = -_J_v * _dt;
    // d(res_v)/dv = -I
    jac_x.bottomRightCorner(nv, nv) = -MatrixXd::Identity(nv, nv);

    jac_u.setZero();
    // d(res_v)/da = -dt * I
    jac_u.bottomRows(nv) = -_dt * MatrixXd::Identity(nv, nv);
}
