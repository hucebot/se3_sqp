#include <trajopt/node.h>
#include <cmath>

Node::Node(pinocchio::Model mdl)
    : _model_ptr(std::make_shared<pinocchio::Model>(std::move(mdl))),
      _data_ptr(std::make_unique<pinocchio::Data>(*_model_ptr)),
      _x_ptr(nullptr),
      _u_ptr(nullptr) {
    _nq = _model_ptr->nq;
    _nv = _model_ptr->nv;
    // _x and _u are bound later via bind_trajectory()

    _cost = 0.;
    _defect = 0.;
    _violation = 0.;

    _grad_cost_x.setZero(ndx());
    _grad_cost_u.setZero(ndu());
    _grad_defect_x.setZero(ndx());
    _grad_defect_u.setZero(ndu());
    _grad_violation_x.setZero(ndx());
    _grad_violation_u.setZero(ndu());
    _viol_tmp.resize(0);
}

void Node::bind_trajectory(VectorXd* x, VectorXd* u) {
    _x_ptr = x;
    _u_ptr = u;
}

void Node::cached_update(){
    //TODO - update the pinocchio data and cache the computations
}

void Node::x_oplus(VectorXdRef x0, VectorXdRef dx, VectorXdRef x1){
    // x = [q; v],  dx = [dq; dv]  (both tangent vectors have size nv)
    auto q0 = x0.head(_nq);
    auto v0 = x0.tail(_nv);
    auto dq = dx.head(_nv);
    auto dv = dx.tail(_nv);

    pinocchio::integrate(*_model_ptr, q0, dq, x1.head(_nq));
    x1.tail(_nv) = v0 + dv;
}

void Node::u_oplus(VectorXdRef u0, VectorXdRef du, VectorXdRef u1) {
    // Control u = a (acceleration) lives in Euclidean space R^nv
    u1 = u0 + du;
}

void Node::add_cost(std::shared_ptr<AbstractCost> cost) {
    _cost_list.push_back(cost);
    cost->set_node(this);
    cost->allocate_slices();
}

void Node::add_constraint(std::shared_ptr<AbstractConstraint> constraint) {
    _constraint_list.push_back(constraint);
    constraint->set_node(this);
    constraint->allocate_slices();
}

void Node::add_dynamics(std::shared_ptr<AbstractConstraint> constraint) {
    _dynamics = constraint;
    constraint->set_node(this);
    constraint->allocate_slices();
}

nlohmann::json Node::to_json() const {
    nlohmann::json j;
    auto q_vec = q();
    auto v_vec = v();
    j["q"] = std::vector<double>(q_vec.data(), q_vec.data() + q_vec.size());
    j["v"] = std::vector<double>(v_vec.data(), v_vec.data() + v_vec.size());
    if (_u_ptr && _u_ptr->size() > 0) {
        auto u_vec = u();
        j["u"] = std::vector<double>(u_vec.data(), u_vec.data() + u_vec.size());
    }
    return j;
}

void Node::rebind_constraints() {
    // Update all constraints' node pointers to this node
    if (_dynamics) {
        _dynamics->set_node(this);
    }
    for (auto& constraint : _constraint_list) {
        constraint->set_node(this);
    }
    for (auto& cost : _cost_list) {
        cost->set_node(this);
    }
}

void Node::calc_cost(){
    _cost = 0.;
    for (auto& cost: _cost_list){
        const VectorXd& val = cost->get_value();
        _cost += 0.5 * val.dot(cost->get_weight() * val);
    }
}

void Node::calc_dynamics_defect(){
    _defect = 0.;
    if (_dynamics){
        _defect = _dynamics->get_value().cwiseAbs().maxCoeff();
    }
}

void Node::calc_constraint_violation(){
    _violation = 0.;
    for (auto& constraint: _constraint_list){
        const VectorXd& val = constraint->get_value();
        const VectorXd& lb = constraint->get_lower_bound();
        const VectorXd& ub = constraint->get_upper_bound();
        _violation += ((lb - val).cwiseMax(0.0) + (val - ub).cwiseMax(0.0)).maxCoeff();
    }
}

//TODO fix weights
void Node::calc_cost_gradient() {
    _grad_cost_x.setZero();
    _grad_cost_u.setZero();
    for (const auto& cost : _cost_list) {
        const VectorXd& f = cost->get_value();
        // get_jac_x() returns MatrixXdConstRef (Eigen::Ref wrapper) — no data copy
        auto Jx = cost->get_jac_x();
        _grad_cost_x.noalias() +=  cost->get_weight() * (Jx.transpose() * f);
        // get_jac_u() returns MatrixXd by value; cols()==0 means no control dep
        auto Ju = cost->get_jac_u();
        if (Ju.cols() > 0) {
            _grad_cost_u.noalias() +=  cost->get_weight() * (Ju.transpose() * f);
        }
    }
}

void Node::calc_defect_gradient() {
    _grad_defect_x.setZero();
    _grad_defect_u.setZero();
    if (!_dynamics) return;

    const VectorXd& g = _dynamics->get_value();
    if (g.size() == 0) return;

    // Subgradient of ||g||_inf selects the max-magnitude component
    int k_star;
    g.cwiseAbs().maxCoeff(&k_star);
    const double sign_g = (g(k_star) >= 0.0) ? 1.0 : -1.0;

    auto Jx = _dynamics->get_jac_x();
    _grad_defect_x.noalias() = sign_g * Jx.row(k_star).transpose();

    auto Ju = _dynamics->get_jac_u();
    if (Ju.cols() > 0) {
        _grad_defect_u.noalias() = sign_g * Ju.row(k_star).transpose();
    }
}

void Node::calc_violation_gradient() {
    _grad_violation_x.setZero();
    _grad_violation_u.setZero();
    for (const auto& constraint : _constraint_list) {
        const VectorXd& g  = constraint->get_value();
        const VectorXd& lb = constraint->get_lower_bound();
        const VectorXd& ub = constraint->get_upper_bound();

        // Reuse scratch buffer — resize only when the output dimension changes
        _viol_tmp.resize(g.size());
        _viol_tmp = (lb - g).cwiseMax(0.0) + (g - ub).cwiseMax(0.0);

        int k_star;
        const double max_viol = _viol_tmp.maxCoeff(&k_star);
        if (max_viol <= 0.0) continue;

        auto Jx = constraint->get_jac_x();
        auto Ju = constraint->get_jac_u();

        // Subgradient sign: lb-g active → d/dx = -Jx row; g-ub active → +Jx row
        const double sign = (g(k_star) < lb(k_star)) ? -1.0 : 1.0;
        _grad_violation_x.noalias() += sign * Jx.row(k_star).transpose();
        if (Ju.cols() > 0) {
            _grad_violation_u.noalias() += sign * Ju.row(k_star).transpose();
        }
    }
}



