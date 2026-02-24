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
    invalidate_cache();
}

void Node::invalidate_cache(){
    _cache_flags = 0;
}

void Node::require_forward_kinematics(){
    if (!(_cache_flags & CACHE_FK)) {
        pinocchio::forwardKinematics(*_model_ptr, *_data_ptr, q(), v());
        _cache_flags |= CACHE_FK;
    }
}

void Node::require_frame_placements(){
    require_forward_kinematics();
    if (!(_cache_flags & CACHE_FRAME_PLACEMENTS)) {
        pinocchio::updateFramePlacements(*_model_ptr, *_data_ptr);
        _cache_flags |= CACHE_FRAME_PLACEMENTS;
    }
}

void Node::require_fk_derivatives(){
    if (!(_cache_flags & CACHE_FK_DERIVATIVES)) {
        // Computes FK + joint Jacobians + derivatives all at once
        // Terminal node has no control, so use zero acceleration
        if (_u_ptr) {
            pinocchio::computeForwardKinematicsDerivatives(*_model_ptr, *_data_ptr, q(), v(), a());
        } else {
            VectorXd a_zero = VectorXd::Zero(_nv);
            pinocchio::computeForwardKinematicsDerivatives(*_model_ptr, *_data_ptr, q(), v(), a_zero);
        }
        _cache_flags |= CACHE_FK_DERIVATIVES | CACHE_FK;

        // Update frame placements (oMf) - required by constraints that access data.oMf[]
        if (!(_cache_flags & CACHE_FRAME_PLACEMENTS)) {
            pinocchio::updateFramePlacements(*_model_ptr, *_data_ptr);
            _cache_flags |= CACHE_FRAME_PLACEMENTS;
        }
    }
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

void Node::add_contact(const std::string& frame_name) {
    // Check if contact already registered
    if (_contact_name_to_idx.count(frame_name) > 0) {
        throw std::runtime_error("Contact '" + frame_name + "' already registered");
    }

    // Look up frame ID from model
    if (!_model_ptr->existFrame(frame_name)) {
        throw std::runtime_error("Frame '" + frame_name + "' does not exist in model");
    }
    int frame_id = _model_ptr->getFrameId(frame_name);
    int parent_joint_id = _model_ptr->frames[frame_id].parentJoint;

    Contact contact;
    contact.name = frame_name;
    contact.frame_id = frame_id;
    contact.parent_joint_id = parent_joint_id;
    contact.active = true;

    int idx = static_cast<int>(_contacts.size());
    _contacts.push_back(contact);
    _contact_name_to_idx[frame_name] = idx;

    // Resize gradient vectors since control dimension changed
    _grad_cost_u.conservativeResize(ndu());
    _grad_defect_u.conservativeResize(ndu());
    _grad_violation_u.conservativeResize(ndu());
    // Zero out new entries (3 new dimensions)
    _grad_cost_u.tail(3).setZero();
    _grad_defect_u.tail(3).setZero();
    _grad_violation_u.tail(3).setZero();
}

void Node::add_contacts(const std::vector<std::string>& frame_names) {
    for (const auto& name : frame_names) {
        add_contact(name);
    }
}

void Node::set_active_contacts(const std::vector<std::string>& names) {
    // Deactivate all contacts first
    for (auto& contact : _contacts) {
        contact.active = false;
    }

    // Activate the named contacts using fast lookup
    for (const auto& name : names) {
        auto it = _contact_name_to_idx.find(name);
        if (it == _contact_name_to_idx.end()) {
            throw std::runtime_error("Contact '" + name + "' not registered. Call add_contact() first.");
        }
        _contacts[it->second].active = true;
    }
}

nlohmann::json Node::to_json() const {
    nlohmann::json j;
    auto q_vec = q();
    auto v_vec = v();
    j["q"] = std::vector<double>(q_vec.data(), q_vec.data() + q_vec.size());
    j["v"] = std::vector<double>(v_vec.data(), v_vec.data() + v_vec.size());
    if (_u_ptr && _u_ptr->size() > 0) {
        auto a_vec = a();
        j["a"] = std::vector<double>(a_vec.data(), a_vec.data() + a_vec.size());

        // Serialize contact forces with frame names
        if (n_contacts() > 0) {
            nlohmann::json forces = nlohmann::json::object();
            for (int i = 0; i < n_contacts(); ++i) {
                auto f_vec = fc(i);
                forces[_contacts[i].name] = std::vector<double>(f_vec.data(), f_vec.data() + f_vec.size());
            }
            j["contact_forces"] = forces;
        }
    }

    // Serialize torques if inverse dynamics constraint is linked
    if (_tau_ptr && _tau_ptr->size() > 0) {
        j["tau"] = std::vector<double>(_tau_ptr->data(), _tau_ptr->data() + _tau_ptr->size());
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

void Node::calc_cost_gradient() {
    _grad_cost_x.setZero();
    _grad_cost_u.setZero();
    for (const auto& cost : _cost_list) {
        const VectorXd& f = cost->get_value();
        const MatrixXd& W = cost->get_weight();
        auto Jx = cost->get_jac_x();
        auto Ju = cost->get_jac_u();

        _grad_cost_x.noalias() += Jx.transpose() * W * f;
        if (Ju.cols() > 0) {
            _grad_cost_u.noalias() += Ju.transpose() * W * f;
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



