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
        _cost += cost->get_weight() * val.squaredNorm();
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


// TODO calc derivatives



