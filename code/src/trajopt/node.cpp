#include <trajopt/node.h>


Node::Node(pinocchio::Model mdl)
    : _model_ptr(std::make_shared<pinocchio::Model>(std::move(mdl))),
      _data_ptr(std::make_unique<pinocchio::Data>(*_model_ptr)),
      _x_ptr(nullptr),
      _u_ptr(nullptr) {
    _nq = _model_ptr->nq;
    _nv = _model_ptr->nv;
    // _x and _u are bound later via bind_trajectory()
}

void Node::bind_trajectory(VectorXd* x, VectorXd* u) {
    _x_ptr = x;
    _u_ptr = u;
}

void Node::cached_update(){
    //TODO - update the pinocchio data and cache the computations
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
