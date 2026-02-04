#include <trajopt/node.h>


Node::Node(pinocchio::Model mdl)
    : _model_ptr(std::make_shared<pinocchio::Model>(std::move(mdl))),
      _data(*_model_ptr) {
    _nq = _model_ptr->nq;
    _nv = _model_ptr->nv;

    _x.resize(_nq + _nv);  // State: [q, v]
    _u.resize(_nv);        // Control: acceleration by default
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
