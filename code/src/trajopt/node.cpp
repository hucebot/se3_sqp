#include <trajopt/node.h>


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
