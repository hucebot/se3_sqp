#include <trajopt/node.h>

Node::Node(int num_vars)
    : state_dim(0), control_dim(0), parameter_dim(0), registry(num_vars) {}

Node::~Node() {}

void Node::set_dimensions(int nx, int nu, int np) {
    state_dim = nx;
    control_dim = nu;
    parameter_dim = np;
}

void Node::add_cost(std::shared_ptr<AbstractCost> cost) {
    cost_list.push_back(cost);
    // Allocate slices immediately when adding the cost
    cost->allocate_slices(*this);
}

void Node::add_constraint(std::shared_ptr<AbstractConstraint> constraint) {
    constraint_list.push_back(constraint);
    // Allocate slices immediately when adding the constraint
    constraint->allocate_slices(*this);
}
