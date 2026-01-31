#pragma once

#include <constraints/abstract_constraint.h>
#include <costs/abstract_cost.h>

#include <memory>
#include <string>
#include <vector>

/**
 * NodeSliceRegistry manages variable and constraint slice allocation
 * for a single node in the trajectory optimization problem.
 *
 * This decouples the Node from constraint internals by providing
 * a registry pattern for allocating slices of variables and constraints.
 */
class NodeSliceRegistry {
   private:
    int nv;           // Number of variables
    int var_offset;   // Current variable offset
    int con_offset;   // Current constraint offset

   public:
    NodeSliceRegistry(int num_vars)
        : nv(num_vars), var_offset(0), con_offset(0) {}

    /**
     * Allocate a variable slice with given name and size
     * Returns the starting offset and updates internal counter
     */
    int allocate_var_slice(const std::string& name, int size) {
        int start = var_offset;
        var_offset += size;
        return start;
    }

    /**
     * Allocate a constraint slice with given name and size
     * Returns the starting offset and updates internal counter
     */
    int allocate_con_slice(const std::string& name, int size) {
        int start = con_offset;
        con_offset += size;
        return start;
    }

    // Getters
    int get_var_offset() const { return var_offset; }
    int get_con_offset() const { return con_offset; }
    int get_num_vars() const { return nv; }

    // Reset offsets (useful for re-initialization)
    void reset() {
        var_offset = 0;
        con_offset = 0;
    }
};

/**
 * Node represents a single stage in the trajectory optimization problem.
 * It manages optimization variables, constraints, and costs.
 *
 * The Node uses a registry pattern for slice allocation, allowing
 * constraints to register their variable and constraint slices without
 * the Node needing to know constraint-specific details.
 */
class Node {
   private:
    // Placeholder for Pinocchio model
    void* model;  // Will be replaced with Pinocchio model type

    // Placeholder for dynamics
    std::shared_ptr<AbstractConstraint> dynamics;  // Will be replaced with dynamics type

    // Optimization variables dimensions
    int state_dim;
    int control_dim;
    int parameter_dim;

    // Slice registry for variable and constraint allocation
    NodeSliceRegistry registry;

    // Constraints and costs (stored as shared pointers)
    std::vector<std::shared_ptr<AbstractCost>> cost_list;
    std::vector<std::shared_ptr<AbstractConstraint>> constraint_list;

   public:
    Node(int num_vars = 0);
    ~Node();

    // Getters and setters for optimization variable dimensions
    void set_dimensions(int nx, int nu, int np);
    int get_state_dim() const { return state_dim; }
    int get_control_dim() const { return control_dim; }
    int get_parameter_dim() const { return parameter_dim; }

    // Registry access
    NodeSliceRegistry& get_registry() { return registry; }
    const NodeSliceRegistry& get_registry() const { return registry; }

    /**
     * Add a cost to the node and allocate its slices immediately.
     * The cost's allocate_slices() method is called automatically.
     *
     * @param cost Shared pointer to the cost to add
     */
    void add_cost(std::shared_ptr<AbstractCost> cost);

    /**
     * Add a constraint to the node and allocate its slices immediately.
     * The constraint's allocate_slices() method is called automatically.
     *
     * @param constraint Shared pointer to the constraint to add
     */
    void add_constraint(std::shared_ptr<AbstractConstraint> constraint);

    // Get total variable and constraint counts
    int get_num_vars() const { return registry.get_var_offset(); }
    int get_num_constraints() const { return registry.get_con_offset(); }

    // Access to constraint and cost lists
    const std::vector<std::shared_ptr<AbstractCost>>& get_costs() const {
        return cost_list;
    }
    const std::vector<std::shared_ptr<AbstractConstraint>>& get_constraints() const {
        return constraint_list;
    }
};

Node::Node(int num_vars)
    : state_dim(0),
      control_dim(0),
      parameter_dim(0),
      registry(num_vars) {}

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
