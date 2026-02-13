#pragma once

#include <common/types.h>

#include <Eigen/Dense>
#include <string>

// Forward declaration (don't include node.h - circular dependency)
class Node;
/**
 * AbstractFunction is the base class for all functions in trajectory
 * optimization.
 *
 * This includes costs, constraints, and dynamics. All derived classes must
 * implement:
 * - allocate_slices(): Register variable/data slices in the node
 * - evaluate_impl(): Compute the function value
 * - jacobian_impl(): Compute first derivatives
 *
 * Computation is decoupled from access:
 * - Call evaluate() / jacobian() to trigger computation (stored internally).
 * - Call get_value() / get_jac_x() / get_jac_u() to read stored results.
 *
 * Derived classes:
 * - AbstractCost: Adds weight/scaling for soft constraints
 * - AbstractConstraint: Adds bounds for hard constraints
 */
class AbstractFunction {
   protected:
    int _input_dim;
    int _output_dim;
    std::string _name;

    Node* _node = nullptr;  // Owning node (set by Node::add_cost/add_constraint)

    // Internal storage (pre-allocated in allocate_slices())
    VectorXd _value;    // result of last evaluate()
    MatrixXd _jacobian; // result of last jacobian()

   public:
    AbstractFunction() : _name("") {}
    virtual ~AbstractFunction() = default;

    /**
     * Set the owning node. Called by Node::add_cost/add_constraint.
     * Must be called before allocate_slices().
     */
    void set_node(Node* node) { _node = node; }
    Node* get_node() const { return _node; }

    /**
     * Allocate function-specific slices in the node's registry.
     * Must also resize _value and _jacobian.
     */
    virtual void allocate_slices() = 0;

    /**
     * Trigger computation of the function value.
     * Result is stored in _value, accessible via get_value().
     */
    void evaluate() { evaluate_impl(); }

    /**
     * Trigger computation of the Jacobian.
     * Result is stored in _jacobian, accessible via get_jac_x() / get_jac_u().
     */
    void jacobian() { jacobian_impl(); }

    /**
     * Get the stored function value (result of last evaluate()).
     */
    const VectorXd& get_value() const { return _value; }

    /**
     * Get Jacobian block w.r.t. state x (∂f/∂x).
     * Override in derived classes for specific block extraction.
     */
    virtual MatrixXdConstRef get_jac_x() const {
        return _jacobian;  // Default: return full Jacobian
    }

    /**
     * Get Jacobian block w.r.t. control u (∂f/∂u).
     * Override in derived classes that depend on control.
     */
    virtual MatrixXd get_jac_u() const {
        return MatrixXd::Zero(_output_dim, 0);  // Default: no control dependency
    }

    // Getters
    int get_input_dim() const { return _input_dim; }
    int get_output_dim() const { return _output_dim; }
    virtual std::string get_name() const { return _name; }

   protected:
    virtual void evaluate_impl() = 0;
    virtual void jacobian_impl() = 0;
};
