#pragma once

#include <common/types.h>

#include <Eigen/Dense>
#include <string>

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
    virtual void set_node(Node* node) { _node = node; }
    Node* get_node() const { return _node; }

    /**
     * Allocate function-specific slices in the node's registry.
     * Calls allocate_dims() to set dimensions and custom members,
     * then resizes _value and _jacobian, then calls post_allocate().
     */
    void allocate_slices() {
        allocate_dims();
        _value.resize(_output_dim);
        _jacobian.resize(_output_dim, _input_dim);
        _jacobian.setZero();
        post_allocate();
    }

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
     */
    virtual MatrixXdConstRef get_jac_x() const;

    /**
     * Get Jacobian block w.r.t. control u (∂f/∂u).
     */
    virtual MatrixXdConstRef get_jac_u() const;

    // Getters
    int get_input_dim() const { return _input_dim; }
    int get_output_dim() const { return _output_dim; }
    virtual std::string get_name() const { return _name; }

   protected:
    /** Set _output_dim, _input_dim, _ndx, and allocate custom members. */
    virtual void allocate_dims() = 0;

    /** Hook called after _value/_jacobian are resized. Default: no-op. */
    virtual void post_allocate() {}

    virtual void evaluate_impl() = 0;
    virtual void jacobian_impl() = 0;
};
