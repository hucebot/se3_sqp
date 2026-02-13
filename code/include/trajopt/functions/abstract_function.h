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
 * - evaluate(): Compute the function value
 * - jacobian(): Compute first derivatives
 * - hessian(): (Optional) Compute second derivatives
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

    // Internal Jacobian storage (pre-allocated, size: _output_dim × _input_dim)
    MatrixXd _jacobian;

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
     * This is called during problem setup to register the function's
     * variable and data requirements.
     */
    virtual void allocate_slices() = 0;

    /**
     * Evaluate the function value (public, non-virtual).
     * Logs the call in DEBUG mode, then delegates to evaluate_impl().
     */
    void evaluate(VectorXdRef output) {
        DEBUG_PRINT(_name << " - evaluate");
        evaluate_impl(output);
    }

    /**
     * Compute the Jacobian - legacy interface (public, non-virtual).
     * Logs the call in DEBUG mode, then delegates to jacobian_impl().
     */
    void jacobian(MatrixXdRef jac) {
        DEBUG_PRINT(_name << " - jacobian");
        jacobian_impl(jac);
    }

    /**
     * Compute the Jacobian - new interface (public, non-virtual).
     * Stores result in internal _jacobian matrix.
     */
    void jacobian() {
        DEBUG_PRINT(_name << " - jacobian");
        jacobian_impl(_jacobian);
    }

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

    // /**
    //  * Compute the Hessian (second derivatives) - optional, mainly for costs.
    //  * Default implementation does nothing (use Gauss-Newton approximation).
    //  *
    //  * @param x Input variables
    //  * @param hess Hessian matrix (input_dim x input_dim, column-major)
    //  */
    // virtual void hessian(VectorXdConstRef x, VectorXdConstRef u,
    //                      MatrixXdConstRef hes_xx, MatrixXdConstRef hes_uu) {
    //     // Default: no second-order information (Gauss-Newton approximation)
    //     // Can be overridden by derived classes for exact Hessian
    // }

    // Getters (dimensions determined by derived class implementation)
    int get_input_dim() const { return _input_dim; }
    int get_output_dim() const { return _output_dim; }
    virtual std::string get_name() const { return _name; }

   protected:
    /**
     * Implement the function evaluation.
     * @param output Output buffer (residual/value), size = output_dim
     */
    virtual void evaluate_impl(VectorXdRef output) = 0;

    /**
     * Implement the Jacobian computation.
     * @param jac Output Jacobian matrix
     */
    virtual void jacobian_impl(MatrixXdRef jac) = 0;
};
