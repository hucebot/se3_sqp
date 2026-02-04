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
     * Evaluate the function value.
     *
     * @param var Input variables (decision variables)
     * @param output Output buffer (residual/value), size = output_dim
     */
    virtual void evaluate(VectorXdRef output) = 0;

    /**
     * Compute the Jacobian (first derivatives).
     */
    virtual void jacobian(MatrixXdRef jac) = 0;

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
    virtual std::string get_name() const { return _name; }
};
