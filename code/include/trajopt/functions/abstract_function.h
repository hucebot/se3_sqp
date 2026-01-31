#pragma once

#include <string>

// Forward declarations
class Node;

/**
 * AbstractFunction is the base class for all functions in trajectory optimization.
 *
 * This includes costs, constraints, and dynamics. All derived classes must implement:
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
    int input_dim;   // Dimension of input variables (set by derived class)
    int output_dim;  // Dimension of output/residual (set by derived class)
    std::string name;

   public:
    AbstractFunction() : input_dim(0), output_dim(0), name("") {}
    virtual ~AbstractFunction() {}

    /**
     * Allocate function-specific slices in the node's registry.
     * This is called during problem setup to register the function's
     * variable and data requirements.
     *
     * @param node Reference to the node where slices will be allocated
     */
    virtual void allocate_slices(Node& node) = 0;

    /**
     * Evaluate the function value.
     *
     * @param x Pointer to input variables (decision variables)
     * @param output Pointer to output buffer (residual/value)
     */
    virtual void evaluate(const double* x, double* output) = 0;

    /**
     * Compute the Jacobian (first derivatives).
     *
     * @param x Pointer to input variables
     * @param jac Pointer to Jacobian matrix (output_dim x input_dim, row-major)
     */
    virtual void jacobian(const double* x, double* jac) = 0;

    /**
     * Compute the Hessian (second derivatives) - optional, mainly for costs.
     * Default implementation does nothing (use Gauss-Newton approximation).
     *
     * @param x Pointer to input variables
     * @param hess Pointer to Hessian matrix (input_dim x input_dim, row-major)
     */
    virtual void hessian(const double* x, double* hess) {
        // Default: no second-order information (Gauss-Newton approximation)
        // Can be overridden by derived classes for exact Hessian
    }

    // Getters (dimensions determined by derived class implementation)
    virtual int get_input_dim() const { return input_dim; }
    virtual int get_output_dim() const { return output_dim; }
    virtual std::string get_name() const { return name; }
};
