#pragma once

#include <Eigen/Dense>
#include <string>

// Forward declarations
class Node;

// Type aliases for Eigen types used in function interfaces
// Column-major is critical: HPIPM expects column-major layout
using ColMajorMatrix =
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;

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
     * @param x Input variables (decision variables)
     * @param output Output buffer (residual/value), size = output_dim
     */
    virtual void evaluate(Eigen::Ref<const Eigen::VectorXd> x,
                          Eigen::Ref<Eigen::VectorXd> output) = 0;

    /**
     * Compute the Jacobian (first derivatives).
     *
     * @param x Input variables
     * @param jac Jacobian matrix (output_dim x input_dim, column-major)
     */
    virtual void jacobian(Eigen::Ref<const Eigen::VectorXd> x,
                          Eigen::Ref<ColMajorMatrix> jac) = 0;

    /**
     * Compute the Hessian (second derivatives) - optional, mainly for costs.
     * Default implementation does nothing (use Gauss-Newton approximation).
     *
     * @param x Input variables
     * @param hess Hessian matrix (input_dim x input_dim, column-major)
     */
    virtual void hessian(Eigen::Ref<const Eigen::VectorXd> x,
                         Eigen::Ref<ColMajorMatrix> hess) {
        // Default: no second-order information (Gauss-Newton approximation)
        // Can be overridden by derived classes for exact Hessian
    }

    // Getters (dimensions determined by derived class implementation)
    virtual int get_input_dim() const { return input_dim; }
    virtual int get_output_dim() const { return output_dim; }
    virtual std::string get_name() const { return name; }
};
