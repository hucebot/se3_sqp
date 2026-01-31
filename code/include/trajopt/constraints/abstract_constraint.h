#pragma once

#include <functions/abstract_function.h>
#include <Eigen/Dense>

/**
 * AbstractConstraint extends AbstractFunction with constraint-specific semantics.
 *
 * Constraints represent hard bounds that must be satisfied.
 * They can be equality constraints (lower == upper) or inequality constraints.
 *
 * Derived classes must implement:
 * - allocate_slices(): Register variable slices
 * - evaluate(): Compute constraint residual
 * - jacobian(): Compute constraint Jacobian
 *
 * Constraint form: lower <= g(x) <= upper
 */
class AbstractConstraint : public AbstractFunction {
   protected:
    Eigen::VectorXd lower_bound;  // Lower bounds on constraint
    Eigen::VectorXd upper_bound;  // Upper bounds on constraint

   public:
    AbstractConstraint() : AbstractFunction() {}
    virtual ~AbstractConstraint() {}

    /**
     * Set bounds for the constraint.
     *
     * @param lower Lower bounds (size must match output_dim)
     * @param upper Upper bounds (size must match output_dim)
     */
    void set_bounds(const Eigen::VectorXd& lower, const Eigen::VectorXd& upper) {
        lower_bound = lower;
        upper_bound = upper;
    }

    /**
     * Set as equality constraint: g(x) = value
     */
    void set_equality(const Eigen::VectorXd& value) {
        lower_bound = value;
        upper_bound = value;
    }

    /**
     * Set as equality constraint to zero: g(x) = 0
     */
    void set_equality_to_zero() {
        lower_bound = Eigen::VectorXd::Zero(output_dim);
        upper_bound = Eigen::VectorXd::Zero(output_dim);
    }

    /**
     * Get the lower bounds
     */
    const Eigen::VectorXd& get_lower_bound() const { return lower_bound; }

    /**
     * Get the upper bounds
     */
    const Eigen::VectorXd& get_upper_bound() const { return upper_bound; }

    /**
     * Check if this is an equality constraint
     */
    bool is_equality() const {
        if (lower_bound.size() != upper_bound.size()) return false;
        return lower_bound.isApprox(upper_bound);
    }
};
