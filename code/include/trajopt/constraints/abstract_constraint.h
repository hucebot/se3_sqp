#pragma once

#include <trajopt/functions/abstract_function.h>

#include <Eigen/Dense>

/**
 * AbstractConstraint extends AbstractFunction with constraint-specific
 * semantics.
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
    VectorXd _lower_bound;  // Lower bounds on constraint
    VectorXd _upper_bound;  // Upper bounds on constraint

   public:
    AbstractConstraint() : AbstractFunction() {}
    virtual ~AbstractConstraint() {}

    /**
     * Set bounds for the constraint.
     *
     * @param lower Lower bounds (size must match output_dim)
     * @param upper Upper bounds (size must match output_dim)
     */
    void set_bounds(VectorXdConstRef lower, VectorXdConstRef upper) {
        _lower_bound = lower;
        _upper_bound = upper;
    }

    /**
     * Set as equality constraint: g(x) = value
     */
    void set_equality(VectorXdConstRef value) {
        _lower_bound = value;
        _upper_bound = value;
    }

    /**
     * Set as equality constraint to zero: g(x) = 0
     */
    void set_equality_to_zero() {
        _lower_bound = VectorXd::Zero(_output_dim);
        _upper_bound = VectorXd::Zero(_output_dim);
    }

    const VectorXd& get_lower_bound() const { return _lower_bound; }

    const VectorXd& get_upper_bound() const { return _upper_bound; }

    bool is_equality() const {
        if (_lower_bound.size() != _upper_bound.size()) return false;
        return _lower_bound.isApprox(_upper_bound);
    }
};
