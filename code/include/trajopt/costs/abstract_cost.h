#pragma once

#include <trajopt/functions/abstract_function.h>

/**
 * AbstractCost extends AbstractFunction with cost-specific semantics.
 *
 * Costs represent soft constraints with weights/scaling factors.
 * They contribute to the objective function being minimized.
 *
 * Derived classes must implement:
 * - allocate_slices(): Register variable slices
 * - evaluate(): Compute cost value
 * - jacobian(): Compute gradient
 * - hessian(): (Optional) Compute Hessian for better convergence
 */
class AbstractCost : public AbstractFunction {
   protected:
    double weight;  // Cost weight/scaling factor

   public:
    AbstractCost() : AbstractFunction(), weight(1.0) {}
    virtual ~AbstractCost() {}

    // TODO weight should be a matrix 

    /**
     * Get the weight/scaling factor for this cost
     */
    double get_weight() const { return weight; }

    /**
     * Set the weight/scaling factor for this cost
     */
    void set_weight(double w) { weight = w; }
};