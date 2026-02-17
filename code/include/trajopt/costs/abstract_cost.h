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
    MatrixXd _weight;  // Cost weight/scaling factor

   public:
    AbstractCost() : AbstractFunction() {}
    virtual ~AbstractCost() {}

    // TODO weight should be a matrix 

    /**
     * Get the weight/scaling factor for this cost
     */
    MatrixXd get_weight() const { return _weight; }

    /**
     * Set the weight/scaling factor for this cost
     */
    void set_weight(double w) { _weight = Eigen::MatrixXd(_output_dim, _output_dim).setIdentity() * w;}

    void set_weight(const MatrixXd& w) {
        if (w.rows() != _output_dim || w.cols() != _output_dim) {
            throw std::invalid_argument(
                _name + ": Weight matrix must be " +
                std::to_string(_output_dim) + "x" + std::to_string(_output_dim) +
                ", but got " + std::to_string(w.rows()) + "x" + std::to_string(w.cols())
            );
        }
        _weight = w;
    }
};