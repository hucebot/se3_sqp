#pragma once

#include <trajopt/costs/abstract_cost.h>
#include <trajopt/node.h>

/**
 * VelocityCost penalizes deviation from a desired velocity v_ref.
 * Excludes the floating base if present.
 */
class VelocityCost : public AbstractCost {
   private:
    VectorXd _v_ref;

    // Floating base detection (computed once in allocate_dims)
    int _fb_nv = 0;  // Number of floating base DOFs in tangent space

   public:
    explicit VelocityCost(const VectorXd& v_ref, double weight = 1.0);
    VelocityCost(const VectorXd& v_ref, const MatrixXd& weight);
    explicit VelocityCost(double weight = 1.0);

    void allocate_dims() override;

    void evaluate_impl() override;

    void jacobian_impl() override;

    void set_ref(const VectorXd& v_ref) { _v_ref = v_ref; }
    const VectorXd& get_ref() const { return _v_ref; }
};
