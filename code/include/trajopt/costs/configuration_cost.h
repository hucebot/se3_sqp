#pragma once

#include <trajopt/costs/abstract_cost.h>
#include <trajopt/node.h>

#include <pinocchio/algorithm/joint-configuration.hpp>

/**
 * ConfigurationCost penalizes deviation from a desired configuration q_ref.
 * Excludes the floating base if present.
 *
 * Residual (dimension nv or nv-fb_nv):
 *   r = q_joints - q_ref_joints  (simple vector difference, no SE3)
 *
 * Jacobian w.r.t. state x = [q; v]:
 *   ∂r/∂q_joints  =  I   (identity)
 *   ∂r/∂v         =  0
 *
 * No control dependency.
 */
class ConfigurationCost : public AbstractCost {
   private:
    VectorXd _q_ref;

    // Floating base detection (computed once in allocate_dims)
    int _fb_nv = 0;  // Number of floating base DOFs in tangent space

   public:
    explicit ConfigurationCost(const VectorXd& q_ref, double weight = 1.0);
    ConfigurationCost(const VectorXd& q_ref, const MatrixXd& weight);

    void allocate_dims() override;

    void evaluate_impl() override;

    void jacobian_impl() override;

    MatrixXdConstRef get_jac_x() const override;
    MatrixXdConstRef get_jac_u() const override;

    void set_q_ref(const VectorXd& q_ref) { _q_ref = q_ref; }
    const VectorXd& get_q_ref() const { return _q_ref; }
};
