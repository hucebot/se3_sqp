#pragma once

#include <trajopt/costs/abstract_cost.h>
#include <trajopt/node.h>

#include <pinocchio/algorithm/joint-configuration.hpp>

/**
 * ConfigurationCost penalizes deviation from a desired configuration q_ref.
 *
 * Residual (in tangent space, dimension nv):
 *   r = q ⊖ q_ref  =  difference(q_ref, q)
 *
 * Jacobian w.r.t. state x = [q; v]:
 *   ∂r/∂q  =  dDifference(q_ref, q, ARG1)   (nv × nv)
 *   ∂r/∂v  =  0                              (nv × nv)
 *
 * No control dependency.
 */
class ConfigurationCost : public AbstractCost {
   private:
    VectorXd _q_ref;

    // Pre-allocated Jacobian block
    MatrixXd _J_dq;

   public:
    explicit ConfigurationCost(const VectorXd& q_ref);

    void allocate_slices() override;

    void evaluate_impl() override;

    void jacobian_impl() override;

    MatrixXdConstRef get_jac_x() const override;
    MatrixXd get_jac_u() const override;

    void set_q_ref(const VectorXd& q_ref) { _q_ref = q_ref; }
    const VectorXd& get_q_ref() const { return _q_ref; }
};
