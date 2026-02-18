#pragma once

#include <trajopt/costs/abstract_cost.h>
#include <trajopt/node.h>

#include <pinocchio/algorithm/joint-configuration.hpp>

/**
 * AccelerationCost penalizes deviation from a desired acceleration a_ref.
 */
class AccelerationCost : public AbstractCost {
   private:
    VectorXd _a_ref;

   public:
    explicit AccelerationCost(const VectorXd& a_ref, double weight = 1.0);
    AccelerationCost(const VectorXd& a_ref, const MatrixXd& weight);
    explicit AccelerationCost(double weight = 1.0);

    void allocate_dims() override;

    void evaluate_impl() override;

    void jacobian_impl() override;

    MatrixXdConstRef get_jac_x() const override;
    MatrixXd get_jac_u() const override;

    void set_ref(const VectorXd& a_ref) { _a_ref = a_ref; }
    const VectorXd& get_ref() const { return _a_ref; }
};
