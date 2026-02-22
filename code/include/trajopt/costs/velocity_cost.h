#pragma once

#include <trajopt/costs/abstract_cost.h>
#include <trajopt/node.h>

/**
 * VelocityCost penalizes deviation from a desired velocity v_ref.
 */
class VelocityCost : public AbstractCost {
   private:
    VectorXd _v_ref;

   public:
    explicit VelocityCost(const VectorXd& v_ref, double weight = 1.0);
    VelocityCost(const VectorXd& v_ref, const MatrixXd& weight);
    explicit VelocityCost(double weight = 1.0);

    void allocate_dims() override;

    void evaluate_impl() override;

    void jacobian_impl() override;

    MatrixXdConstRef get_jac_x() const override;
    MatrixXdConstRef get_jac_u() const override;

    void set_ref(const VectorXd& v_ref) { _v_ref = v_ref; }
    const VectorXd& get_ref() const { return _v_ref; }
};
