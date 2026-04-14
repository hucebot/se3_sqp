#pragma once

#include <trajopt/costs/abstract_cost.h>
#include <trajopt/functions/inverse_dynamics_function.h>

#include <pinocchio/algorithm/joint-configuration.hpp>


/**
 * TorqueCost penalizes deviation from a desired torque tau_ref.
 * Excludes the floating base if present.
 */
class TorqueCost : public AbstractCost {
private:
    VectorXd _tau_ref;
    InvDynamicsFunction _id;

    // Floating base detection (computed once in allocate_dims)
    int _fb_nv = 0;  // Number of floating base DOFs in tangent space

public:
    explicit TorqueCost(const VectorXd& tau_ref, double weight = 1.0);
    TorqueCost(const VectorXd& a_ref, const MatrixXd& weight);
    explicit TorqueCost(double weight = 1.0);

    void set_node(Node* node) override { AbstractCost::set_node(node); _id.set_node(node); }

    void allocate_dims() override;

    void evaluate_impl() override;

    void jacobian_impl() override;

    void set_ref(const VectorXd& tau_ref) { _tau_ref = tau_ref; }
    const VectorXd& get_ref() const { return _tau_ref; }
};
