#pragma once

#include <common/var_slice.h>
#include <trajopt/constraints/abstract_constraint.h>

#include <pinocchio/algorithm/joint-configuration.hpp>
#include <pinocchio/multibody/model.hpp>

/**
 * EulerIntegration implements explicit Euler as an equality constraint.
 *
 * For state x = [q; v] and control u = a (acceleration):
 *   q_{k+1} = q_k ⊕ (dt * v_k)   (Pinocchio integrate for Lie groups)
 *   v_{k+1} = v_k + dt * a_k
 *
 * Constraint residual (equality to zero):
 *   g = [ q_{k+1} ⊖ integrate(q_k, dt * v_k) ]
 *       [ v_{k+1} - v_k - dt * a_k           ]
 */
class EulerIntegration : public AbstractConstraint {
   private:
    double _dt;

    // Pre-allocated temporaries
    VectorXd _q;
    VectorXd _q_next;
    VectorXd _vq;
    VectorXd _vq_next;
    VectorXd _aq;
    VectorXd _q_integrated;
    VectorXd _res;
    MatrixXd _J_q;
    MatrixXd _J_v;


   public:
    explicit EulerIntegration(double dt);

    ~EulerIntegration() = default;

    void allocate_slices() override;

    void evaluate(VectorXdRef output) override;

    void jacobian(MatrixXdRef jac) override;
    void jacobian() override;

    // Accessors for SQP solver
    MatrixXdConstRef get_jac_x() const override;
    MatrixXd get_jac_u() const override;
};
