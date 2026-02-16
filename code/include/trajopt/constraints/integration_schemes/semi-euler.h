#pragma once

#include <trajopt/constraints/abstract_constraint.h>

#include <pinocchio/algorithm/joint-configuration.hpp>
#include <pinocchio/multibody/model.hpp>

/**
 * SemiEulerIntegration implements explicit Euler as an equality constraint.
 *
 * For state x = [q; v] and control u = a (acceleration):
 *   v_{k+1} = v_k + dt * a_k
 *   q_{k+1} = q_k ⊕ (dt * v_{k+1})   (Pinocchio integrate for Lie groups)
 *
 * Constraint residual (equality to zero):
 *   g = [ q_{k+1} ⊖ integrate(q_k, dt * v_k) ]
 *       [ v_{k+1} - v_k - dt * a_k           ]
 */
class SemiEulerIntegration : public AbstractConstraint {
   private:
    double _dt;

    // Pre-allocated temporaries
    VectorXd _q;
    VectorXd _q_next;
    VectorXd _vq;
    VectorXd _vq_next;
    VectorXd _aq;
    VectorXd _q_integrated;
    MatrixXd _J_q;
    MatrixXd _J_v;
    MatrixXd _J_diff_qnext;


   public:
    explicit SemiEulerIntegration(double dt);

    ~SemiEulerIntegration() = default;

    void allocate_slices() override;

    void evaluate_impl() override;

    void jacobian_impl() override;

    // Accessors for SQP solver
    MatrixXdConstRef get_jac_x() const override;
    MatrixXd get_jac_u() const override;
};
