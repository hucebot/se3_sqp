#pragma once

#include <trajopt/constraints/abstract_constraint.h>
#include <trajopt/node.h>

#include <pinocchio/algorithm/joint-configuration.hpp>
#include <pinocchio/multibody/model.hpp>

/**
 * JointLimitsConstraint enforces both position and velocity limits from the model.
 *
 * For a robot model with nv degrees of freedom, this constraint enforces:
 *   q_lower <= q <= q_upper  (position limits)
 *   -v_max <= v <= v_max     (velocity limits)
 *
 * where the limits are taken from the pinocchio model.
 *
 * The constraint function is expressed in tangent space (nv + nv = 2*nv dimensions):
 *   g(q, v) = [q; v]  (both in tangent coordinates)
 *
 * For position, we project q onto the tangent space at q_ref using difference operation.
 * For velocity, it's already in tangent space.
 *
 * Bounds are handled by projecting the limit configurations onto tangent space.
 *
 * Jacobian:
 *   ∂g/∂q = [I; 0]  (in tangent coordinates)
 *   ∂g/∂v = [0; I]
 *   ∂g/∂u = 0
 *
 * Note: Output dimension is 2*nv (tangent space), even for manifolds where nq != nv.
 */
class JointLimitsConstraint : public AbstractConstraint {
   public:
    /**
     * Constructor.
     * Position and velocity limits are automatically extracted from the model.
     */
    explicit JointLimitsConstraint();

    void allocate_dims() override;
    void evaluate_impl() override;
    void jacobian_impl() override;

    MatrixXdConstRef get_jac_x() const override;
    MatrixXd get_jac_u() const override;
};
