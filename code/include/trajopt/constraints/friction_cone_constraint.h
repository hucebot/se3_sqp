#pragma once

#include <trajopt/constraints/abstract_constraint.h>
#include <trajopt/node.h>

#include "pinocchio/algorithm/frames.hpp"
#include "pinocchio/algorithm/kinematics.hpp"

/**
 * FrictionConeConstraint enforces friction cone constraints on contact forces.
 *
 * The contact force f_local (expressed in the contact frame) is transformed to
 * world frame: f_world = R_world_frame * f_local, where R_world_frame is the
 * rotation matrix from the contact frame to world frame.
 *
 * For a contact frame with force f_world = [fx, fy, fz] (in world frame):
 *   - Normal force constraint: fz >= 0 (no pulling)
 *   - Friction cone (linearized pyramid): |fx| <= μ * fz, |fy| <= μ * fz
 *
 * The constraint is only active when the contact is active in the node.
 * When inactive, the constraint is relaxed (unbounded).
 *
 * Linearized pyramid approximation of the friction cone:
 *   g(q, f) = [ f_world_z;                    // >= 0
 *               μ*f_world_z - f_world_x;      // >= 0
 *               μ*f_world_z + f_world_x;      // >= 0
 *               μ*f_world_z - f_world_y;      // >= 0
 *               μ*f_world_z + f_world_y ]     // >= 0
 *
 * This gives a 5-dimensional constraint vector.
 *
 * Jacobian:
 *   ∂g/∂q: depends on rotation matrix derivative (from frame orientation)
 *   ∂g/∂f: depends on rotation matrix (linear transformation)
 *
 * Reference: OpenSoT FrictionCone constraint implementation
 */
class FrictionConeConstraint : public AbstractConstraint {
   private:
    std::string _frame_name;
    int _frame_id;
    int _contact_idx;  // Index in node's contact list
    double _mu;        // Friction coefficient

    // Pre-allocated scratch
    MatrixXd _Jframe;       // 6 x nv (frame Jacobian in LOCAL frame)
    Matrix3d _R_world;      // Rotation matrix from frame to world
    Matrix3d _Adj_rot;      // Adjoint for rotation (transposes for force transformation)
    Vector3d _f_local;      // Force in local frame
    Vector3d _f_world;      // Force in world frame

   public:
    /**
     * Constructor.
     * @param frame_name Name of the contact frame (must be registered in node's contact list)
     * @param mu Friction coefficient (default: 0.5)
     */
    explicit FrictionConeConstraint(const std::string& frame_name, double mu = 0.5);

    void allocate_dims() override;
    void evaluate_impl() override;
    void jacobian_impl() override;

    MatrixXdConstRef get_jac_x() const override;
    MatrixXdConstRef get_jac_u() const override;

    /**
     * Set friction coefficient.
     */
    void set_friction_coefficient(double mu) { _mu = mu; }

    double get_friction_coefficient() const { return _mu; }
    int get_contact_idx() const { return _contact_idx; }
    int get_frame_id() const { return _frame_id; }
};
