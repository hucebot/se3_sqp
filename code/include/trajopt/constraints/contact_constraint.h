#pragma once

#include <trajopt/constraints/abstract_constraint.h>
#include <trajopt/node.h>

#include "pinocchio/algorithm/frames.hpp"
#include "pinocchio/algorithm/kinematics.hpp"

/**
 * ContactConstraint enforces contact conditions for a given frame.
 *
 * The contact state (active/inactive) is read from the Node's contact list.
 *
 * When in contact (active in node):
 *   - Position: z = ground_height (frame z-coordinate at ground level)
 *   - Velocity: v_frame = 0 (no sliding, 3D linear velocity is zero)
 *
 * When not in contact (inactive in node):
 *   - Position: z >= ground_height (frame stays above ground)
 *   - Velocity: unconstrained
 *
 * The constraint has different dimensions based on contact state:
 *   - Active (in contact): 4 constraints (z=height, vx=0, vy=0, vz=0)
 *   - Inactive (not in contact): 1 constraint (z>=height)
 *
 * Residual form:
 *   Active: g(x) = [p_z - ground_height; v_frame] = 0
 *   Inactive: g(x) = p_z - ground_height >= 0
 *
 * Jacobian w.r.t. state x = [q; v]:
 *   ∂g/∂q for position: from frame Jacobian (position part)
 *   ∂g/∂v for velocity: from frame Jacobian (velocity part)
 *
 * No control dependency.
 */
class ContactConstraint : public AbstractConstraint {
   private:
    std::string _frame_name;
    int _frame_id;
    int _contact_idx;  // Index in node's contact list
    double _ground_height;

    // Pre-allocated scratch
    MatrixXd _Jframe;       // 6 x nv (frame Jacobian in world frame)
    MatrixXd _Jvf_dq;
    MatrixXd _Jvf_dv;


   public:
    /**
     * Constructor.
     * @param frame_name Name of the contact frame (must be registered in node's contact list)
     * @param ground_height Ground level z-coordinate (default: 0.0)
     */
    explicit ContactConstraint(const std::string& frame_name,
                              double ground_height = 0.0);

    void allocate_dims() override;
    void evaluate_impl() override;
    void jacobian_impl() override;

    MatrixXdConstRef get_jac_x() const override;
    MatrixXd get_jac_u() const override;

    /**
     * Set ground height.
     */
    void set_ground_height(double height) { _ground_height = height; }

    double get_ground_height() const { return _ground_height; }
    int get_frame_id() const { return _frame_id; }
    int get_contact_idx() const { return _contact_idx; }
};
