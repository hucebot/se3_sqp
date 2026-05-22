#pragma once

#include <trajopt/functions/abstract_function.h>
#include <trajopt/node.h>

#include "pinocchio/algorithm/frames.hpp"
#include "pinocchio/algorithm/kinematics.hpp"


/**
 * FrameOrientation computes the deviation of a frame's orientation from a target.
 *
 * Residual (dimension 3):
 *   r = log3(R_frame^T * R_ref)
 *
 * where log3 : SO(3) -> R^3 is the rotation vector (axis * angle),
 * expressed in the frame's local body frame.
 *
 * Jacobian w.r.t. state x = [q; v]:
 *   dr/dq = Jlog3(R_frame^T * R_ref) * J_angular_local
 *
 * where J_angular_local is the angular part (bottom 3 rows) of the frame
 * Jacobian in LOCAL convention. No control dependency.
 */
class FrameOrientation : public AbstractFunction {
private:
    std::string _frame_name;
    int _frame_id;
    Eigen::Matrix3d _R_ref;

    // Pre-allocated scratch
    Eigen::Matrix<double, 6, Eigen::Dynamic> _Jframe;   // 6 x nv
    Eigen::Matrix3d _Jlog;     // 3 x 3
    Eigen::Matrix3d _R_err;

public:
    explicit FrameOrientation(const std::string& frame_name, const Eigen::Matrix3d& R_ref = Eigen::Matrix3d::Identity());

    void allocate_dims() override;
    void evaluate_impl() override;
    void jacobian_impl() override;

    void set_ref(const Eigen::Matrix3d& R_ref) { _R_ref = R_ref; }
    const Eigen::Matrix3d& get_ref() const { return _R_ref; }
    int get_frame_id() const { return _frame_id; }
};
