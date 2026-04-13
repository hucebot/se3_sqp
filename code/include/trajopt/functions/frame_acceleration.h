#pragma once

#include <trajopt/functions/abstract_function.h>
#include <trajopt/node.h>

#include "pinocchio/algorithm/frames.hpp"
#include "pinocchio/algorithm/kinematics.hpp"

#include <trajopt/functions/frame_velocity.h>

/**
 * FrameAcceleration computes the deviation of a frame's acceleration.
 *
 */
class FrameAcceleration : public AbstractFunction {
private:
    Vector6d _a_ref;
    std::string _frame_name;
    pinocchio::ReferenceFrame _re_ref_frame;
    pinocchio::FrameIndex _frame_id;
    pinocchio::Motion _acc;

    Eigen::MatrixXd _dv_dq;
    Eigen::MatrixXd _da_dq;
    Eigen::MatrixXd _da_dv;
    Eigen::MatrixXd _da_da;

public:
    explicit FrameAcceleration(const std::string& frame_name, const Vector6d& a_ref = Vector6d::Zero());

    void allocate_dims() override;
    void evaluate_impl() override;
    void jacobian_impl() override;

    void set_ref(const Vector6d& a_ref) { _a_ref = a_ref; }
    const Vector6d& get_ref() const { return _a_ref; }
    void set_re_reference_frame(const pinocchio::ReferenceFrame& re_ref_rame) { _re_ref_frame = re_ref_rame; }
};
