#pragma once

#include <trajopt/functions/abstract_function.h>
#include <trajopt/node.h>

#include "pinocchio/algorithm/frames.hpp"
#include "pinocchio/algorithm/kinematics.hpp"

typedef Eigen::Matrix<double, 6, 1> Vector6d;

/**
 * FrameVelocity computes the deviation of a frame's velocity.
 *
 */
class FrameVelocity : public AbstractFunction {
private:
    Vector6d _v_ref;
    std::string _frame_name;
    pinocchio::ReferenceFrame _re_ref_frame;
    std::string _base_frame_name;
    pinocchio::FrameIndex _frame_id;

    Eigen::MatrixXd _dv_dq;
    Eigen::MatrixXd _dv_dqdot;     // 3 x 3

public:
    explicit FrameVelocity(const std::string& frame_name, const Vector6d& v_ref = Vector6d::Zero());

    void allocate_dims() override;
    void evaluate_impl() override;
    void jacobian_impl() override;

    void set_ref(const Vector6d& v_ref) { _v_ref = v_ref; }
    const Vector6d& get_ref() const { return _v_ref; }
    void set_re_reference_frame(const pinocchio::ReferenceFrame& re_ref_rame) { _re_ref_frame = re_ref_rame; }
    void set_base_frame_name(const std::string& base_frame_name) { _base_frame_name = base_frame_name; }
};
