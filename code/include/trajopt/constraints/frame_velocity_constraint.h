#pragma once

#include <pinocchio/multibody/fwd.hpp>
#include <trajopt/constraints/abstract_constraint.h>

typedef Eigen::Matrix<double, 6, 1> Vector6d;

/**
 * FrameVelocityConstraint constrains a frame's twist.
 *
 */


class FrameVelocityConstraint : public AbstractConstraint {
private:
    Vector6d _v_ref;
    std::string _frame_name;
    pinocchio::ReferenceFrame _re_ref_frame;
    std::string _base_frame_name;
    pinocchio::FrameIndex _frame_id;

    Eigen::MatrixXd _dv_dq;
    Eigen::MatrixXd _dv_dqdot;

public:

    explicit FrameVelocityConstraint(const std::string& frame_name,
                                        const Vector6d& v_ref = Vector6d::Zero());


    void allocate_dims() override;
    void evaluate_impl() override;
    void jacobian_impl() override;

    void set_ref(const Vector6d& v_ref) { _v_ref = v_ref; }
    const Vector6d& get_ref() const { return _v_ref; }
    void set_re_reference_frame(const pinocchio::ReferenceFrame& re_ref_rame) { _re_ref_frame = re_ref_rame; }
    void set_base_frame_name(const std::string& base_frame_name) { _base_frame_name = base_frame_name; }
};
