#pragma once

#include <trajopt/costs/abstract_cost.h>
#include <trajopt/functions/frame_acceleration.h>
#include <trajopt/costs/frame_velocity_cost.h>

/**
 * FrameAccelerationCost penalizes deviation of a frame's acceleration from a target.
 *
 * Thin wrapper around FrameAcceleration computation, which evaluates the frame's
 * spatial acceleration relative to a reference target.
 */
class FrameAccelerationCost : public AbstractCost {
private:
    FrameAcceleration _fa;

public:
    explicit FrameAccelerationCost(const std::string& frame_name,
                               const Vector6d& a_ref = Vector6d::Zero(),
                               double weight = 1.0);

    FrameAccelerationCost(const std::string& frame_name,
                      const Vector6d& a_ref,
                      const Matrix6d& weight);

    void set_node(Node* node) override { AbstractCost::set_node(node); _fa.set_node(node); }

    void allocate_dims() override;
    void evaluate_impl() override;
    void jacobian_impl() override;

    void set_ref(const Vector6d& a_ref) { _fa.set_ref(a_ref); }
    const Vector6d& get_ref() const { return _fa.get_ref(); }
    void set_re_reference_frame(const pinocchio::ReferenceFrame& re_ref_rame) { _fa.set_re_reference_frame(re_ref_rame); }
};
