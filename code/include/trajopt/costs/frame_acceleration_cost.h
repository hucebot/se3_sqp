#pragma once

#include <trajopt/costs/abstract_cost.h>
#include <trajopt/functions/frame_acceleration.h>

/**
 * FrameAccelerationCost penalizes deviation of a frame's acceleration from a target.
 *
 * Thin wrapper around FrameTranslation computation.
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
                      const MatrixXd& weight);

    void set_node(Node* node) override { AbstractCost::set_node(node); _fa.set_node(node); }

    void allocate_dims() override;
    void evaluate_impl() override;
    void jacobian_impl() override;

    void set_ref(const Vector6d& a_ref) { _fa.set_ref(a_ref); }
    const Vector6d& get_ref() const { return _fa.get_ref(); }
    void set_re_reference_frame(const pinocchio::ReferenceFrame& re_ref_rame) { _fa.set_re_reference_frame(re_ref_rame); }
    void set_base_frame_name(const std::string& base_frame_name) { _fa.set_base_frame_name(base_frame_name); }
};
