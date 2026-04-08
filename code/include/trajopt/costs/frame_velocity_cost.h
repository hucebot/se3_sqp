#pragma once

#include <trajopt/costs/abstract_cost.h>
#include <trajopt/functions/frame_velocity.h>

/**
 * FrameVelocityCost penalizes deviation of a frame's velocity/twist from a target.
 *
 * Thin wrapper around FrameVelocity computation.
 */
class FrameVelocityCost : public AbstractCost {
private:
    FrameVelocity _fv;

public:
    explicit FrameVelocityCost(const std::string& frame_name,
                                  const Vector6d& v_ref = Vector6d::Zero(),
                                  double weight = 1.0);

    FrameVelocityCost(const std::string& frame_name,
                         const Vector6d& v_ref,
                         const MatrixXd& weight);

    void set_node(Node* node) override { AbstractCost::set_node(node); _fv.set_node(node); }

    void allocate_dims() override;
    void evaluate_impl() override;
    void jacobian_impl() override;

    void set_ref(const Vector6d& v_ref) { _fv.set_ref(v_ref); }
    const Vector6d& get_ref() const { return _fv.get_ref(); }
    void set_re_reference_frame(const pinocchio::ReferenceFrame& re_ref_rame) { _fv.set_re_reference_frame(re_ref_rame); }
    void set_base_frame_name(const std::string& base_frame_name) { _fv.set_base_frame_name(base_frame_name); }
};
