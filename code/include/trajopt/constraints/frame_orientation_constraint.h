#pragma once

#include <trajopt/constraints/abstract_constraint.h>
#include <trajopt/functions/frame_orientation.h>

/**
 * FrameOrientationConstraint constrains a frame's 3D orientation.
 *
 * Thin wrapper around FrameOrientation computation.
 * Bounds must be set after construction (default: equality to zero).
 */
class FrameOrientationConstraint : public AbstractConstraint {
private:
    FrameOrientation _fo;

public:
    explicit FrameOrientationConstraint(const std::string& frame_name,
                                        const Eigen::Matrix3d& R_ref = Eigen::Matrix3d::Identity());

    void set_node(Node* node) override { AbstractConstraint::set_node(node); _fo.set_node(node); }

    void allocate_dims() override;
    void evaluate_impl() override;
    void jacobian_impl() override;

    void set_ref(const Eigen::Matrix3d& p_ref) { _fo.set_ref(p_ref); }
    const Eigen::Matrix3d& get_ref() const { return _fo.get_ref(); }
};
