#pragma once

#include <trajopt/constraints/abstract_constraint.h>
#include <trajopt/functions/frame_translation.h>

/**
 * FrameTranslationConstraint constrains a frame's 3D position.
 *
 * Thin wrapper around FrameTranslation computation.
 * Bounds must be set after construction (default: equality to zero).
 */
class FrameTranslationConstraint : public AbstractConstraint {
   private:
    FrameTranslation _ft;

   public:
    explicit FrameTranslationConstraint(const std::string& frame_name,
                                        const Eigen::Vector3d& p_ref = Eigen::Vector3d::Zero());

    void set_node(Node* node) override { AbstractConstraint::set_node(node); _ft.set_node(node); }

    void allocate_dims() override;
    void evaluate_impl() override;
    void jacobian_impl() override;

    MatrixXdConstRef get_jac_x() const override;
    MatrixXdConstRef get_jac_u() const override;

    void set_ref(const Eigen::Vector3d& p_ref) { _ft.set_ref(p_ref); }
    const Eigen::Vector3d& get_ref() const { return _ft.get_ref(); }
};
