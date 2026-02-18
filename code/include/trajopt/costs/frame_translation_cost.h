#pragma once

#include <trajopt/costs/abstract_cost.h>
#include <trajopt/functions/frame_translation.h>

/**
 * FrameTranslationCost penalizes deviation of a frame's position from a target.
 *
 * Thin wrapper around FrameTranslation computation.
 */
class FrameTranslationCost : public AbstractCost {
   private:
    FrameTranslation _ft;

   public:
    explicit FrameTranslationCost(const std::string& frame_name,
                                  const Eigen::Vector3d& p_ref = Eigen::Vector3d::Zero(),
                                  double weight = 1.0);

    FrameTranslationCost(const std::string& frame_name,
                         const Eigen::Vector3d& p_ref,
                         const MatrixXd& weight);

    void set_node(Node* node) override { AbstractCost::set_node(node); _ft.set_node(node); }

    void allocate_dims() override;
    void evaluate_impl() override;
    void jacobian_impl() override;

    MatrixXdConstRef get_jac_x() const override;
    MatrixXd get_jac_u() const override;

    void set_ref(const Eigen::Vector3d& p_ref) { _ft.set_ref(p_ref); }
    const Eigen::Vector3d& get_ref() const { return _ft.get_ref(); }
};
