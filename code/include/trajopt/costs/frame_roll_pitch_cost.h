#pragma once

#include <trajopt/costs/frame_orientation_cost.h>

/**
 * @brief The FrameRollPitchCost class is simlar to FrameOrientationCost but YAW IS NOT regulated
 */
class FrameRollPitchCost : public AbstractCost {
    private:
        FrameOrientationCost _foc;

    public:
        explicit FrameRollPitchCost(const std::string& frame_name,
                                      const Eigen::Matrix3d& R_ref = Eigen::Matrix3d::Identity(),
                                      double weight = 1.0);

        FrameRollPitchCost(const std::string& frame_name,
                             const Eigen::Matrix3d& R_ref,
                             const Matrix3d& weight);

        void set_node(Node* node) override { AbstractCost::set_node(node); _foc.set_node(node); }

        void allocate_dims() override;
        void evaluate_impl() override;
        void jacobian_impl() override;

        void set_ref(const Eigen::Matrix3d& R_ref) { _foc.set_ref(R_ref); }
        const Eigen::Matrix3d& get_ref() const { return _foc.get_ref(); }
};
