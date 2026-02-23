#pragma once

#include <trajopt/functions/abstract_function.h>
#include <trajopt/node.h>

#include "pinocchio/algorithm/frames.hpp"
#include "pinocchio/algorithm/kinematics.hpp"

/**
 * FrameTranslation computes the 3D position of a frame relative to a target.
 *
 * Residual (dimension 3):
 *   r = p_frame(q) - p_ref
 *
 * Jacobian w.r.t. state x = [q; v]:
 *   ∂r/∂q = J_frame_linear  (3 × nv, linear part of frame Jacobian in LOCAL_WORLD_ALIGNED)
 *   ∂r/∂v = 0                (3 × nv)
 *
 * No control dependency.
 *
 * This class inherits from AbstractFunction and can be used directly
 * or wrapped by FrameTranslationCost / FrameTranslationConstraint.
 */
class FrameTranslation : public AbstractFunction {
   private:
    std::string _frame_name;
    int _frame_id;
    Eigen::Vector3d _p_ref;

    // Pre-allocated scratch
    MatrixXd _Jframe;  // 6 x nv

   public:
    explicit FrameTranslation(const std::string& frame_name, const Eigen::Vector3d& p_ref = Eigen::Vector3d::Zero());

    void allocate_dims() override;
    void evaluate_impl() override;
    void jacobian_impl() override;

    void set_ref(const Eigen::Vector3d& p_ref) { _p_ref = p_ref; }
    const Eigen::Vector3d& get_ref() const { return _p_ref; }
    int get_frame_id() const { return _frame_id; }
};
