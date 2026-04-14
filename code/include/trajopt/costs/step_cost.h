#pragma once

#include "trajopt/costs/frame_translation_cost.h"
#include <trajopt/costs/abstract_cost.h>


class StepCost : public AbstractCost {
private:
    FrameTranslationCost _ftc;
    double _step_height_ref;
    double _ground_ref;
    std::string _frame_name;
    int _contact_idx;  // Index in node's contact list
    Eigen::Vector3d _p_ref;

public:
    StepCost(const std::string& frame_name,
             double step_height_ref,
             double ground_ref = 0.,
             double weight = 1.0);

    StepCost(const std::string& frame_name,
             double step_height_ref,
             double ground_ref,
             const Matrix3d& weight);

    void set_node(Node* node) override { AbstractCost::set_node(node); _ftc.set_node(node); }

    void allocate_dims() override;
    void evaluate_impl() override;
    void jacobian_impl() override;

    void set_ref(const double& step_height_ref) { _step_height_ref = step_height_ref; }
    double get_ref() const { return _step_height_ref; }
    void set_ground_ref(const double& ground_ref) { _ground_ref = ground_ref; }
    double get_ground_ref() const { return _ground_ref; }
};
