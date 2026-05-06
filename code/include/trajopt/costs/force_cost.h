#pragma once

#include <trajopt/costs/abstract_cost.h>
#include <trajopt/node.h>


/**
 * ForceCost penalizes deviation from a desired force f_ref.
 */
class ForceCost : public AbstractCost {
private:
    Vector3d _f_ref;

    std::string _frame_name;
    int _contact_idx;

    Vector3d _f_local;

public:
    explicit ForceCost(const std::string& frame_name, const Vector3d& f_ref = Vector3d(Vector3d::Zero()), double weight = 1.0);
    ForceCost(const std::string& frame_name, const Vector3d& f_ref, const Matrix3d& weight);

    void allocate_dims() override;

    void evaluate_impl() override;

    void jacobian_impl() override;

    void set_ref(const Vector3d& f_ref) { _f_ref = f_ref; }
    const Vector3d& get_ref() const { return _f_ref; }
};
