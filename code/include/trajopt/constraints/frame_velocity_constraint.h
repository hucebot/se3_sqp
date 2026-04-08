#pragma once

#include <pinocchio/multibody/fwd.hpp>
#include <trajopt/constraints/abstract_constraint.h>
#include <trajopt/functions/frame_velocity.h>


/**
 * FrameVelocityConstraint constrains a frame's twist between -v_ref and v_ref.
 *
 */


class FrameVelocityConstraint : public AbstractConstraint {
private:
    FrameVelocity _fv;
    Vector6d _v_ref;

public:

    explicit FrameVelocityConstraint(const std::string& frame_name,
                                        const Vector6d& v_ref = Vector6d::Zero());

    void set_node(Node* node) override { AbstractConstraint::set_node(node); _fv.set_node(node); }


    void allocate_dims() override;
    void evaluate_impl() override;
    void jacobian_impl() override;

    void set_ref(const Vector6d& v_ref) {
        _v_ref = v_ref;
    }
    const Vector6d& get_ref() const { return _v_ref; }
    void set_re_reference_frame(const pinocchio::ReferenceFrame& re_ref_rame) { _fv.set_re_reference_frame(re_ref_rame); }
    void set_base_frame_name(const std::string& base_frame_name) { _fv.set_base_frame_name(base_frame_name); }
};
