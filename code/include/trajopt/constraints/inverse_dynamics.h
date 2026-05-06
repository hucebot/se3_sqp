#pragma once

#include <trajopt/node.h>
#include <trajopt/constraints/abstract_constraint.h>
#include <trajopt/functions/inverse_dynamics_function.h>

#include "pinocchio/algorithm/rnea.hpp"
#include "pinocchio/algorithm/rnea-derivatives.hpp"
#include "pinocchio/algorithm/frames.hpp"


class InvDynamics : public AbstractConstraint {
   private:
    InvDynamicsFunction _id;

   public:
    explicit InvDynamics();
    void set_node(Node* node) override { AbstractConstraint::set_node(node); _id.set_node(node); }

    void allocate_dims() override;

    void evaluate_impl() override;

    void jacobian_impl() override;
};
