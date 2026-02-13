#pragma once

#include <trajopt/node.h>
#include <trajopt/constraints/abstract_constraint.h>

#include "pinocchio/algorithm/rnea.hpp"
#include "pinocchio/algorithm/rnea-derivatives.hpp"


class InvDynamics : public AbstractConstraint {
   private:
    // Pre-allocated temporaries
    VectorXd _q;
    VectorXd _vq;
    VectorXd _aq;
    VectorXd _res;
    MatrixXd _dtau_dq;
    MatrixXd _dtau_dv;
    MatrixXd _dtau_da;

   public:
    explicit InvDynamics();

    void allocate_slices() override;

    void evaluate_impl(VectorXdRef output) override;

    void jacobian_impl(MatrixXdRef jac) override;

    MatrixXdConstRef get_jac_x() const override;
    MatrixXd get_jac_u() const override;
};
