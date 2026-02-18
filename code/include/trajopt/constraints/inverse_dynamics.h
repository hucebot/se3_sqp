#pragma once

#include <trajopt/node.h>
#include <trajopt/constraints/abstract_constraint.h>

#include "pinocchio/algorithm/rnea.hpp"
#include "pinocchio/algorithm/rnea-derivatives.hpp"
#include "pinocchio/algorithm/frames.hpp"


class InvDynamics : public AbstractConstraint {
   private:
    // Pre-allocated temporaries
    VectorXd _q;
    VectorXd _vq;
    VectorXd _aq;
    MatrixXd _dtau_dq;
    MatrixXd _dtau_dv;
    MatrixXd _dtau_da;

    // External forces (size = model.njoints, all initialized to zero)
    PINOCCHIO_ALIGNED_STD_VECTOR(pinocchio::Force) _fext;

    // Frame Jacobian scratch buffer (6 x nv)
    MatrixXd _Jframe;

   private:
    // Helper to build external forces vector from contact forces
    void build_fext();

   public:
    explicit InvDynamics();

    void allocate_slices() override;

    void evaluate_impl() override;

    void jacobian_impl() override;

    MatrixXdConstRef get_jac_x() const override;
    MatrixXd get_jac_u() const override;
};
