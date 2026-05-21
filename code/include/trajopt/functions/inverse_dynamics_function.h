#pragma once

#include <trajopt/node.h>
#include <trajopt/functions/abstract_function.h>

#include "pinocchio/algorithm/rnea.hpp"
#include "pinocchio/algorithm/rnea-derivatives.hpp"
#include "pinocchio/algorithm/frames.hpp"


class InvDynamicsFunction : public AbstractFunction {
private:
    // Pre-allocated temporaries
    VectorXd _q;
    VectorXd _vq;
    VectorXd _aq;
    MatrixXd _dtau_dq;
    MatrixXd _dtau_dv;
    MatrixXd _dtau_da;

    // External forces (size = model.njoints, all initialized to zero)
    std::vector<pinocchio::Force> _fext;

    // Frame Jacobian scratch buffer (6 x nv)
    MatrixXd _Jframe;

private:
    // Helper to build external forces vector from contact forces
    void build_fext();

public:
    explicit InvDynamicsFunction();

    void allocate_dims() override;
    void evaluate_impl() override;
    void jacobian_impl() override;
};
