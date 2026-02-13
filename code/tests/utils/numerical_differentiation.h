#pragma once

#include <common/types.h>
#include <trajopt/node.h>
#include <functional>
#include <cmath>

/**
 * Compute numerical Jacobian using central differences.
 *
 * @param func Function f: R^n -> R^m
 * @param x Point at which to evaluate Jacobian
 * @param output_dim Dimension m of function output
 * @param eps Finite difference step size
 * @return Numerical Jacobian matrix (m x n)
 */
inline MatrixXd numerical_jacobian(
    std::function<VectorXd(const VectorXd&)> func,
    const VectorXd& x,
    int output_dim,
    double eps = 1e-7
) {
    const int n = x.size();
    MatrixXd jac(output_dim, n);

    VectorXd x_plus = x;
    VectorXd x_minus = x;

    for (int i = 0; i < n; ++i) {
        x_plus(i) = x(i) + eps;
        x_minus(i) = x(i) - eps;

        VectorXd f_plus = func(x_plus);
        VectorXd f_minus = func(x_minus);

        jac.col(i) = (f_plus - f_minus) / (2.0 * eps);

        x_plus(i) = x(i);
        x_minus(i) = x(i);
    }

    return jac;
}

/**
 * Compute numerical Jacobian w.r.t. [x, u] using Lie-group-aware perturbations.
 *
 * State is perturbed via node.x_oplus (pinocchio::integrate for q, + for v).
 * Control is perturbed via node.u_oplus (plain Euclidean addition, u ∈ R^nv).
 *
 * Returns a (output_dim × (ndx + ndu)) matrix: [∂f/∂x | ∂f/∂u].
 * Pass perturb_x=false or perturb_u=false to skip the respective block.
 *
 * @param func       Nullary function returning R^m; reads state/control from node
 * @param node       Node providing x_oplus, u_oplus, and current x/u
 * @param output_dim Dimension m of the function output
 * @param perturb_x  Whether to compute the state Jacobian block
 * @param perturb_u  Whether to compute the control Jacobian block
 * @param eps        Finite difference step size
 */
inline MatrixXd numerical_jacobian_node(
    std::function<VectorXd()> func,
    Node& node,
    int output_dim,
    bool perturb_x = true,
    bool perturb_u = true,
    double eps = 1e-7
) {
    const int ndx = node.ndx();
    const int ndu = node.ndu();
    const int total_cols = (perturb_x ? ndx : 0) + (perturb_u ? ndu : 0);
    MatrixXd jac(output_dim, total_cols);

    VectorXd x0 = node.x();
    VectorXd u0 = node.u();
    VectorXd x_pert(node.nx());
    VectorXd u_pert(ndu);

    int col = 0;

    if (perturb_x) {
        VectorXd dx = VectorXd::Zero(ndx);
        for (int i = 0; i < ndx; ++i) {
            dx(i) = eps;
            node.x_oplus(x0, dx, x_pert);
            node.x() = x_pert;
            VectorXd f_plus = func();

            dx(i) = -eps;
            node.x_oplus(x0, dx, x_pert);
            node.x() = x_pert;
            VectorXd f_minus = func();

            jac.col(col++) = (f_plus - f_minus) / (2.0 * eps);
            dx(i) = 0.0;
        }
        node.x() = x0;
    }

    if (perturb_u) {
        VectorXd du = VectorXd::Zero(ndu);
        for (int i = 0; i < ndu; ++i) {
            du(i) = eps;
            node.u_oplus(u0, du, u_pert);
            node.u() = u_pert;
            VectorXd f_plus = func();

            du(i) = -eps;
            node.u_oplus(u0, du, u_pert);
            node.u() = u_pert;
            VectorXd f_minus = func();

            jac.col(col++) = (f_plus - f_minus) / (2.0 * eps);
            du(i) = 0.0;
        }
        node.u() = u0;
    }

    return jac;
}

/**
 * Compare analytical and numerical Jacobians with tolerance.
 * Uses relative error for large values, absolute for small.
 */
inline bool jacobians_match(
    const MatrixXd& analytical,
    const MatrixXd& numerical,
    double rel_tol = 1e-5,
    double abs_tol = 1e-8
) {
    if (analytical.rows() != numerical.rows() ||
        analytical.cols() != numerical.cols()) {
        return false;
    }

    for (int i = 0; i < analytical.rows(); ++i) {
        for (int j = 0; j < analytical.cols(); ++j) {
            double a = analytical(i, j);
            double n = numerical(i, j);
            double diff = std::abs(a - n);
            double scale = std::max(std::abs(a), std::abs(n));

            if (scale > abs_tol) {
                if (diff / scale > rel_tol) return false;
            } else {
                if (diff > abs_tol) return false;
            }
        }
    }
    return true;
}
