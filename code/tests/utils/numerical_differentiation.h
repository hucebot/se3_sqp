#pragma once

#include <common/types.h>
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
