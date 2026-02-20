#pragma once

#include <Eigen/Dense>
#include <iostream>

#ifndef NDEBUG
    #define DEBUG_PRINT(x) std::cout <<std::scientific << std::setprecision(2) << x << std::endl
#else
    #define DEBUG_PRINT(x)
#endif


using str = std::string;

// =============================================================================
// Vector Types
// =============================================================================

using VectorXd = Eigen::VectorXd;
using Vector2d = Eigen::Vector2d;
using Vector3d = Eigen::Vector3d;
using Vector4d = Eigen::Vector4d;

// References
using VectorXdRef = Eigen::Ref<Eigen::VectorXd>;
using VectorXdConstRef = Eigen::Ref<const Eigen::VectorXd>;

// Pointers
using VectorXdPtr = Eigen::VectorXd*;
using VectorXdConstPtr = const Eigen::VectorXd*;

// =============================================================================
// Matrix Types
// =============================================================================

// Column-major (default) - required by HPIPM
using MatrixXd = Eigen::MatrixXd;
using Matrix2d = Eigen::Matrix2d;
using Matrix3d = Eigen::Matrix3d;
using Matrix4d = Eigen::Matrix4d;

// References
using MatrixXdRef = Eigen::Ref<Eigen::MatrixXd>;
using MatrixXdConstRef = Eigen::Ref<const Eigen::MatrixXd>;

// Pointers
using MatrixXdPtr = Eigen::MatrixXd*;
using MatrixXdConstPtr = const Eigen::MatrixXd*;

// =============================================================================
// Map Types (wrapping external memory)
// =============================================================================

using VectorXdMap = Eigen::Map<Eigen::VectorXd>;
using VectorXdConstMap = Eigen::Map<const Eigen::VectorXd>;

using MatrixXdMap = Eigen::Map<Eigen::MatrixXd>;
using MatrixXdConstMap = Eigen::Map<const Eigen::MatrixXd>;