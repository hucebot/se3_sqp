#include <iostream>

// Eigen
#include <Eigen/Dense>

// HPIPM (C library)
extern "C" {
#include <hpipm_d_ocp_qp.h>
#include <hpipm_d_ocp_qp_dim.h>
}

#include <sqp_solver/sqp_solver.h>

// Pinocchio
#include <pinocchio/algorithm/joint-configuration.hpp>
#include <pinocchio/multibody/data.hpp>
#include <pinocchio/multibody/model.hpp>

int main() {
    std::cout << "=== Library Verification ===" << std::endl;

    // Test Eigen
    std::cout << "\n[Eigen] ";
    Eigen::Matrix3d A = Eigen::Matrix3d::Identity();
    Eigen::Vector3d v(1.0, 2.0, 3.0);
    Eigen::Vector3d result = A * v;
    std::cout << "OK - Matrix-vector product: [" << result.transpose() << "]"
              << std::endl;

    // SQPSolver mpla;

    // Test Pinocchio
    std::cout << "\n[Pinocchio] ";
    pinocchio::Model model;
    model.name = "test_model";
    pinocchio::Data data(model);
    std::cout << "OK - Created model '" << model.name << "' with " << model.nq
              << " DOF" << std::endl;

    std::cout << "\n=== All libraries imported successfully ===" << std::endl;
    return 0;
}
