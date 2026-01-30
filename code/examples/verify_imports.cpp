#include <iostream>

// Eigen
#include <Eigen/Dense>

// HPIPM (C library)
extern "C" {
#include <hpipm_d_ocp_qp_dim.h>
#include <hpipm_d_ocp_qp.h>
}

// Pinocchio
#include <pinocchio/multibody/model.hpp>
#include <pinocchio/multibody/data.hpp>
#include <pinocchio/algorithm/joint-configuration.hpp>

int main() {
    std::cout << "=== Library Verification ===" << std::endl;

    // Test Eigen
    std::cout << "\n[Eigen] ";
    Eigen::Matrix3d A = Eigen::Matrix3d::Identity();
    Eigen::Vector3d v(1.0, 2.0, 3.0);
    Eigen::Vector3d result = A * v;
    std::cout << "OK - Matrix-vector product: ["
              << result.transpose() << "]" << std::endl;

    // Test HPIPM
    std::cout << "\n[HPIPM] ";
    int N = 10;  // Horizon length
    hpipm_size_t dim_size = d_ocp_qp_dim_memsize(N);
    void* dim_mem = malloc(dim_size);
    struct d_ocp_qp_dim dim;
    d_ocp_qp_dim_create(N, &dim, dim_mem);
    std::cout << "OK - Created OCP QP dimension for N=" << N
              << ", mem=" << dim_size << " bytes" << std::endl;
    free(dim_mem);

    // Test Pinocchio
    std::cout << "\n[Pinocchio] ";
    pinocchio::Model model;
    model.name = "test_model";
    pinocchio::Data data(model);
    std::cout << "OK - Created model '" << model.name
              << "' with " << model.nq << " DOF" << std::endl;

    std::cout << "\n=== All libraries imported successfully ===" << std::endl;
    return 0;
}
