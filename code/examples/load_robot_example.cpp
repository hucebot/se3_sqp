#include <pinocchio/parsers/urdf.hpp>
#include <pinocchio/multibody/model.hpp>
#include <pinocchio/multibody/data.hpp>
#include <pinocchio/algorithm/joint-configuration.hpp>
#include <pinocchio/algorithm/kinematics.hpp>


#include <iostream>

int main() {
    // Load UR5 robot from example-robot-data
    const std::string urdf_path = "/workspace/code/resources/pendubot.urdf";
        

    pinocchio::Model model;
    pinocchio::urdf::buildModel(urdf_path, model);
    pinocchio::Data data(model);

    std::cout << "=== Robot Model Info ===" << std::endl;
    std::cout << "Model name: " << model.name << std::endl;
    std::cout << "Number of joints: " << model.njoints << std::endl;
    std::cout << "Number of DOFs (nv): " << model.nv << std::endl;
    std::cout << "Configuration dimension (nq): " << model.nq << std::endl;

    std::cout << "\n=== Joint Names ===" << std::endl;
    for (int i = 0; i < model.njoints; ++i) {
        std::cout << "  [" << i << "] " << model.names[i] << std::endl;
    }

    std::cout << "\n=== Joint Limits ===" << std::endl;
    std::cout << "Lower: " << model.lowerPositionLimit.transpose() << std::endl;
    std::cout << "Upper: " << model.upperPositionLimit.transpose() << std::endl;

    // Generate a random valid configuration
    Eigen::VectorXd q = pinocchio::randomConfiguration(model);
    std::cout << "\n=== Random Configuration ===" << std::endl;
    std::cout << "q: " << q.transpose() << std::endl;

    // Compute forward kinematics
    pinocchio::forwardKinematics(model, data, q);

    std::cout << "\n=== End-Effector Pose ===" << std::endl;
    const auto& ee_frame = data.oMi[model.njoints - 1];
    std::cout << "Position: " << ee_frame.translation().transpose() << std::endl;
    std::cout << "Rotation:\n" << ee_frame.rotation() << std::endl;

    return 0;
}
