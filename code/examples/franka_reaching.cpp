#include <sqp_solver/sqp_solver.h>
#include <trajopt/ocp.h>
#include <trajopt/node.h>
#include <trajopt/constraints/integration_schemes/euler.h>
#include <trajopt/constraints/inverse_dynamics.h>
#include "trajopt/constraints/joint_limits_constraint.h"
#include <trajopt/costs/velocity_cost.h>
#include <trajopt/costs/acceleration_cost.h>
#include <trajopt/costs/frame_translation_cost.h>
#include <trajopt/constraints/frame_translation_constraint.h>
#include <trajopt/costs/frame_orientation_cost.h>

#include <pinocchio/parsers/urdf.hpp>
#include <pinocchio/algorithm/joint-configuration.hpp>
#include <pinocchio/algorithm/kinematics.hpp>
#include <pinocchio/algorithm/frames.hpp>

#include <cmath>
#include <iostream>

int main() {
    // ── Load Franka Panda ──
    const std::string urdf_path = "/workspace/code/resources/urdf/panda/panda.urdf";
    pinocchio::Model model;
    pinocchio::urdf::buildModel(urdf_path, model);

    std::cout << "Loaded " << model.name
              << " (nq=" << model.nq << ", nv=" << model.nv << ")\n";

    // ── Circle parameters ──
    const int N = 200;
    const double dt = 0.02;            // 2 second horizon
    const std::string ee_frame = "panda_ee";

    // Circle in the YZ plane, centered at the EE's ready-pose position
    const double radius = 0.2;
    const Eigen::Vector3d center(0.307, 0.0, 0.487-radius);
    

    // Classic Panda "ready" pose: EE at ~(0.307, 0.0, 0.487)
    // Circle starts at angle=0: center + (0, radius, 0) = (0.307, 0.1, 0.487)
    VectorXd q0(model.nq);
    q0 << 0.0, -M_PI/4, 0.0, -3*M_PI/4, 0.0, M_PI/2, M_PI/4;

    // Print info
    
    pinocchio::Data data(model);
    pinocchio::forwardKinematics(model, data, q0);
    pinocchio::updateFramePlacements(model, data);
    int fid = model.getFrameId(ee_frame);
    std::cout << "Initial EE position: "
                << data.oMf[fid].translation().transpose() << "\n";
    std::cout << "Circle center:       " << center.transpose()
                << "  radius: " << radius << "\n\n";

    Matrix3d R_target = data.oMf[fid].rotation();

    Vector3d ub;
    ub << 1e3, 0.1, 1e3;
    Vector3d lb;
    lb << -1e3, -0.1, -1e3;

    // ── Build OCP ──
    OCP ocp(N);
    for (int k = 0; k < N; k++) {
        Node node(model);

        if (k < N - 1) {
            node.add_dynamics(std::make_shared<EulerIntegration>(dt));
            node.add_constraint(std::make_shared<InvDynamics>());
            node.add_constraint(std::make_shared<JointLimitsConstraint>());
        }

        // Smoothness: penalize velocity and acceleration
        node.add_cost(std::make_shared<VelocityCost>(1e-6));
        if (k < N - 1) {
            node.add_cost(std::make_shared<AccelerationCost>(1e-6));
        }

        // Task-space waypoint: track circle at every node
        double t = static_cast<double>(k) / (N - 1);  // 0 to 1
        double angle = 2. * M_PI * t;
        Eigen::Vector3d p_target = center + Eigen::Vector3d(
            0.0,
            radius * std::sin(angle),
            radius * std::cos(angle)
        );

        double weight = (k == 0 || k == N - 1) ? 1e0 : 1e0;
        auto ee_cost = std::make_shared<FrameTranslationCost>(ee_frame, p_target, weight);
        node.add_cost(ee_cost);
        node.add_cost(std::make_shared<FrameOrientationCost>(ee_frame, R_target, 1e-3));

        auto ee_const = std::make_shared<FrameTranslationConstraint>(ee_frame);

        ee_const->set_bounds(lb, ub);

        node.add_constraint(ee_const);

        ocp.addNode(std::move(node));
    }
    ocp.finalize();

    // ── Initial guess ──
    for (int k = 0; k < N; k++) {
        ocp.get_node(k).q() = q0;
        ocp.get_node(k).v().setZero();
        if (k < N - 1) {
            ocp.get_node(k).u().setZero();
        }
    }

    // ── Solve ──
    SQPSolver solver(ocp);

    SQPoptions opts;
    opts.max_sqp_iters = 100;
    opts.ls_type = LSType::MERIT;
    opts.max_ls_iters = 5;
    opts.tolerance = 1e-2;
    solver.set_options(opts);

    std::cout << "Solving...\n";
    solver.solve();

    // ── Print results ──
    {
        pinocchio::Data data(model);
        std::cout << "\nEE trajectory (sampled):\n";
        for (int k = 0; k < N; k += N / 10) {
            VectorXd qk = ocp.get_node(k).q();
            pinocchio::forwardKinematics(model, data, qk);
            pinocchio::updateFramePlacements(model, data);
            int fid = model.getFrameId(ee_frame);
            Eigen::Vector3d p = data.oMf[fid].translation();

            double t = static_cast<double>(k) / (N - 1);
            double angle = 2.0 * M_PI * t;
            Eigen::Vector3d p_target = center + Eigen::Vector3d(
                0.0,
                radius * std::sin(angle),
                radius * std::cos(angle));

            std::cout << "  k=" << k
                      << "  pos=" << p.transpose()
                      << "  target=" << p_target.transpose()
                      << "  err=" << (p - p_target).norm() << "\n";
        }
    }

    // ── Save trajectory ──
    ocp.save_trajectory("/workspace/code/resources/trajectories/franka_reaching.json",
                        dt, urdf_path);
    std::cout << "Trajectory saved.\n";

    return 0;
}
