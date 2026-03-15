#include <sqp_solver/sqp_solver.h>
#include <trajopt/ocp.h>
#include <trajopt/node.h>
#include <trajopt/scheduler.h>
#include <trajopt/constraints/integration_schemes/semi-euler.h>
#include <trajopt/constraints/integration_schemes/euler.h>
#include <trajopt/constraints/inverse_dynamics.h>
#include <trajopt/constraints/joint_limits_constraint.h>
#include <trajopt/constraints/contact_constraint.h>
#include <trajopt/constraints/friction_cone_constraint.h>
#include <trajopt/costs/velocity_cost.h>
#include <trajopt/costs/acceleration_cost.h>
#include <trajopt/costs/configuration_cost.h>

#include <pinocchio/parsers/urdf.hpp>
#include <pinocchio/algorithm/joint-configuration.hpp>
#include <pinocchio/algorithm/kinematics.hpp>
#include <pinocchio/algorithm/frames.hpp>

#include <cmath>
#include <iostream>

int main() {
    // ── Load Go2 quadruped ──
    const std::string urdf_path = "/workspace/code/resources/urdf/go2/go2.urdf";
    pinocchio::Model model;
    pinocchio::urdf::buildModel(urdf_path, pinocchio::JointModelFreeFlyer(), model);

    std::cout << "Loaded " << model.name
              << " (nq=" << model.nq << ", nv=" << model.nv << ")\n";

    // ── Parameters ──
    int N = 5;
    const double dt = 0.01;
    const double mu = 0.8;  // friction coefficient

    // Foot frame names
    const std::vector<std::string> feet = {"FL_foot", "FR_foot", "RL_foot", "RR_foot"};


    std::cout << "Horizon: N=" << N << " nodes, T=" << N * dt << "s\n";

    // ── Nominal standing configuration ──
    // Floating base: [x, y, z, qx, qy, qz, qw] + 12 joint angles
    VectorXd q0 = pinocchio::neutral(model);
    q0[2] = 0.30;  // base height ~30cm

    // Hip, thigh, calf angles for a standing pose
    // FL
    q0[7]  =  0.0;    // FL_hip
    q0[8]  =  0.8;    // FL_thigh
    q0[9]  = -1.6;    // FL_calf
    // FR
    q0[10] =  0.0;    // FR_hip
    q0[11] =  0.8;    // FR_thigh
    q0[12] = -1.6;    // FR_calf
    // RL
    q0[13] =  0.0;    // RL_hip
    q0[14] =  0.8;    // RL_thigh
    q0[15] = -1.6;    // RL_calf
    // RR
    q0[16] =  0.0;    // RR_hip
    q0[17] =  0.8;    // RR_thigh
    q0[18] = -1.6;    // RR_calf

    // Compute actual base height from FK so feet touch ground
    {
        pinocchio::Data data(model);
        pinocchio::forwardKinematics(model, data, q0);
        pinocchio::updateFramePlacements(model, data);

        // Find lowest foot z-coordinate
        double min_foot_z = 1e10;
        for (const auto& foot : feet) {
            int fid = model.getFrameId(foot);
            double fz = data.oMf[fid].translation()[2];
            min_foot_z = std::min(min_foot_z, fz);
        }
        // Adjust base height so feet are at ground level (z=0)
        q0[2] -= min_foot_z;
    }

    // ── Build OCP ──
    OCP ocp(N);

    for (int k = 0; k < N; k++) {
        Node node(model);

        // Register all four feet as contacts
        node.add_contacts(feet);

        if (k < N - 1) {
            node.add_dynamics(std::make_shared<SemiEulerIntegration>(dt));
            node.add_constraint(std::make_shared<InvDynamics>());
            // node.add_constraint(std::make_shared<JointLimitsConstraint>());
        }

        // Contact and friction constraints for each foot
        for (const auto& foot : feet) {
            
            if (k < N - 1) {
                node.add_constraint(std::make_shared<ContactConstraint>(foot));
                node.add_constraint(std::make_shared<FrictionConeConstraint>(foot, mu));
            }
        }

        // Costs
        node.add_cost(std::make_shared<ConfigurationCost>(q0, 1e-3));
        if (k == N-1) node.add_cost(std::make_shared<VelocityCost>(1e3));
        if (k < N - 1) {
            node.add_cost(std::make_shared<VelocityCost>(1e-3));
            node.add_cost(std::make_shared<AccelerationCost>(1e-9));
        }

        ocp.addNode(std::move(node));
    }
    ocp.finalize();

    // ── Initial guess: standing pose with zero velocity ──
    std::cout<<"q0: "<<q0.transpose()<<std::endl;
    for (int k = 0; k < N; k++) {
        ocp.get_node(k).q() = q0;
        ocp.get_node(k).v().setZero();
        ocp.get_node(k).u().setZero();
    }

    // ── Solve ──
    SQPSolver solver(ocp);

    SQPoptions opts;
    opts.max_sqp_iters = 200;
    opts.ls_type = LSType::MERIT;
    opts.max_ls_iters = 5;
    opts.tolerance = 1e-3;
    // opts.verbose = 2;
    solver.set_options(opts);

    std::cout << "Solving trotting trajectory...\n";
    solver.solve();

    // ── Save trajectory ──
    ocp.save_trajectory("/workspace/code/resources/trajectories/go2_stance.json",
                        dt, urdf_path);

    return 0;
}
