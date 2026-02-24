#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>

#include <pinocchio/parsers/urdf.hpp>

#include <sqp_solver/sqp_solver.h>
#include <trajopt/ocp.h>
#include <trajopt/node.h>
#include <trajopt/scheduler.h>
#include <trajopt/costs/abstract_cost.h>
#include <trajopt/costs/configuration_cost.h>
#include <trajopt/costs/velocity_cost.h>
#include <trajopt/costs/acceleration_cost.h>
#include <trajopt/costs/frame_translation_cost.h>
#include <trajopt/costs/frame_orientation_cost.h>
#include <trajopt/constraints/abstract_constraint.h>
#include <trajopt/constraints/inverse_dynamics.h>
#include <trajopt/constraints/joint_limits_constraint.h>
#include <trajopt/constraints/contact_constraint.h>
#include <trajopt/constraints/friction_cone_constraint.h>
#include <trajopt/constraints/frame_translation_constraint.h>
#include <trajopt/constraints/integration_schemes/euler.h>
#include <trajopt/constraints/integration_schemes/semi-euler.h>

namespace py = pybind11;

PYBIND11_MODULE(sqp_solver, m) {
    m.doc() = "Python bindings for the SQP trajectory optimization solver";

    // ── LSType enum ──────────────────────────────────────────────────────────
    py::enum_<LSType>(m, "LSType")
        .value("NONE",   LSType::NONE)
        .value("MERIT",  LSType::MERIT)
        .value("FILTER", LSType::FILTER);

    // ── SQPoptions ───────────────────────────────────────────────────────────
    py::class_<SQPoptions>(m, "SQPoptions")
        .def(py::init<>())
        .def_readwrite("max_sqp_iters",        &SQPoptions::max_sqp_iters)
        .def_readwrite("ls_type",              &SQPoptions::ls_type)
        .def_readwrite("max_ls_iters",         &SQPoptions::max_ls_iters)
        .def_readwrite("ls_scale_factor",      &SQPoptions::ls_scale_factor)
        .def_readwrite("ls_merit_eta",         &SQPoptions::ls_merit_eta)
        .def_readwrite("tolerance",            &SQPoptions::tolerance)
        .def_readwrite("regularization",       &SQPoptions::regularization)
        .def_readwrite("regularization_scale", &SQPoptions::regularization_scale)
        .def("print", &SQPoptions::print);

    // ── SQPstatistics ────────────────────────────────────────────────────────
    py::class_<SQPstatistics>(m, "SQPstatistics")
        .def_readonly("number_of_iterations",       &SQPstatistics::number_of_iterations)
        .def_readonly("total_cost",                 &SQPstatistics::total_cost)
        .def_readonly("total_constraint_violation", &SQPstatistics::total_constraint_violation)
        .def_readonly("total_dynamics_defect",      &SQPstatistics::total_dynamics_defect)
        .def_readonly("step_norm",                  &SQPstatistics::step_norm)
        .def_readonly("dual_infeasibility",         &SQPstatistics::dual_infeasibility)
        .def_readonly("linesearch_iterations",      &SQPstatistics::linesearch_iterations)
        .def_readonly("qp_status",                  &SQPstatistics::qp_status)
        .def_readonly("qp_iterations",              &SQPstatistics::qp_iterations)
        .def_readonly("total_time_ms",              &SQPstatistics::total_time_ms)
        .def_readonly("last_iteration_time_ms",     &SQPstatistics::last_iteration_time_ms)
        .def("print", [](const SQPstatistics& s, int v) { s.print(v); },
             py::arg("verbosity") = 1);

    // ── Abstract base classes (needed for shared_ptr hierarchy) ──────────────
    py::class_<AbstractCost, std::shared_ptr<AbstractCost>>(m, "AbstractCost");
    py::class_<AbstractConstraint, std::shared_ptr<AbstractConstraint>>(m, "AbstractConstraint");

    // ── Costs ────────────────────────────────────────────────────────────────
    py::class_<ConfigurationCost, AbstractCost, std::shared_ptr<ConfigurationCost>>(m, "ConfigurationCost")
        .def(py::init<const VectorXd&, double>(),
             py::arg("q_ref"), py::arg("weight") = 1.0)
        .def(py::init<const VectorXd&, const MatrixXd&>(),
             py::arg("q_ref"), py::arg("weight"))
        .def("set_q_ref", &ConfigurationCost::set_q_ref)
        .def("get_q_ref", &ConfigurationCost::get_q_ref);

    py::class_<VelocityCost, AbstractCost, std::shared_ptr<VelocityCost>>(m, "VelocityCost")
        .def(py::init<double>(), py::arg("weight") = 1.0)
        .def(py::init<const VectorXd&, double>(),
             py::arg("v_ref"), py::arg("weight") = 1.0);

    py::class_<AccelerationCost, AbstractCost, std::shared_ptr<AccelerationCost>>(m, "AccelerationCost")
        .def(py::init<double>(), py::arg("weight") = 1.0)
        .def(py::init<const VectorXd&, double>(),
             py::arg("a_ref"), py::arg("weight") = 1.0);

    py::class_<FrameTranslationCost, AbstractCost, std::shared_ptr<FrameTranslationCost>>(m, "FrameTranslationCost")
        .def(py::init<const std::string&, const Eigen::Vector3d&, double>(),
             py::arg("frame_name"),
             py::arg("p_ref")  = Eigen::Vector3d::Zero(),
             py::arg("weight") = 1.0)
        .def(py::init<const std::string&, const Eigen::Vector3d&, const MatrixXd&>(),
             py::arg("frame_name"), py::arg("p_ref"), py::arg("weight"))
        .def("set_ref", &FrameTranslationCost::set_ref)
        .def("get_ref", &FrameTranslationCost::get_ref);

    py::class_<FrameOrientationCost, AbstractCost, std::shared_ptr<FrameOrientationCost>>(m, "FrameOrientationCost")
        .def(py::init<const std::string&, const Eigen::Matrix3d&, double>(),
             py::arg("frame_name"),
             py::arg("R_ref")   = Eigen::Matrix3d::Identity(),
             py::arg("weight")  = 1.0)
        .def(py::init<const std::string&, const Eigen::Matrix3d&, const MatrixXd&>(),
             py::arg("frame_name"), py::arg("R_ref"), py::arg("weight"))
        .def("set_ref", &FrameOrientationCost::set_ref)
        .def("get_ref", &FrameOrientationCost::get_ref);

    // ── Constraints ──────────────────────────────────────────────────────────
    py::class_<EulerIntegration, AbstractConstraint, std::shared_ptr<EulerIntegration>>(m, "EulerIntegration")
        .def(py::init<double>(), py::arg("dt"));

    py::class_<SemiEulerIntegration, AbstractConstraint, std::shared_ptr<SemiEulerIntegration>>(m, "SemiEulerIntegration")
        .def(py::init<double>(), py::arg("dt"));

    py::class_<InvDynamics, AbstractConstraint, std::shared_ptr<InvDynamics>>(m, "InvDynamics")
        .def(py::init<>());

    py::class_<JointLimitsConstraint, AbstractConstraint, std::shared_ptr<JointLimitsConstraint>>(m, "JointLimitsConstraint")
        .def(py::init<>());

    py::class_<ContactConstraint, AbstractConstraint, std::shared_ptr<ContactConstraint>>(m, "ContactConstraint")
        .def(py::init<const std::string&, double>(),
             py::arg("frame_name"), py::arg("ground_height") = 0.0)
        .def("set_ground_height", &ContactConstraint::set_ground_height);

    py::class_<FrictionConeConstraint, AbstractConstraint, std::shared_ptr<FrictionConeConstraint>>(m, "FrictionConeConstraint")
        .def(py::init<const std::string&, double>(),
             py::arg("frame_name"), py::arg("mu") = 0.5)
        .def("set_friction_coefficient", &FrictionConeConstraint::set_friction_coefficient);

    py::class_<FrameTranslationConstraint, AbstractConstraint, std::shared_ptr<FrameTranslationConstraint>>(m, "FrameTranslationConstraint")
        .def(py::init<const std::string&, const Eigen::Vector3d&>(),
             py::arg("frame_name"),
             py::arg("p_ref") = Eigen::Vector3d::Zero())
        .def("set_ref", &FrameTranslationConstraint::set_ref)
        .def("get_ref", &FrameTranslationConstraint::get_ref);

    // ── ContactScheduler ─────────────────────────────────────────────────────
    py::class_<ContactScheduler>(m, "ContactScheduler")
        .def(py::init<>())
        .def("define_contact", &ContactScheduler::define_contact,
             py::arg("contact_name"), py::arg("contact_frame_names"))
        .def("addPhase", &ContactScheduler::addPhase,
             py::arg("contacts_list"), py::arg("duration"),
             py::arg("sequence_name") = "_")
        .def("getNodesNumber", &ContactScheduler::getNodesNumber)
        .def("getSequence", &ContactScheduler::getSequence,
             py::arg("sampling_rate"),
             py::arg("sequence_name") = "_",
             py::arg("nodes_number")  = -1,
             py::arg("current_time")  = 0.0);

    // ── Node ─────────────────────────────────────────────────────────────────
    // Node takes a URDF path and builds the pinocchio model internally using
    // the system C++ pinocchio. This avoids any type mismatch between the
    // pip-installed pinocchio Python package (separate .so) and the system
    // pinocchio our library links against.
    // floating_base=True adds a JointModelFreeFlyer root joint (e.g. for Go2).
    py::class_<Node>(m, "Node")
        .def(py::init([](const std::string& urdf_path, bool floating_base) {
            pinocchio::Model model;
            if (floating_base)
                pinocchio::urdf::buildModel(
                    urdf_path, pinocchio::JointModelFreeFlyer(), model);
            else
                pinocchio::urdf::buildModel(urdf_path, model);
            return Node(model);
        }), py::arg("urdf_path"), py::arg("floating_base") = false)
        .def("add_cost",            &Node::add_cost)
        .def("add_dynamics",        &Node::add_dynamics)
        .def("add_constraint",      &Node::add_constraint)
        .def("add_contact",         &Node::add_contact)
        .def("add_contacts",        &Node::add_contacts)
        .def("set_active_contacts", &Node::set_active_contacts)
        .def("n_contacts",          &Node::n_contacts)
        .def("nq", &Node::nq)
        .def("nv", &Node::nv)
        .def("nx", &Node::nx)
        .def("nu", &Node::nu)
        // Zero-copy numpy views into trajectory memory (valid after ocp.finalize())
        .def("q",  [](Node& n) -> Eigen::Ref<Eigen::VectorXd> { return n.q(); })
        .def("v",  [](Node& n) -> Eigen::Ref<Eigen::VectorXd> { return n.v(); })
        .def("a",  [](Node& n) -> Eigen::Ref<Eigen::VectorXd> { return n.a(); })
        .def("x",  [](Node& n) -> Eigen::Ref<Eigen::VectorXd> { return n.x(); })
        .def("u",  [](Node& n) -> Eigen::Ref<Eigen::VectorXd> { return n.u(); })
        .def("fc", [](Node& n, int i) -> Eigen::Ref<Eigen::VectorXd> { return n.fc(i); })
        .def("tau", [](const Node& n) { return Eigen::VectorXd(n.tau()); });

    // ── OCP ──────────────────────────────────────────────────────────────────
    // addNode takes Node&& — we accept by reference and std::move into the OCP.
    py::class_<OCP>(m, "OCP")
        .def(py::init<int>())
        .def("addNode", [](OCP& ocp, Node& node) { ocp.addNode(std::move(node)); })
        .def("finalize",    &OCP::finalize)
        .def("num_nodes",   &OCP::num_nodes)
        .def("get_node",    (Node& (OCP::*)(int)) &OCP::get_node,
             py::return_value_policy::reference_internal)
        .def("cost",                 &OCP::cost)
        .def("dynamics_defect",      &OCP::dynamics_defect)
        .def("constraint_violation", &OCP::constraint_violation)
        .def("x_traj", [](OCP& ocp) {
            py::list result;
            for (auto& v : ocp.x_traj()) result.append(Eigen::VectorXd(v));
            return result;
        })
        .def("u_traj", [](OCP& ocp) {
            py::list result;
            for (auto& v : ocp.u_traj()) result.append(Eigen::VectorXd(v));
            return result;
        })
        .def("save_trajectory", &OCP::save_trajectory,
             py::arg("filepath"), py::arg("dt") = 0.0, py::arg("urdf_path") = "");

    // ── SQPSolver ────────────────────────────────────────────────────────────
    py::class_<SQPSolver>(m, "SQPSolver")
        .def(py::init<OCP&>())
        .def("set_options", &SQPSolver::set_options)
        .def("solve",       &SQPSolver::solve)
        .def("get_stats",   &SQPSolver::get_stats,
             py::return_value_policy::reference_internal);
}
