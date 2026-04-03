#include <boost/python.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>
#include <eigenpy/eigenpy.hpp>

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
#include <trajopt/costs/frame_velocity_cost.h>
#include <trajopt/costs/frame_acceleration_cost.h>
#include <trajopt/constraints/abstract_constraint.h>
#include <trajopt/constraints/inverse_dynamics.h>
#include <trajopt/constraints/joint_limits_constraint.h>
#include <trajopt/constraints/contact_constraint.h>
#include <trajopt/constraints/friction_cone_constraint.h>
#include <trajopt/constraints/frame_translation_constraint.h>
#include <trajopt/constraints/frame_orientation_constraint.h>
#include <trajopt/constraints/frame_velocity_constraint.h>
#include <trajopt/constraints/integration_schemes/euler.h>
#include <trajopt/constraints/integration_schemes/semi-euler.h>

namespace bp = boost::python;

// ============================================================================
// Free functions for Node (factory and accessors)
// ============================================================================

// Thin wrappers for Eigen::Ref returns (zero-copy numpy views)
static Eigen::Ref<Eigen::VectorXd> node_q(Node& n)  { return n.q(); }
static Eigen::Ref<Eigen::VectorXd> node_v(Node& n)  { return n.v(); }
static Eigen::Ref<Eigen::VectorXd> node_a(Node& n)  { return n.a(); }
static Eigen::Ref<Eigen::VectorXd> node_x(Node& n)  { return n.x(); }
static Eigen::Ref<Eigen::VectorXd> node_u(Node& n)  { return n.u(); }
static Eigen::Ref<Eigen::VectorXd> node_fc(Node& n, int i) { return n.fc(i); }
static Eigen::VectorXd node_tau(const Node& n) { return Eigen::VectorXd(n.tau()); }

// ============================================================================
// Free functions for OCP
// ============================================================================

static void ocp_add_node(OCP& ocp, Node& node) {
    ocp.addNode(std::move(node));
}

static bp::list ocp_x_traj(OCP& ocp) {
    bp::list result;
    for (auto& v : ocp.x_traj()) result.append(Eigen::VectorXd(v));
    return result;
}

static bp::list ocp_u_traj(OCP& ocp) {
    bp::list result;
    for (auto& v : ocp.u_traj()) result.append(Eigen::VectorXd(v));
    return result;
}

// ============================================================================
// Free function for SQPstatistics print
// ============================================================================

static void stats_print(const SQPstatistics& s, int verbosity) {
    s.print(verbosity);
}

// ============================================================================
// Module definition
// ============================================================================

BOOST_PYTHON_MODULE(sqp_solver) {
    // Enable eigenpy Eigen <-> numpy converters
    eigenpy::enableEigenPy();
    eigenpy::enableEigenPySpecific<Eigen::Matrix<double,6,1>>();

    // ── LSType enum ────────────────────────────────────────────────────────
    bp::enum_<LSType>("LSType")
        .value("NONE",   LSType::NONE)
        .value("MERIT",  LSType::MERIT)
        .value("FILTER", LSType::FILTER);

    // ── SQPoptions ─────────────────────────────────────────────────────────
    bp::class_<SQPoptions>("SQPoptions")
        .def(bp::init<>())
        .def_readwrite("max_sqp_iters",        &SQPoptions::max_sqp_iters)
        .def_readwrite("ls_type",              &SQPoptions::ls_type)
        .def_readwrite("max_ls_iters",         &SQPoptions::max_ls_iters)
        .def_readwrite("ls_scale_factor",      &SQPoptions::ls_scale_factor)
        .def_readwrite("ls_merit_eta",         &SQPoptions::ls_merit_eta)
        .def_readwrite("ls_merit_mu",          &SQPoptions::ls_merit_mu)
        .def_readwrite("tolerance",            &SQPoptions::tolerance)
        .def_readwrite("regularization",       &SQPoptions::regularization)
        .def_readwrite("regularization_scale", &SQPoptions::regularization_scale)
        .def_readwrite("eps_inequality",       &SQPoptions::eps_inequality)
        .def_readwrite("verbose",              &SQPoptions::verbose)
        .def_readwrite("hpipm_iter_max",       &SQPoptions::hpipm_iter_max)
        .def_readwrite("hpipm_tol_stat",       &SQPoptions::hpipm_tol_stat)
        .def_readwrite("hpipm_tol_eq",         &SQPoptions::hpipm_tol_eq)
        .def_readwrite("hpipm_tol_ineq",       &SQPoptions::hpipm_tol_ineq)
        .def_readwrite("hpipm_tol_comp",       &SQPoptions::hpipm_tol_comp)
        .def_readwrite("hpipm_warm_start",     &SQPoptions::hpipm_warm_start)
        .def("print", &SQPoptions::print);

    // ── SQPstatistics ──────────────────────────────────────────────────────
    bp::class_<SQPstatistics>("SQPstatistics", bp::no_init)
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
        .def("print", &stats_print, (bp::arg("verbosity") = 1));

    // ── Abstract base classes ──────────────────────────────────────────────
    bp::class_<AbstractCost, std::shared_ptr<AbstractCost>,
               boost::noncopyable>("AbstractCost", bp::no_init);

    bp::class_<AbstractConstraint, std::shared_ptr<AbstractConstraint>,
               boost::noncopyable>("AbstractConstraint", bp::no_init);

    // ── Costs ──────────────────────────────────────────────────────────────

    // ConfigurationCost
    bp::class_<ConfigurationCost, bp::bases<AbstractCost>,
               std::shared_ptr<ConfigurationCost>>("ConfigurationCost",
        bp::init<const VectorXd&, double>(
            (bp::arg("q_ref"), bp::arg("weight") = 1.0)))
        .def(bp::init<const VectorXd&, const MatrixXd&>(
            (bp::arg("q_ref"), bp::arg("weight"))))
        .def("set_q_ref", &ConfigurationCost::set_q_ref)
        .def("get_q_ref", &ConfigurationCost::get_q_ref,
             bp::return_value_policy<bp::copy_const_reference>());
    bp::implicitly_convertible<std::shared_ptr<ConfigurationCost>,
                               std::shared_ptr<AbstractCost>>();

    // VelocityCost
    bp::class_<VelocityCost, bp::bases<AbstractCost>,
               std::shared_ptr<VelocityCost>>("VelocityCost",
        bp::init<bp::optional<double>>((bp::arg("weight") = 1.0)))
        .def(bp::init<const VectorXd&, double>(
            (bp::arg("v_ref"), bp::arg("weight") = 1.0)));
    bp::implicitly_convertible<std::shared_ptr<VelocityCost>,
                               std::shared_ptr<AbstractCost>>();

    // AccelerationCost
    bp::class_<AccelerationCost, bp::bases<AbstractCost>,
               std::shared_ptr<AccelerationCost>>("AccelerationCost",
        bp::init<bp::optional<double>>((bp::arg("weight") = 1.0)))
        .def(bp::init<const VectorXd&, double>(
            (bp::arg("v_ref"), bp::arg("weight") = 1.0)));
    bp::implicitly_convertible<std::shared_ptr<AccelerationCost>,
                               std::shared_ptr<AbstractCost>>();

    // FrameTranslationCost
    bp::class_<FrameTranslationCost, bp::bases<AbstractCost>,
               std::shared_ptr<FrameTranslationCost>>("FrameTranslationCost",
        bp::init<const std::string&, const Eigen::Vector3d&, double>(
            (bp::arg("frame_name"),
             bp::arg("p_ref") = Eigen::Vector3d(Eigen::Vector3d::Zero()),
             bp::arg("weight") = 1.0)))
        .def(bp::init<const std::string&, const Eigen::Vector3d&, const MatrixXd&>(
            (bp::arg("frame_name"), bp::arg("p_ref"), bp::arg("weight"))))
        .def("set_ref", &FrameTranslationCost::set_ref)
        .def("get_ref", &FrameTranslationCost::get_ref,
             bp::return_value_policy<bp::copy_const_reference>());
    bp::implicitly_convertible<std::shared_ptr<FrameTranslationCost>,
                               std::shared_ptr<AbstractCost>>();

    // FrameOrientationCost
    bp::class_<FrameOrientationCost, bp::bases<AbstractCost>,
               std::shared_ptr<FrameOrientationCost>>("FrameOrientationCost",
        bp::init<const std::string&, const Eigen::Matrix3d&, double>(
            (bp::arg("frame_name"),
             bp::arg("R_ref") = Eigen::Matrix3d(Eigen::Matrix3d::Identity()),
             bp::arg("weight") = 1.0)))
        .def(bp::init<const std::string&, const Eigen::Matrix3d&, const MatrixXd&>(
            (bp::arg("frame_name"), bp::arg("R_ref"), bp::arg("weight"))))
        .def("set_ref", &FrameOrientationCost::set_ref)
        .def("get_ref", &FrameOrientationCost::get_ref,
             bp::return_value_policy<bp::copy_const_reference>());
    bp::implicitly_convertible<std::shared_ptr<FrameOrientationCost>,
                               std::shared_ptr<AbstractCost>>();

    // FrameVelocityCost
    bp::class_<FrameVelocityCost, bp::bases<AbstractCost>,
               std::shared_ptr<FrameVelocityCost>>("FrameVelocityCost",
        bp::init<const std::string&, const Vector6d&, double>(
            (bp::arg("frame_name"),
             bp::arg("v_ref") = Vector6d(Vector6d::Zero()),
             bp::arg("weight") = 1.0)))
        .def(bp::init<const std::string&, const Vector6d&, const MatrixXd&>(
            (bp::arg("frame_name"), bp::arg("v_ref"), bp::arg("weight"))))
        .def("set_ref", &FrameVelocityCost::set_ref)
        .def("get_ref", &FrameVelocityCost::get_ref,
             bp::return_value_policy<bp::copy_const_reference>())
        .def("set_re_reference_frame", &FrameVelocityCost::set_re_reference_frame)
        .def("set_base_frame_name", &FrameVelocityCost::set_base_frame_name);
    bp::implicitly_convertible<std::shared_ptr<FrameVelocityCost>,
                               std::shared_ptr<AbstractCost>>();

    // FrameAccelerationCost
    bp::class_<FrameAccelerationCost, bp::bases<AbstractCost>,
               std::shared_ptr<FrameAccelerationCost>>("FrameAccelerationCost",
                                                   bp::init<const std::string&, const Vector6d&, double>(
                                                       (bp::arg("frame_name"),
                                                        bp::arg("a_ref") = Vector6d(Vector6d::Zero()),
                                                        bp::arg("weight") = 1.0)))
        .def(bp::init<const std::string&, const Vector6d&, const MatrixXd&>(
            (bp::arg("frame_name"), bp::arg("a_ref"), bp::arg("weight"))))
        .def("set_ref", &FrameAccelerationCost::set_ref)
        .def("get_ref", &FrameAccelerationCost::get_ref,
             bp::return_value_policy<bp::copy_const_reference>())
        .def("set_re_reference_frame", &FrameAccelerationCost::set_re_reference_frame)
        .def("set_base_frame_name", &FrameAccelerationCost::set_base_frame_name);
    bp::implicitly_convertible<std::shared_ptr<FrameAccelerationCost>,
                               std::shared_ptr<AbstractCost>>();

    // ── Constraints ────────────────────────────────────────────────────────

    // EulerIntegration
    bp::class_<EulerIntegration, bp::bases<AbstractConstraint>,
               std::shared_ptr<EulerIntegration>>("EulerIntegration",
        bp::init<double>((bp::arg("dt"))));
    bp::implicitly_convertible<std::shared_ptr<EulerIntegration>,
                               std::shared_ptr<AbstractConstraint>>();

    // SemiEulerIntegration
    bp::class_<SemiEulerIntegration, bp::bases<AbstractConstraint>,
               std::shared_ptr<SemiEulerIntegration>>("SemiEulerIntegration",
        bp::init<double>((bp::arg("dt"))));
    bp::implicitly_convertible<std::shared_ptr<SemiEulerIntegration>,
                               std::shared_ptr<AbstractConstraint>>();

    // InvDynamics
    bp::class_<InvDynamics, bp::bases<AbstractConstraint>,
               std::shared_ptr<InvDynamics>>("InvDynamics",
        bp::init<>());
    bp::implicitly_convertible<std::shared_ptr<InvDynamics>,
                               std::shared_ptr<AbstractConstraint>>();

    // JointLimitsConstraint
    bp::class_<JointLimitsConstraint, bp::bases<AbstractConstraint>,
               std::shared_ptr<JointLimitsConstraint>>("JointLimitsConstraint",
        bp::init<>());
    bp::implicitly_convertible<std::shared_ptr<JointLimitsConstraint>,
                               std::shared_ptr<AbstractConstraint>>();

    // ContactConstraint
    bp::class_<ContactConstraint, bp::bases<AbstractConstraint>,
               std::shared_ptr<ContactConstraint>>("ContactConstraint",
        bp::init<const std::string&, double>(
            (bp::arg("frame_name"), bp::arg("ground_height") = 0.0)))
        .def("set_ground_height", &ContactConstraint::set_ground_height);
    bp::implicitly_convertible<std::shared_ptr<ContactConstraint>,
                               std::shared_ptr<AbstractConstraint>>();

    // FrictionConeConstraint
    bp::class_<FrictionConeConstraint, bp::bases<AbstractConstraint>,
               std::shared_ptr<FrictionConeConstraint>>("FrictionConeConstraint",
        bp::init<const std::string&, double>(
            (bp::arg("frame_name"), bp::arg("mu") = 0.5)))
        .def("set_friction_coefficient", &FrictionConeConstraint::set_friction_coefficient);
    bp::implicitly_convertible<std::shared_ptr<FrictionConeConstraint>,
                               std::shared_ptr<AbstractConstraint>>();

    // FrameTranslationConstraint
    bp::class_<FrameTranslationConstraint, bp::bases<AbstractConstraint>,
               std::shared_ptr<FrameTranslationConstraint>>("FrameTranslationConstraint",
        bp::init<const std::string&, const Eigen::Vector3d&>(
            (bp::arg("frame_name"), bp::arg("p_ref") = Eigen::Vector3d(Eigen::Vector3d::Zero()))))
        .def("set_ref", &FrameTranslationConstraint::set_ref)
        .def("get_ref", &FrameTranslationConstraint::get_ref,
             bp::return_value_policy<bp::copy_const_reference>());
    bp::implicitly_convertible<std::shared_ptr<FrameTranslationConstraint>,
                               std::shared_ptr<AbstractConstraint>>();

    // FrameOrientationConstraint
    bp::class_<FrameOrientationConstraint, bp::bases<AbstractConstraint>,
               std::shared_ptr<FrameOrientationConstraint>>("FrameOrientationConstraint",
                                                            bp::init<const std::string&, const Eigen::Matrix3d&>(
                                                                (bp::arg("frame_name"), bp::arg("R_ref") = Eigen::Matrix3d(Eigen::Matrix3d::Zero()))))
        .def("set_ref", &FrameOrientationConstraint::set_ref)
        .def("get_ref", &FrameOrientationConstraint::get_ref,
             bp::return_value_policy<bp::copy_const_reference>());
    bp::implicitly_convertible<std::shared_ptr<FrameOrientationConstraint>,
                               std::shared_ptr<AbstractConstraint>>();

    // FrameVelocityConstraint
    bp::class_<FrameVelocityConstraint, bp::bases<AbstractConstraint>,
            std::shared_ptr<FrameVelocityConstraint>>("FrameVelocityConstraint",
        bp::init<const std::string&, const Vector6d&>(
            (bp::arg("frame_name"), bp::arg("v_ref") = Vector6d(Vector6d::Zero()))))
        .def("set_ref", &FrameVelocityConstraint::set_ref)
        .def("get_ref", &FrameVelocityConstraint::get_ref,
            bp::return_value_policy<bp::copy_const_reference>())
        .def("set_re_reference_frame", &FrameVelocityConstraint::set_re_reference_frame)
        .def("set_base_frame_name", &FrameVelocityConstraint::set_base_frame_name);
    bp::implicitly_convertible<std::shared_ptr<FrameVelocityConstraint>,
                               std::shared_ptr<AbstractConstraint>>();

    // ── ContactScheduler ───────────────────────────────────────────────────
    bp::class_<ContactScheduler>("ContactScheduler")
        .def(bp::init<>())
        .def("define_contact", &ContactScheduler::define_contact,
             (bp::arg("contact_name"), bp::arg("contact_frame_names")))
        .def("addPhase", &ContactScheduler::addPhase,
             (bp::arg("contacts_list"), bp::arg("duration"),
              bp::arg("sequence_name") = "_"))
        .def("getSequence", &ContactScheduler::getSequence,
             (bp::arg("sampling_rate"),
              bp::arg("sequence_name") = "_",
              bp::arg("nodes_number") = -1,
              bp::arg("current_time") = 0.0));

    // ── Node ───────────────────────────────────────────────────────────────
    bp::class_<Node, boost::noncopyable>("Node", bp::init<pinocchio::Model>())
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
        // Zero-copy numpy views via thin wrappers
        .def("q",   &node_q)
        .def("v",   &node_v)
        .def("a",   &node_a)
        .def("x",   &node_x)
        .def("u",   &node_u)
        .def("fc",  &node_fc)
        .def("tau", &node_tau);

    // ── OCP ────────────────────────────────────────────────────────────────
    bp::class_<OCP, boost::noncopyable>("OCP", bp::init<int>())
        .def("addNode",              &ocp_add_node)
        .def("finalize",             &OCP::finalize)
        .def("num_nodes",            &OCP::num_nodes)
        .def("get_node",             (Node& (OCP::*)(int)) &OCP::get_node,
             bp::return_internal_reference<>())
        .def("cost",                 &OCP::cost)
        .def("dynamics_defect",      &OCP::dynamics_defect)
        .def("constraint_violation", &OCP::constraint_violation)
        .def("x_traj",               &ocp_x_traj)
        .def("u_traj",               &ocp_u_traj)
        .def("save_trajectory",      &OCP::save_trajectory,
             (bp::arg("filepath"), bp::arg("dt") = 0.0, bp::arg("urdf_path") = ""));

    // ── SQPSolver ──────────────────────────────────────────────────────────
    bp::class_<SQPSolver, boost::noncopyable>("SQPSolver", bp::init<OCP&>())
        .def("set_options", &SQPSolver::set_options)
        .def("solve",       &SQPSolver::solve)
        .def("get_stats",   &SQPSolver::get_stats,
             bp::return_internal_reference<>());
}
