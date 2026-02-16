#pragma once

#include <trajopt/node.h>

#include <Eigen/Dense>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>

class Node;

class OCP {
   private:
    // Horizon as a list of nodes
    std::vector<Node> _horizon;

    // Trajectory storage (owned by OCP)
    std::vector<VectorXd> _x_traj;
    std::vector<VectorXd> _u_traj;

    // Problem dimensions
    int _num_nodes;

    double _total_cost;
    double _total_dynamics_defect;
    double _total_constraint_violation;

   public:
    OCP(int num_nodes);
    ~OCP();

    // Methods to build the horizon (move-only, Node is not copyable)
    void addNode(Node&& node);

    // Node accessors
    int num_nodes() const { return _num_nodes; }
    Node& get_node(int k) { return _horizon[k]; }
    const Node& get_node(int k) const { return _horizon[k]; }

    // Pointer accessors for HPIPM (zero-copy)
    double* x_ptr(int k) { return _horizon[k].x().data(); }
    double* u_ptr(int k) { return _horizon[k].u().data(); }

    // Allocate trajectories and bind nodes (call after adding all nodes)
    void finalize();

    // Bind nodes to external trajectory (for SQP line search)
    void bind_trajectory(std::vector<VectorXd>& x, std::vector<VectorXd>& u);

    // Trigger evaluate() on all functions at the current node binding.
    // Must be called after bind_trajectory() to refresh _value for LS checks.
    void evaluate_all();

    double cost();
    double dynamics_defect();
    double constraint_violation();

    // Access to OCP-owned trajectory
    std::vector<VectorXd>& x_traj() { return _x_traj; }
    std::vector<VectorXd>& u_traj() { return _u_traj; }
    const std::vector<VectorXd>& x_traj() const { return _x_traj; }
    const std::vector<VectorXd>& u_traj() const { return _u_traj; }

    // Serialize final trajectory to a JSON file
    void save_trajectory(const std::string& filepath, double dt = 0.0,
                         const std::string& urdf_path = "") const;
};