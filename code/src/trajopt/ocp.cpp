#include <trajopt/ocp.h>

#include <fstream>
#include <iostream>

OCP::OCP(int num_nodes) : _num_nodes(num_nodes) {
    _horizon.reserve(num_nodes);
    _x_traj.reserve(num_nodes);
    _u_traj.reserve(num_nodes);
}

OCP::~OCP() {}

void OCP::addNode(Node&& node) {
    _horizon.push_back(std::move(node));
    _num_nodes = _horizon.size();
    // Relink all nodes and rebind constraints (vector may have reallocated)
    for (int i = 0; i < _num_nodes; ++i) {
        _horizon[i].rebind_constraints();
        if (i < _num_nodes - 1) {
            _horizon[i].next_node = &_horizon[i + 1];
        }
    }
}

void OCP::finalize() {
    _x_traj.resize(_num_nodes);
    _u_traj.resize(_num_nodes);

    for (int k = 0; k < _num_nodes; ++k) {
        _x_traj[k].setZero(_horizon[k].nx());
        _u_traj[k].setZero(_horizon[k].nu());
    }

    // Bind nodes to OCP-owned trajectory
    bind_trajectory(_x_traj, _u_traj);
}

void OCP::bind_trajectory(std::vector<VectorXd>& x, std::vector<VectorXd>& u) {
    for (int k = 0; k < _num_nodes; ++k) {
        VectorXd* u_ptr = (k < static_cast<int>(u.size())) ? &u[k] : nullptr;
        _horizon[k].bind_trajectory(&x[k], u_ptr);
    }
}


void OCP::evaluate_all() {
    for (auto& node : _horizon) {
        if (node.get_dynamics()) node.get_dynamics()->evaluate();
        for (auto& c : node.get_costs())       c->evaluate();
        for (auto& c : node.get_constraints()) c->evaluate();
    }
}

double OCP::cost(){
    _total_cost = 0.;
    for (auto& node: _horizon){
        node.calc_cost();
        _total_cost += node.get_cost();
    }
    return _total_cost;
}

double OCP::dynamics_defect(){
    _total_dynamics_defect = 0.;
    for (auto& node: _horizon){
        node.calc_dynamics_defect();
        _total_dynamics_defect += node.get_dynamics_defect();
    }
    return _total_dynamics_defect;
}

double OCP::constraint_violation(){
    _total_constraint_violation = 0.;
    for (auto& node: _horizon){
        node.calc_constraint_violation();
        _total_constraint_violation += node.get_constraint_violation();
    }
    return _total_constraint_violation;
}


void OCP::save_trajectory(const std::string& filepath, double dt,
                           const std::string& urdf_path) const {
    const Node& n0 = _horizon[0];
    nlohmann::json j;
    j["N"]         = _num_nodes;
    j["dt"]        = dt;
    j["nq"]        = n0.nq();
    j["nv"]        = n0.nv();
    j["nu"]        = n0.nu();
    j["urdf_path"] = urdf_path;

    nlohmann::json traj = nlohmann::json::array();
    for (const Node& node : _horizon) {
        traj.push_back(node.to_json());
    }
    j["trajectory"] = traj;

    std::ofstream out(filepath);
    out << j.dump(2);
    std::cout << "Trajectory saved to " << filepath << std::endl;
}
