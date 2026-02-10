#include <trajopt/ocp.h>

OCP::OCP(int num_nodes) : _num_nodes(num_nodes) {
    _horizon.reserve(num_nodes);
    _x_traj.reserve(num_nodes);
    _u_traj.reserve(num_nodes);
}

OCP::~OCP() {}

void OCP::addNode(const Node& node) {
    _horizon.push_back(node);
    _num_nodes = _horizon.size();
    // Relink all nodes and rebind constraints (vector may have reallocated)
    for (int i = 0; i < _num_nodes; ++i) {
        _horizon[i].rebind_constraints();
        if (i < _num_nodes - 1) {
            _horizon[i].next_node = &_horizon[i + 1];
        }
    }
}

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

void OCP::setHorizon(const std::vector<Node>& nodes) {
    _horizon = nodes;
    _num_nodes = _horizon.size();
    // Link all nodes and rebind constraints
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
        _horizon[k].bind_trajectory(&x[k], &u[k]);
    }
}
