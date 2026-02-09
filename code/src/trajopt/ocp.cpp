#include <trajopt/ocp.h>

OCP::OCP() : _num_nodes(0) {}

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
