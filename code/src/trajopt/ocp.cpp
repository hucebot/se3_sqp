#include <trajopt/ocp.h>

OCP::OCP() : _num_nodes(0) {}

OCP::~OCP() {}

void OCP::addNode(const Node& node) {
    _horizon.push_back(node);
    _num_nodes = _horizon.size();
}

void OCP::addNode(Node&& node) {
    _horizon.push_back(std::move(node));
    _num_nodes = _horizon.size();
}

void OCP::setHorizon(const std::vector<Node>& nodes) {
    _horizon = nodes;
    _num_nodes = _horizon.size();
}
