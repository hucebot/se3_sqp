#pragma once

#include <trajopt/node.h>

#include <Eigen/Dense>
#include <vector>

class OCP {
   private:
    // Horizon as a list of nodes
    std::vector<Node> horizon;

    // Problem dimensions
    int num_nodes;

    std::vector<Eigen::VectorXd> x_trajectory;
    std::vector<Eigen::VectorXd> u_trajectory;

   public:
    OCP();
    ~OCP();

    // Methods to build the horizon
    void add_node(const Node& node);
    void set_horizon(const std::vector<Node>& nodes);

    // Getters
    int get_num_nodes() const { return num_nodes; }
    const std::vector<Node>& get_horizon() const { return horizon; }
    Node& get_node(int index) { return horizon[index]; }
    const Node& get_node(int index) const { return horizon[index]; }
};

OCP::OCP() : num_nodes(0) {}

OCP::~OCP() {}

void OCP::add_node(const Node& node) {
    horizon.push_back(node);
    num_nodes = horizon.size();
}

void OCP::set_horizon(const std::vector<Node>& nodes) {
    horizon = nodes;
    num_nodes = horizon.size();
}
