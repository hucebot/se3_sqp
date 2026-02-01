#pragma once

#include <trajopt/node.h>

#include <Eigen/Dense>
#include <vector>

class Node;

class OCP {
   private:
    // Horizon as a list of nodes
    std::vector<Node> _horizon;

    // Problem dimensions
    int _num_nodes;

    std::vector<VectorXd> _x_traj;
    std::vector<VectorXd> _u_traj;

   public:
    OCP();
    ~OCP();

    // Methods to build the horizon
    void addNode(const Node& node);
    void setHorizon(const std::vector<Node>& nodes);

    // Getters
    int getNumNodes() const { return _num_nodes; }
    const std::vector<Node>& getHorizon() const { return _horizon; }
    Node& getNode(int index) { return _horizon[index]; }
    const Node& getNode(int index) const { return _horizon[index]; }
};