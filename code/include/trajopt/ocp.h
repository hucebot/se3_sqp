#pragma once

#include <trajopt/node.h>

#include <Eigen/Dense>
#include <vector>

class Node;

class OCP {
   private:
    // Horizon as a list of nodes (nodes own their state/control data)
    std::vector<Node> _horizon;

    // Problem dimensions
    int _num_nodes;

   public:
    OCP();
    ~OCP();

    // Methods to build the horizon
    void addNode(const Node& node);
    void addNode(Node&& node);  // Move version
    void setHorizon(const std::vector<Node>& nodes);

    // Node accessors
    int num_nodes() const { return _num_nodes; }
    Node& get_node(int k) { return _horizon[k]; }
    const Node& get_node(int k) const { return _horizon[k]; }

    // Pointer accessors for HPIPM (zero-copy)
    double* x_ptr(int k) { return _horizon[k].x().data(); }
    double* u_ptr(int k) { return _horizon[k].u().data(); }
};