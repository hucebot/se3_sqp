#pragma once

#include <trajopt/node.h>
#include <memory>
#include <vector>

/**
 * NodeFactory provides factory methods for creating standard node configurations.
 *
 * This factory encapsulates common patterns for node creation, making it easier
 * to build trajectory optimization problems with consistent node structures.
 */
class NodeFactory {
   public:
    /**
     * Create a standard dynamics node with state and control variables.
     *
     * This creates a node with:
     * - State variables (nx)
     * - Control variables (nu)
     * - Optional parameter variables (np)
     *
     * @param nx State dimension
     * @param nu Control dimension
     * @param np Parameter dimension (default: 0)
     * @return Node configured with the specified dimensions
     */
    static Node create_dynamics_node(int nx, int nu, int np = 0) {
        int nv = nx + nu + np;
        Node node(nv);
        node.set_dimensions(nx, nu, np);
        return node;
    }

    /**
     * Create a terminal node (final state, no controls).
     *
     * Terminal nodes typically only have state variables and no controls,
     * representing the final state in the optimization horizon.
     *
     * @param nx State dimension
     * @param np Parameter dimension (default: 0)
     * @return Node configured as a terminal node
     */
    static Node create_terminal_node(int nx, int np = 0) {
        int nv = nx + np;
        Node node(nv);
        node.set_dimensions(nx, 0, np);
        return node;
    }

    /**
     * Create an initial node with fixed initial state.
     *
     * Initial nodes represent the starting state of the trajectory.
     * They typically have state and control variables.
     *
     * @param nx State dimension
     * @param nu Control dimension
     * @param np Parameter dimension (default: 0)
     * @return Node configured as an initial node
     */
    static Node create_initial_node(int nx, int nu, int np = 0) {
        // Initial nodes are the same as dynamics nodes in structure
        return create_dynamics_node(nx, nu, np);
    }

    /**
     * Create a custom node with specified variable count.
     *
     * @param nv Total number of variables
     * @param nx State dimension
     * @param nu Control dimension
     * @param np Parameter dimension
     * @return Node with custom configuration
     */
    static Node create_custom_node(int nv, int nx, int nu, int np) {
        Node node(nv);
        node.set_dimensions(nx, nu, np);
        return node;
    }

    /**
     * Create a sequence of identical dynamics nodes.
     *
     * Useful for creating a horizon of nodes with the same structure.
     *
     * @param num_nodes Number of nodes to create
     * @param nx State dimension
     * @param nu Control dimension
     * @param np Parameter dimension (default: 0)
     * @return Vector of nodes
     */
    static std::vector<Node> create_node_sequence(int num_nodes, int nx, int nu,
                                                   int np = 0) {
        std::vector<Node> nodes;
        nodes.reserve(num_nodes);
        for (int i = 0; i < num_nodes; i++) {
            nodes.push_back(create_dynamics_node(nx, nu, np));
        }
        return nodes;
    }

    /**
     * Create a complete horizon with N dynamics nodes + 1 terminal node.
     *
     * This is a common pattern in trajectory optimization where you have
     * N stages with dynamics and controls, followed by a final terminal state.
     *
     * @param N Number of dynamics nodes
     * @param nx State dimension
     * @param nu Control dimension
     * @param np Parameter dimension (default: 0)
     * @return Vector of N+1 nodes (N dynamics + 1 terminal)
     */
    static std::vector<Node> create_horizon(int N, int nx, int nu, int np = 0) {
        std::vector<Node> nodes;
        nodes.reserve(N + 1);

        // Create N dynamics nodes
        for (int i = 0; i < N; i++) {
            nodes.push_back(create_dynamics_node(nx, nu, np));
        }

        // Add terminal node
        nodes.push_back(create_terminal_node(nx, np));

        return nodes;
    }
};
