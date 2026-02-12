#pragma once

#include <common/types.h>
#include <trajopt/constraints/abstract_constraint.h>
#include <trajopt/costs/abstract_cost.h>

#include <array>
#include <memory>
#include <pinocchio/algorithm/joint-configuration.hpp>
#include <pinocchio/multibody/data.hpp>
#include <pinocchio/multibody/model.hpp>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

/**
 * Node represents a single stage in the trajectory optimization problem.
 * It manages optimization variables, constraints, and costs.
 *
 * The Node uses a registry pattern for slice allocation, allowing
 * constraints to register their variable and constraint slices without
 * the Node needing to know constraint-specific details.
 */
class Node {
   private:
    // Placeholder for dynamics constraint
    std::shared_ptr<AbstractConstraint> _dynamics;

    // Constraints and costs
    std::vector<std::shared_ptr<AbstractCost>> _cost_list;
    std::vector<std::shared_ptr<AbstractConstraint>> _constraint_list;

    // Pointers to trajectory data (owned by OCP)
    VectorXd* _x_ptr = nullptr;
    VectorXd* _u_ptr = nullptr;

   public:
    Node(pinocchio::Model mdl);
    Node(Node&&) = default;
    Node& operator=(Node&&) = default;
    ~Node() = default;

    Node* next_node = nullptr;

    std::shared_ptr<pinocchio::Model> _model_ptr;
    std::unique_ptr<pinocchio::Data> _data_ptr;

    int _nq;
    int _nv;

    // Bind node to trajectory data
    void bind_trajectory(VectorXd* x, VectorXd* u);

    // Zero-copy segment accessors for state components
    auto q() { return _x_ptr->head(_nq); }
    auto v() { return _x_ptr->tail(_nv); }
    auto q() const { return _x_ptr->head(_nq); }
    auto v() const { return _x_ptr->tail(_nv); }

    // Full state/control accessors
    VectorXdRef x() { return *_x_ptr; }
    VectorXdRef u() { return *_u_ptr; }
    VectorXdConstRef x() const { return *_x_ptr; }
    VectorXdConstRef u() const { return *_u_ptr; }

    // Dimensions
    int nq() const { return _nq; }
    int nv() const { return _nv; }
    int nx() const { return _nq + _nv; }
    int ndx() const { return _nv + _nv; }
    int nu() const { return _nv; }
    int ndu() const { return _nv; }

    // Model access
    pinocchio::Model& model() { return *_model_ptr; }
    const pinocchio::Model& model() const { return *_model_ptr; }

    pinocchio::Data& data() {return *_data_ptr;}

    void x_oplus(VectorXdRef x0, VectorXdRef dx, VectorXdRef x1);
    // void u_oplus(VectorXdRef x0, VectorXdRef dx, VectorXdRef x1);

    void cached_update();

    void add_cost(std::shared_ptr<AbstractCost> cost);
    void add_dynamics(std::shared_ptr<AbstractConstraint> dynamics);
    void add_constraint(std::shared_ptr<AbstractConstraint> constraint);

    // Access to constraint and cost lists
    const std::vector<std::shared_ptr<AbstractCost>>& get_costs() const {
        return _cost_list;
    }
    const std::vector<std::shared_ptr<AbstractConstraint>>& get_constraints() const {
        return _constraint_list;
    }
    const std::shared_ptr<AbstractConstraint>& get_dynamics() const {
        return _dynamics;
    }

    // Update all constraints' node pointers to point to this node
    // (needed after node is copied into a container)
    void rebind_constraints();
};
