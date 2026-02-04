#pragma once

#include <common/types.h>
#include <common/var_slice.h>
#include <trajopt/constraints/abstract_constraint.h>
#include <trajopt/costs/abstract_cost.h>

#include <array>
#include <memory>
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

    // State and control vectors
    VectorXd _x;   // State: [q, v] contiguous
    VectorXd _u;   // Control input

   public:
    Node(pinocchio::Model mdl);
    ~Node() = default;

    std::shared_ptr<Node> next_node;

    std::shared_ptr<pinocchio::Model> _model_ptr;
    pinocchio::Data _data;

    int _nq;
    int _nv;

    // Zero-copy segment accessors for state components
    auto q() { return _x.head(_nq); }
    auto v() { return _x.tail(_nv); }
    auto q() const { return _x.head(_nq); }
    auto v() const { return _x.tail(_nv); }

    // Full state/control accessors
    VectorXdRef x() { return _x; }
    VectorXdRef u() { return _u; }
    VectorXdConstRef x() const { return _x; }
    VectorXdConstRef u() const { return _u; }

    // Dimensions
    int nq() const { return _nq; }
    int nv() const { return _nv; }
    int nx() const { return _nq + _nv; }
    int nu() const { return _u.size(); }

    // Model access
    pinocchio::Model& model() { return *_model_ptr; }
    const pinocchio::Model& model() const { return *_model_ptr; }

    void cached_update();

    void add_cost(std::shared_ptr<AbstractCost> cost);
    void add_constraint(std::shared_ptr<AbstractConstraint> constraint);

    // Access to constraint and cost lists
    const std::vector<std::shared_ptr<AbstractCost>>& get_costs() const {
        return _cost_list;
    }
    const std::vector<std::shared_ptr<AbstractConstraint>>& get_constraints() const {
        return _constraint_list;
    }
};
