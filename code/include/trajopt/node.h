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
   protected:
    pinocchio::Data _data;

   private:
    // Placeholder for dynamics constraint
    std::shared_ptr<AbstractConstraint> _dynamics;

    // Configuration and velocity dimensions (from Pinocchio model)
    int _nq = 0;  // Configuration dimension
    int _nv = 0;  // Velocity/tangent dimension

    // Which variable type is used as control input
    VarType _control_type = VarType::ACC;  // Default: acceleration control

    // Slice registry for variable and constraint allocation
    NodeSliceRegistry _registry;

    // Constraints and costs (stored as shared pointers)
    std::vector<std::shared_ptr<AbstractCost>> _cost_list;
    std::vector<std::shared_ptr<AbstractConstraint>> _constraint_list;

   public:
    Node() = default;
    ~Node() = default;

    // Robot model (shared across nodes)
    std::shared_ptr<pinocchio::Model> _model_ptr;

    // =========================================================================
    // Dimension setup
    // =========================================================================

    /**
     * Set configuration and velocity dimensions.
     * This pre-allocates CONF and VEL slices in the registry.
     *
     * @param nq Configuration dimension (e.g., joint positions)
     * @param nv Velocity dimension (e.g., joint velocities)
     */
    void set_dimensions(int nq, int nv);

    /**
     * Set the control variable type and allocate its slice.
     *
     * @param type VarType::ACC (acceleration) or VarType::TORQ
     * @param size Control dimension (typically nv)
     */
    void set_control_type(VarType type, int size);

    int get_nq() const { return _nq; }
    int get_nv() const { return _nv; }
    VarType get_control_type() const { return _control_type; }


    NodeSliceRegistry& get_registry() { return _registry; }
    const NodeSliceRegistry& get_registry() const { return _registry; }

    // =========================================================================
    // Cost and constraint management
    // =========================================================================

    /**
     * Add a cost to the node and allocate its slices immediately.
     * The cost's allocate_slices() method is called automatically.
     *
     * @param cost Shared pointer to the cost to add
     */
    void add_cost(std::shared_ptr<AbstractCost> cost);

    /**
     * Add a constraint to the node and allocate its slices immediately.
     * The constraint's allocate_slices() method is called automatically.
     *
     * @param constraint Shared pointer to the constraint to add
     */
    void add_constraint(std::shared_ptr<AbstractConstraint> constraint);

    // Access to constraint and cost lists
    const std::vector<std::shared_ptr<AbstractCost>>& get_costs() const {
        return _cost_list;
    }
    const std::vector<std::shared_ptr<AbstractConstraint>>& get_constraints() const {
        return _constraint_list;
    }
};
