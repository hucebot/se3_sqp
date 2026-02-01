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
 * NodeSliceRegistry manages variable and constraint slice allocation
 * for a single node in the trajectory optimization problem.
 *
 * Features:
 * - Deduplication: Multiple functions requesting the same variable type
 *   get the same slice (no double allocation)
 * - O(1) lookup for standard variable types (CONF, VEL, ACC, TORQ)
 * - Named forces for contact dynamics (stored in map)
 * - Runtime query API for accessing slices after allocation
 */
class NodeSliceRegistry {
   private:
    // Standard variable slices - O(1) lookup via array indexing
    std::array<VarSlice, static_cast<size_t>(VarType::_COUNT)> _var_slices{};
    std::array<bool, static_cast<size_t>(VarType::_COUNT)> _var_allocated{};

    // Named contact forces (keyed by contact name, e.g., "foot_left")
    std::unordered_map<std::string, VarSlice> _forces;

    // Constraint slices (keyed by constraint name)
    std::unordered_map<std::string, VarSlice> _constraints;

    int _var_offset = 0;  // Running variable offset
    int _con_offset = 0;  // Running constraint offset

   public:
    NodeSliceRegistry() = default;

    // =========================================================================
    // Allocation API (deduplicates standard types)
    // =========================================================================

    /**
     * Allocate or retrieve a standard variable slice.
     * If already allocated with the same size, returns existing slice.
     * If already allocated with different size, throws.
     *
     * @param type Variable type (CONF, VEL, ACC, TORQ)
     * @param size Number of elements
     * @return Reference to the allocated slice
     */
    const VarSlice& allocate_var(VarType type, int size) {
        if (type == VarType::FORCE || type == VarType::_COUNT) {
            throw std::invalid_argument(
                "Use allocate_force() for FORCE type, not allocate_var()");
        }

        size_t idx = static_cast<size_t>(type);
        if (_var_allocated[idx]) {
            // Already allocated - verify size matches
            if (_var_slices[idx].size != size) {
                throw std::runtime_error(
                    "Variable type already allocated with different size");
            }
            return _var_slices[idx];
        }

        // New allocation
        _var_slices[idx] = VarSlice(_var_offset, size);
        _var_allocated[idx] = true;
        _var_offset += size;
        return _var_slices[idx];
    }

    /**
     * Allocate a named contact force slice.
     * Each contact name gets its own slice (no deduplication across names).
     *
     * @param name Contact identifier (e.g., "foot_left", "hand_right")
     * @param size Force dimension (typically 3 for 3D force, 6 for wrench)
     * @return Reference to the allocated slice
     */
    const VarSlice& allocate_force(const std::string& name, int size) {
        auto it = _forces.find(name);
        if (it != _forces.end()) {
            if (it->second.size != size) {
                throw std::runtime_error(
                    "Force '" + name + "' already allocated with different size");
            }
            return it->second;
        }

        VarSlice slice(_var_offset, size);
        _var_offset += size;
        auto result = _forces.emplace(name, slice);
        return result.first->second;
    }

    /**
     * Allocate a constraint slice.
     *
     * @param name Constraint identifier
     * @param size Number of constraint rows
     * @return Reference to the allocated slice
     */
    const VarSlice& allocate_constraint(const std::string& name, int size) {
        auto it = _constraints.find(name);
        if (it != _constraints.end()) {
            if (it->second.size != size) {
                throw std::runtime_error(
                    "Constraint '" + name + "' already allocated with different size");
            }
            return it->second;
        }

        VarSlice slice(_con_offset, size);
        _con_offset += size;
        auto result = _constraints.emplace(name, slice);
        return result.first->second;
    }

    // =========================================================================
    // Query API (fast, O(1) for standard types)
    // =========================================================================

    /**
     * Get slice for a standard variable type.
     * @throws std::runtime_error if not allocated
     */
    const VarSlice& get_var(VarType type) const {
        if (type == VarType::FORCE || type == VarType::_COUNT) {
            throw std::invalid_argument("Use get_force() for FORCE type");
        }

        size_t idx = static_cast<size_t>(type);
        if (!_var_allocated[idx]) {
            throw std::runtime_error("Variable type not allocated");
        }
        return _var_slices[idx];
    }

    /**
     * Get slice for a named force.
     * @throws std::runtime_error if not found
     */
    const VarSlice& get_force(const std::string& name) const {
        auto it = _forces.find(name);
        if (it == _forces.end()) {
            throw std::runtime_error("Force '" + name + "' not allocated");
        }
        return it->second;
    }

    /**
     * Get slice for a custom variable.
     * @throws std::runtime_error if not found
     */
    const VarSlice& get_custom_var(const std::string& name) const {
        auto it = _custom_vars.find(name);
        if (it == _custom_vars.end()) {
            throw std::runtime_error("Custom var '" + name + "' not allocated");
        }
        return it->second;
    }

    /**
     * Get slice for a constraint.
     * @throws std::runtime_error if not found
     */
    const VarSlice& get_constraint(const std::string& name) const {
        auto it = _constraints.find(name);
        if (it == _constraints.end()) {
            throw std::runtime_error("Constraint '" + name + "' not allocated");
        }
        return it->second;
    }

    // =========================================================================
    // Existence checks
    // =========================================================================

    bool has_var(VarType type) const {
        if (type == VarType::FORCE || type == VarType::_COUNT) {
            return false;
        }
        return _var_allocated[static_cast<size_t>(type)];
    }

    bool has_force(const std::string& name) const {
        return _forces.find(name) != _forces.end();
    }

    bool has_custom_var(const std::string& name) const {
        return _custom_vars.find(name) != _custom_vars.end();
    }

    bool has_constraint(const std::string& name) const {
        return _constraints.find(name) != _constraints.end();
    }

    // =========================================================================
    // Dimension queries
    // =========================================================================

    int total_var_dim() const { return _var_offset; }
    int total_con_dim() const { return _con_offset; }

    /**
     * Get dimension of a specific variable type (0 if not allocated).
     */
    int get_var_dim(VarType type) const {
        if (!has_var(type)) return 0;
        return _var_slices[static_cast<size_t>(type)].size;
    }
};

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
