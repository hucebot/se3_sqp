#pragma once

#include <common/types.h>
#include <trajopt/constraints/abstract_constraint.h>
#include <trajopt/costs/abstract_cost.h>

#include <array>
#include <memory>
#include <nlohmann/json.hpp>
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/joint-configuration.hpp>
#include <pinocchio/algorithm/kinematics.hpp>
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
   public:
    // Contact representation for external forces
    struct Contact {
        std::string name;         // frame name (for user API)
        int frame_id;             // pinocchio frame index
        int parent_joint_id;      // parent joint index (cached for efficiency)
        bool active = true;       // toggle for MPC
    };

   private:
    // Placeholder for dynamics constraint
    std::shared_ptr<AbstractConstraint> _dynamics;

    // Constraints and costs
    std::vector<std::shared_ptr<AbstractCost>> _cost_list;
    std::vector<std::shared_ptr<AbstractConstraint>> _constraint_list;

    // Contact configuration
    std::vector<Contact> _contacts;                             // maintains registration order (for Jacobian indexing)
    std::unordered_map<std::string, int> _contact_name_to_idx;  // fast O(1) lookup by name

    // Pointers to trajectory data (owned by OCP)
    VectorXd* _x_ptr = nullptr;
    VectorXd* _u_ptr = nullptr;

    // Pointer to inverse dynamics torque value (linked by InvDynamics constraint)
    VectorXd* _tau_ptr = nullptr;

    double _cost;
    double _defect;
    double _violation;

    // Gradients of merit terms w.r.t. tangent state (ndx) and control (ndu)
    VectorXd _grad_cost_x;
    VectorXd _grad_cost_u;
    VectorXd _grad_defect_x;
    VectorXd _grad_defect_u;
    VectorXd _grad_violation_x;
    VectorXd _grad_violation_u;

    // Scratch buffer for violation computation (avoids per-call heap allocations)
    VectorXd _viol_tmp;

    // Cache flags for Pinocchio computations (dirty-flag pattern)
    enum CacheFlag : uint8_t {
        CACHE_FK               = 1 << 0,  // forwardKinematics(q)
        CACHE_FRAME_PLACEMENTS = 1 << 1,  // updateFramePlacements
    };
    uint8_t _cache_flags = 0;

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
    auto a() { return _u_ptr->head(_nv); }
    auto q() const { return _x_ptr->head(_nq); }
    auto v() const { return _x_ptr->tail(_nv); }
    auto a() const { return _u_ptr->head(_nv); }

    // Contact force accessors (i-th contact force in u)
    auto fc(int i) { return _u_ptr->segment(_nv + 3*i, 3); }
    auto fc(int i) const { return _u_ptr->segment(_nv + 3*i, 3); }
    auto fc() { return _u_ptr->tail(3 * n_contacts()); }  // all forces stacked
    auto fc() const { return _u_ptr->tail(3 * n_contacts()); }

    // Torque accessor (returns inverse dynamics constraint value)
    VectorXdConstRef tau() const {
        if (!_tau_ptr) {
            throw std::runtime_error("No inverse dynamics constraint linked. Call add_constraint(std::make_shared<InvDynamics>()) first.");
        }
        return *_tau_ptr;
    }

    // Allow InvDynamics constraint to link its _value vector
    void link_tau(VectorXd* tau_ptr) { _tau_ptr = tau_ptr; }

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
    int nu() const { return _nv + 3 * n_contacts(); }
    int ndu() const { return _nv + 3 * n_contacts(); }  // forces are Euclidean

    // Model access
    pinocchio::Model& model() { return *_model_ptr; }
    const pinocchio::Model& model() const { return *_model_ptr; }

    pinocchio::Data& data() {return *_data_ptr;}

    void x_oplus(VectorXdRef x0, VectorXdRef dx, VectorXdRef x1);
    void u_oplus(VectorXdRef u0, VectorXdRef du, VectorXdRef u1);

    void invalidate_cache();
    void require_forward_kinematics();
    void require_frame_placements();

    void add_cost(std::shared_ptr<AbstractCost> cost);
    void add_dynamics(std::shared_ptr<AbstractConstraint> dynamics);
    void add_constraint(std::shared_ptr<AbstractConstraint> constraint);

    // Contact management
    void add_contact(const std::string& frame_name);
    void add_contacts(const std::vector<std::string>& frame_names);
    void set_active_contacts(const std::vector<std::string>& names);
    int n_contacts() const { return static_cast<int>(_contacts.size()); }
    const std::vector<Contact>& contacts() const { return _contacts; }

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

    void calc_cost();
    void calc_dynamics_defect();
    void calc_constraint_violation();

    /**
     * Compute subgradients of each merit term w.r.t. tangent state and control.
     *
     * Requires evaluate() and jacobian() to have been called on all
     * costs/constraints first.
     *
     * Gradient dimensions:
     *   x component: ndx = 2*nv  (tangent space of the state manifold)
     *   u component: ndu = nv
     *
     * For defect and violation the functions are non-smooth (L∞ norms /
     * max-based), so a subgradient is returned selecting the max-active component.
     */
    void calc_cost_gradient();
    void calc_defect_gradient();
    void calc_violation_gradient();

    const VectorXd& get_cost_grad_x() const { return _grad_cost_x; }
    const VectorXd& get_cost_grad_u() const { return _grad_cost_u; }
    const VectorXd& get_defect_grad_x() const { return _grad_defect_x; }
    const VectorXd& get_defect_grad_u() const { return _grad_defect_u; }
    const VectorXd& get_violation_grad_x() const { return _grad_violation_x; }
    const VectorXd& get_violation_grad_u() const { return _grad_violation_u; }

    const double get_cost(){ return _cost;}
    const double get_dynamics_defect() { return _defect;}
    const double get_constraint_violation(){ return _violation;}
    // Serialize this node's state and control to JSON
    nlohmann::json to_json() const;
};
