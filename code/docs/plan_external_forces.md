# Plan: Add External Forces to Inverse Dynamics Constraint

## Context

The trajectory optimization currently models control as pure joint accelerations (`u = a`). To support contact-rich tasks (locomotion, manipulation), we need 3D contact forces as decision variables. The constraint `τ = RNEA(q, v, a)` with effort limits must become `τ = RNEA(q, v, a, fext)` where `fext` includes contact forces at specified frames.

**Design choices:**
- Forces are **decision variables** (optimized by the solver)
- **3D linear contact forces** (not 6D wrenches)
- **Per-node contact configuration** (different nodes can have different active contacts)
- Use **Pinocchio's `fext` parameter** directly in `rnea()` and `computeRNEADerivatives()`
- Contacts registered by **frame name** at Node level
- **Fast activate/deactivate** for MPC: pre-register all contacts at setup, set active list at runtime via `set_active_contacts(vector<string>)`
- **Fixed QP dimensions**: inactive forces are free vars with zero Jacobian columns — regularization pushes them to zero, no HPIPM box constraints needed

## Architecture Impact

The control vector expands from `u = [a]` (dim `nv`) to `u = [a; f₁; f₂; ...; f_nc]` (dim `nv + 3*nc`), where `nc` is the number of **registered** contacts (not just active ones). QP dimensions stay fixed regardless of which contacts are active/inactive.

- Active contact: force optimized freely, included in fext for RNEA
- Inactive contact: Jacobian columns zeroed, excluded from fext. Free variable with zero gradient — regularization on R diagonal pushes to zero naturally. No HPIPM box constraints needed.

### Jacobian column ordering

Registration order determines everything:
```
_jacobian layout: [∂τ/∂q (nv) | ∂τ/∂v (nv) | ∂τ/∂a (nv) | ∂τ/∂f₁ (3) | ∂τ/∂f₂ (3) | ... | ∂τ/∂f_nc (3)]
                   <-------- state (2*nv) -------> <-------------- control (nv + 3*nc) ------------->
```

Contact index `i` in `Node::_contacts` maps to:
- Force in u: `u.segment(nv + 3*i, 3)`
- Jacobian columns: `_jacobian.block(0, 2*nv + nv + 3*i, nv, 3)`

`set_active_contacts()` only flips `.active` flags — does NOT reorder the contacts vector.

---

## Codebase Context

### Current architecture
- **Node** (`include/trajopt/node.h`, `src/trajopt/node.cpp`): Manages state `x=[q;v]` and control `u=a`. Key methods: `q()`, `v()`, `a()`, `u()`, `nu()` (returns `_nv`), `ndu()` (returns `_nv`). Stores costs, constraints, dynamics. Gradient vectors `_grad_*_u` sized to `ndu()`.
- **InvDynamics** (`include/trajopt/constraints/inverse_dynamics.h`, `src/trajopt/constraints/inverse_dynamics.cpp`): Computes `τ = rnea(q,v,a)`, constrains `[-effortLimit, effortLimit]`. Jacobian: `[∂τ/∂q | ∂τ/∂v | ∂τ/∂a]` via `pinocchio::computeRNEADerivatives()`. `get_jac_x()` returns left `2*nv` cols, `get_jac_u()` returns right `nv` cols.
- **AccelerationCost** (`src/trajopt/costs/acceleration_cost.cpp`): Penalizes `u - a_ref`. Has `//TODO: change when forces` at line 33. `get_jac_u()` uses `middleCols(2*nv, nv)`.
- **OCP** (`src/trajopt/ocp.cpp`): `finalize()` allocates trajectory per-node using `node.nu()` — already supports variable sizes.
- **SQP Solver** (`src/sqp_solver/sqp_solver.cpp`): Reads per-node `nu()`/`ndu()` in `init()`. HPIPM stages initialized with per-stage dimensions. No changes needed.
- **HPIPM Wrapper** (`include/qp_solver/hpipm_solver.h`): Already has `set_lbu`/`set_ubu`/`set_idxbu`/`nbu` support (unused currently).
- **AbstractFunction** (`include/trajopt/functions/abstract_function.h`): Base class. `get_jac_u()` default returns zero matrix. Derived classes override.
- **AbstractConstraint** (`include/trajopt/constraints/abstract_constraint.h`): Adds bounds `[lb, ub]`.

### Key Pinocchio APIs needed
- `pinocchio::rnea(model, data, q, v, a, fext)` — inverse dynamics with external forces
- `pinocchio::computeRNEADerivatives(model, data, q, v, a, fext, dtau_dq, dtau_dv, dtau_da)` — derivatives with fext
- `pinocchio::forwardKinematics(model, data, q)` — FK for frame poses
- `pinocchio::updateFramePlacements(model, data)` — compute frame placements
- `pinocchio::computeFrameJacobian(model, data, q, frame_id, LOCAL_WORLD_ALIGNED, J)` — 6×nv frame Jacobian
- `model.getFrameId(name)` — frame name to ID lookup
- `model.frames[frame_id].parentJoint` — parent joint of frame
- `model.frames[frame_id].placement` — SE3 placement of frame w.r.t. parent joint

### Test infrastructure
- Tests in `tests/constraints/test_inverse_dynamics.cpp`
- Pattern: `DoubleIntegratorFixture`, random state, compute analytical Jacobian, compare with `numerical_jacobian_node()` finite differences
- Tolerance: absolute 1e-4, relative 1e-6

---

## Step 1: Add contact configuration to Node

**Files:** `include/trajopt/node.h`, `src/trajopt/node.cpp`

### Contact storage
```cpp
struct Contact {
    std::string name;       // frame name (for user API)
    int frame_id;           // pinocchio frame index (looked up from model)
    bool active = true;     // toggle for MPC
};
std::vector<Contact> _contacts;
```

### API
- `add_contact(const std::string& frame_name)` — registers contact by name, looks up `frame_id` from `model.getFrameId(frame_name)`. Must be called before `add_constraint`/`add_cost`.
- `set_active_contacts(const std::vector<std::string>& names)` — sets the full active set in one call. Clears all active flags, then activates the named contacts. Fast for MPC: one call per node per MPC step.
- `int n_contacts() const` — total registered (not just active)
- `const std::vector<Contact>& contacts() const`

### Dimension changes
- `nu()` → `_nv + 3 * n_contacts()`
- `ndu()` → same (forces are Euclidean)

### Accessors
- `a()` remains `_u_ptr->head(_nv)` — unchanged
- `f(int i)` → `_u_ptr->segment(_nv + 3*i, 3)` (i-th contact force)
- `f()` → `_u_ptr->tail(3 * n_contacts())` (all forces stacked)

### Other updates
- Resize gradient vectors `_grad_*_u` in `add_contact()` since dimensions change
- Update `to_json()` to serialize contact forces and frame names
- `u_oplus()` — still Euclidean, no change needed

**Note:** Contacts must be added before any costs/constraints since `allocate_slices()` reads `nu()`/`ndu()`.

## Step 2: Modify InvDynamics to use fext

**Files:** `include/trajopt/constraints/inverse_dynamics.h`, `src/trajopt/constraints/inverse_dynamics.cpp`

### New members
- `PINOCCHIO_ALIGNED_STD_VECTOR(pinocchio::Force) _fext` (size `model.njoints`, initialized to zero)
- `MatrixXd _Jframe` (6×nv scratch for frame Jacobian computation)

### `allocate_slices()`
- `nc = _node->n_contacts()`
- `_input_dim = 2*nv + nv + 3*nc` (state + accel + forces)
- Resize `_jacobian` to `(nv × _input_dim)`
- Allocate `_fext` vector of size `model.njoints`, all zero
- Allocate `_Jframe` (6 × nv)

### `evaluate_impl()`
- Extract `q, v, a` from node
- Call `pinocchio::forwardKinematics(model, data, q)` + `pinocchio::updateFramePlacements(model, data)` for frame poses
- Build `_fext`: zero all, then for each **active** contact `i`:
  - Get `frame_id` and parent joint from model
  - Get 3D force `f_i = node->f(i)` (world frame)
  - Transform to spatial wrench at parent joint in local frame using frame placement
  - Accumulate into `_fext[parent_joint]`
- Call `pinocchio::rnea(model, data, q, v, a, _fext)`
- `_value = data.tau`

### `jacobian_impl()`
- Build `_fext` same as evaluate (or cache from evaluate)
- Call `pinocchio::computeRNEADerivatives(model, data, q, v, a, _fext, dtau_dq, dtau_dv, dtau_da)` — gives correct ∂τ/∂q, ∂τ/∂v, ∂τ/∂a accounting for fext
- Fill `_jacobian` blocks for q, v, a as before
- For each **active** contact `i`: compute frame Jacobian via `pinocchio::computeFrameJacobian(model, data, q, frame_id, LOCAL_WORLD_ALIGNED, _Jframe)`, extract linear part (top 3 rows). Set `∂τ/∂fᵢ = -_Jframe.topRows(3).transpose()` (nv × 3 block, negative sign because external forces reduce required torques)
- For **inactive** contacts: corresponding columns in Jacobian are zero

### `get_jac_u()`
- Return `_jacobian.rightCols(nv + 3*nc)` — covers both acceleration and force columns

### Minor: Update AccelerationCost

**File:** `src/trajopt/costs/acceleration_cost.cpp`

- `evaluate_impl()`: change `_node->u()` to `_node->a()`
- `get_jac_u()`: change from `middleCols(2*nv, nv)` to `_jacobian.rightCols(_node->ndu())`
- `allocate_slices()` / `jacobian_impl()`: already correct (identity block position unchanged, `_input_dim` from `ndu()` auto-expands)

## Step 3: Update tests

**File:** `tests/constraints/test_inverse_dynamics.cpp`

- Add a test case with external forces: create node with contacts (by frame name), set random forces in `u`, verify Jacobian against finite differences (including ∂τ/∂f columns)
- Add test for activate/deactivate: verify inactive contacts produce zero force contribution
- The existing no-contact test should still pass (nc=0 → same behavior)

---

## Files to modify
1. `include/trajopt/node.h` — Contact struct, contact storage, new accessors, updated `nu()`/`ndu()`
2. `src/trajopt/node.cpp` — `add_contact()`, `set_active_contacts()`, gradient resizing, `to_json()`
3. `include/trajopt/constraints/inverse_dynamics.h` — new members (fext, Jframe)
4. `src/trajopt/constraints/inverse_dynamics.cpp` — fext-aware evaluation and Jacobians
5. `src/trajopt/costs/acceleration_cost.cpp` — minor: use `a()` instead of `u()`, fix `get_jac_u()`
6. `tests/constraints/test_inverse_dynamics.cpp` — new test with contacts

## Files NOT modified
- `src/sqp_solver/sqp_solver.cpp` — already reads per-node `nu()`/`ndu()`, no box constraint changes needed
- `include/qp_solver/hpipm_solver.h` — no changes
- `src/trajopt/ocp.cpp` — uses per-node `nu()` in `finalize()`, works automatically
- Integration schemes (`euler.h`, `semi-euler.h`) — don't depend on control dimension directly

## Verification
1. Build: `cmake --build build`
2. Run existing inverse dynamics test (no contacts → same as before)
3. Run new test with contacts: verify finite-difference Jacobian matches analytical (including ∂τ/∂f)
4. Run new test: `set_active_contacts()` toggle, verify inactive forces don't affect τ
5. Run full SQP pipeline test to verify nothing broke
