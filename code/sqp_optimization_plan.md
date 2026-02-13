# SQP Solver Optimization Plan

## Context

The SQP solver (`code/src/sqp_solver/sqp_solver.cpp`) has a clear, well-structured pipeline but contains several logical computational bottlenecks — places where expensive function evaluations are repeated unnecessarily, temporaries are allocated inside hot loops, and solver configuration is left in a suboptimal state. The goal is to eliminate redundant work without changing the algorithm's correctness.

### Pipeline Overview

```
solve() [sqp_solver.cpp]
  for each SQP iteration:
    1. linearize()        → per node: dyn.evaluate/jacobian, cost.evaluate/jacobian, con.evaluate/jacobian
    2. populate_qp()      → copy A,B,b,Q,q,R,r,C,D,lg,ug into HPIPM
    3. qp_solver.solve()  → HPIPM interior point
    4. step()             → compute candidate trajectory at current alpha
    5. [linesearch loop]  → ls_filter() or ls_merit() x max_ls_iters
    6. accept_step()      → swap candidate into nominal (O(1))
    7. stats.update + print → re-evaluates cost/viol/defect for display
    8. break_criteria()   → step_norm < tolerance
```

---

## Identified Bottlenecks (by priority)

### 1. [HIGH] Triple re-evaluation in `ls_filter()` — `code/src/sqp_solver/sqp_ls_filter.cpp`

**Root cause:** Each of `_ocproblem.cost()`, `_ocproblem.constraint_violation()`, and `_ocproblem.dynamics_defect()` independently loops over all N nodes and calls `calc_cost()`/`calc_constraint_violation()`/`calc_dynamics_defect()` which re-invoke the underlying function evaluations.

**Current code:**
```cpp
bool SQPSolver::ls_filter() {
    if (_ocproblem.cost() <  _prev_cost ||
        std::fabs(_ocproblem.cost() - _prev_cost) <= std::numeric_limits<double>::epsilon() ||
        _ocproblem.constraint_violation() < _prev_viol ||
        std::fabs(_ocproblem.constraint_violation() - _prev_viol) <= std::numeric_limits<double>::epsilon() ||
        _ocproblem.dynamics_defect() < _prev_defect ||
        std::fabs(_ocproblem.dynamics_defect() - _prev_defect) <= std::numeric_limits<double>::epsilon())
        return true;
    return false;
}
```

`cost()` is even called TWICE in the condition due to short-circuit evaluation not helping when both branches reference it.

**Fix:** Cache the three values at the top of the function:
```cpp
bool SQPSolver::ls_filter() {
    double cur_cost   = _ocproblem.cost();
    double cur_viol   = _ocproblem.constraint_violation();
    double cur_defect = _ocproblem.dynamics_defect();
    if (cur_cost < _prev_cost   || std::fabs(cur_cost   - _prev_cost)   <= std::numeric_limits<double>::epsilon() ||
        cur_viol < _prev_viol   || std::fabs(cur_viol   - _prev_viol)   <= std::numeric_limits<double>::epsilon() ||
        cur_defect < _prev_defect || std::fabs(cur_defect - _prev_defect) <= std::numeric_limits<double>::epsilon())
        return true;
    return false;
}
```

Also: store these as members (`_last_cost`, `_last_viol`, `_last_defect`) to reuse them in the stats update (see #2).

---

### 2. [HIGH] Statistics re-evaluation after `accept_step()` — `code/src/sqp_solver/sqp_solver.cpp:34-38`

**Current code (in `solve()`):**
```cpp
accept_step();
_stats.update();
_stats.update_cost(_ocproblem.cost());                       // Full re-evaluation
_stats.update_constraint_violation(_ocproblem.constraint_violation());  // Full re-evaluation
_stats.update_dynamics_defect(_ocproblem.dynamics_defect());            // Full re-evaluation
_stats.print();
```

This is a **fourth full-trajectory evaluation cycle** every SQP iteration (after linearize, after each step(), and after ls_filter's per-LS-iter checks).

**Fix:** Store the last accepted values in SQPSolver members:
- Add to `sqp_solver.h`: `double _last_cost, _last_viol, _last_defect;`
- In `ls_filter()` (or after the LS loop in `solve()`): populate them once
- In `solve()`: pass cached values to stats:
  ```cpp
  _stats.update_cost(_last_cost);
  _stats.update_constraint_violation(_last_viol);
  _stats.update_dynamics_defect(_last_defect);
  ```

Note: `_prev_cost`, `_prev_defect`, `_prev_viol` already exist as members in `sqp_solver.h` but **are never updated** — they stay at their initial zero values. These should be updated after each `accept_step()` using the cached values.

---

### 3. [MEDIUM] `get_jac_u()` allocates a `MatrixXd` by value — `code/include/trajopt/functions/abstract_function.h:86`

**Current code:**
```cpp
virtual MatrixXd get_jac_u() const {
    return MatrixXd::Zero(_output_dim, 0);  // Allocates a new zero matrix every call
}
```

Called in `linearize()` for every cost and every constraint per node per SQP iteration:
```cpp
MatrixXd Ju = cost->get_jac_u();  // Heap allocation, even for state-only costs
if (Ju.cols() > 0 && i < _Nu) { ... }
```

For state-only costs (the common case), this allocates a zero-column matrix only to immediately check `cols() == 0` and discard it.

**Fix:** Add a virtual query method to `AbstractFunction`:
```cpp
virtual bool has_jac_u() const { return false; }
```
Override in concrete functions that have control dependency (`InvDynamics`, `EulerIntegration`, etc.) to return `true`. Guard the call in `linearize()`:
```cpp
if (cost->has_jac_u() && i < _Nu) {
    MatrixXd Ju = cost->get_jac_u();
    _R[i].noalias() += w * Ju.transpose() * Ju;
    _r[i].noalias() += w * Ju.transpose() * cost_val;
}
```

**Files to modify:**
- `code/include/trajopt/functions/abstract_function.h` — add `virtual bool has_jac_u() const { return false; }`
- `code/include/trajopt/constraints/inverse_dynamics.h` — override to `return true;`
- `code/include/trajopt/constraints/integration_schemes/euler.h` — override to `return true;`
- `code/src/sqp_solver/sqp_solver.cpp` — guard `get_jac_u()` calls in both cost and constraint loops

---

### 4. [MEDIUM] Temporary vector allocations in `step()` — `code/src/sqp_solver/sqp_solver.cpp:203,210`

**Current code:**
```cpp
for (int k = 0; k < _N; ++k) {
    _qp_solver.get_x(k, _dx[k].data());
    _step_norm = std::max(_step_norm, (_ls_alpha * _dx[k]).cwiseAbs().maxCoeff());
    VectorXd scaled_dx = _ls_alpha * _dx[k];   // Heap allocation per node per call
    _ocproblem.get_node(k).x_oplus(x_nom[k], scaled_dx, _x_candidate[k]);

    if (k < _Nu) {
        _qp_solver.get_u(k, _du[k].data());
        _step_norm = std::max(_step_norm, (_ls_alpha * _du[k]).cwiseAbs().maxCoeff());
        VectorXd scaled_du = _ls_alpha * _du[k];  // Heap allocation per node per call
        _ocproblem.get_node(k).u_oplus(u_nom[k], scaled_du, _u_candidate[k]);
    }
}
```

`step()` is called up to `max_ls_iters=20` times per SQP iteration. Each call allocates `2N` temporary vectors.

**Fix:** Add pre-allocated scratch vectors to `sqp_solver.h`:
```cpp
std::vector<VectorXd> _scaled_dx;
std::vector<VectorXd> _scaled_du;
```
Initialize them in `init()` alongside `_dx`/`_du`. Use in `step()`:
```cpp
_scaled_dx[k].noalias() = _ls_alpha * _dx[k];
_ocproblem.get_node(k).x_oplus(x_nom[k], _scaled_dx[k], _x_candidate[k]);
```
Also: the norm computation `(_ls_alpha * _dx[k]).cwiseAbs()` creates another temporary. Use `_scaled_dx[k]` for this too (compute once):
```cpp
_scaled_dx[k].noalias() = _ls_alpha * _dx[k];
_step_norm = std::max(_step_norm, _scaled_dx[k].cwiseAbs().maxCoeff());
_ocproblem.get_node(k).x_oplus(x_nom[k], _scaled_dx[k], _x_candidate[k]);
```

---

### 5. [LOW] `std::cout` inside innermost `linearize()` loop — `code/src/sqp_solver/sqp_solver.cpp`

**Current code:**
```cpp
for (auto& cost : _ocproblem.get_node(i).get_costs()) {
    std::cout << "Linearizing :" << cost->get_name() << std::endl;  // I/O + flush per cost per node
    cost->evaluate();
    ...
}
```

`std::endl` flushes stdout every call. This is in the innermost loop (per cost × per node × per SQP iter). Should be guarded with `DEBUG_PRINT` macro (already used in `sqp_solver.cpp` for other debug prints).

**Fix:**
```cpp
DEBUG_PRINT("Linearizing :" << cost->get_name());
```

---

### 6. [LOW / Config] Misleading warm-start comment — `code/src/sqp_solver/sqp_solver.cpp:159`

**Current code:**
```cpp
_qp_solver.set_warm_start(false);  // CRITICAL: Enable warm-starting (huge speedup)
```

Comment says to enable but code sets `false`. This is likely intentional for correctness (cold start on first QP may be needed), but the comment is misleading and should be clarified or the feature should be tested and enabled if safe.

---

## Files to Modify

| File | Changes |
|------|---------|
| `code/src/sqp_solver/sqp_ls_filter.cpp` | Cache cost/viol/defect into locals; store in member vars |
| `code/src/sqp_solver/sqp_solver.cpp` | Use cached values for stats; pre-alloc scratch vectors; guard cout; fix comment |
| `code/include/sqp_solver/sqp_solver.h` | Add `_scaled_dx`, `_scaled_du`, `_last_cost`, `_last_viol`, `_last_defect` members |
| `code/include/trajopt/functions/abstract_function.h` | Add `virtual bool has_jac_u() const { return false; }` |
| `code/include/trajopt/constraints/inverse_dynamics.h` | Override `has_jac_u()` to return `true` |
| `code/include/trajopt/constraints/integration_schemes/euler.h` | Override `has_jac_u()` to return `true` |

---

## Codebase Notes

- **`DEBUG_PRINT` macro**: Already used in `sqp_solver.cpp` — use it for all debug output
- **`_prev_cost / _prev_viol / _prev_defect`**: Declared as members in `sqp_solver.h` but never updated — part of the same fix (update them from `_last_*` after `accept_step()`)
- **`noalias()`**: Already used correctly throughout the Gauss-Newton accumulation — maintain this pattern
- **Abstract function Jacobian storage**: `_jacobian` (MatrixXd) in `AbstractFunction` stores the full Jacobian; `get_jac_x()` returns a const reference to it; `get_jac_u()` is virtual and returns by value (the allocation source)
- **OCP evaluation methods**: `ocp.cost()` → `node.calc_cost()` → `cost->get_value()` (uses **cached** value from last `evaluate()` call); `ocp.constraint_violation()` → `node.calc_constraint_violation()` → re-evaluates constraints via `get_value()`. So the OCP-level functions do NOT re-call `evaluate()` — they use cached values. The cost of repeated calls is one full N-node traversal per call, but NOT full re-differentiation. This makes fix #1 still important (three separate traversals → one) but the impact is proportional to N, not N × (costs+constraints).

---

## Verification

After implementing:
```bash
cd /workspace
cmake --build build -j$(nproc)
ctest --test-dir build -V
```
- Check no functional regression (same convergence curve, same final cost)
- Compare `last_iteration_time_ms` from `SQPstatistics` before/after
