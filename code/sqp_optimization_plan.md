# SQP Solver Optimization Plan

## Context

The SQP solver (`src/sqp_solver/sqp_solver.cpp`) has a clear, well-structured
pipeline but contains correctness bugs and computational bottlenecks. The goal
is to fix correctness first, then eliminate redundant work.

> **Note on evaluate/get separation (commit `6bfef86`):** `evaluate()` triggers
> computation and stores the result in `_value`. `get_value()` reads it back
> without recomputing. `calc_cost()` / `calc_dynamics_defect()` /
> `calc_constraint_violation()` are **read-only** — they call `get_value()`,
> they do NOT call `evaluate()`. Any plan item that describes them as
> "re-invoking evaluations" is outdated.

### Pipeline Overview

```
solve() [sqp_solver.cpp]
  for each SQP iteration:
    1. linearize()        → per node: dyn.evaluate/jacobian, cost.evaluate/jacobian, con.evaluate/jacobian
    2. populate_qp()      → copy A,B,b,Q,q,R,r,C,D,lg,ug into HPIPM
    3. qp_solver.solve()  → HPIPM interior point
    4. step()             → compute candidate trajectory at current alpha, bind nodes to candidates
    5. [linesearch loop]  → ls_filter() or ls_merit() x max_ls_iters
    6. accept_step()      → swap candidate into nominal (O(1))
    7. stats.update       → reads cost/viol/defect (stale cached values)
    8. break_criteria()   → step_norm < tolerance
```

---

## Issues (by priority)

---

### BUG-1. `ls_filter()` evaluates on **nominal**, not candidate — `src/sqp_solver/sqp_ls_filter.cpp`

**Root cause:** After `step()`, nodes are bound to the candidate trajectory.
But `ocp.cost()` / `ocp.constraint_violation()` / `ocp.dynamics_defect()` call
`node.calc_cost()` / `calc_constraint_violation()` / `calc_dynamics_defect()`,
which only call `get_value()` — a read of the last stored result. The last
`evaluate()` was called inside `linearize()`, around the **nominal** point.
So `ls_filter()` is comparing nominal-point residuals against `_prev_*`,
not candidate residuals. The filter does not actually observe the step.

**Fix:** Add a method on `OCP` that calls `evaluate()` on every function for
every node (a "forward pass" at the current binding). Call it inside `step()`
after `bind_trajectory()`.

```cpp
// In ocp.h / ocp.cpp:
void OCP::evaluate_all() {
    for (auto& node : _nodes) {
        if (node.get_dynamics()) node.get_dynamics()->evaluate();
        for (auto& c : node.get_costs())       c->evaluate();
        for (auto& c : node.get_constraints()) c->evaluate();
    }
}
```

Call site in `step()` (after `bind_trajectory`):
```cpp
_ocproblem.bind_trajectory(_x_candidate, _u_candidate);
_ocproblem.evaluate_all();   // refresh _value for all functions at candidate
```

**Files:**
- `include/trajopt/ocp.h` — declare `evaluate_all()`
- `src/trajopt/ocp.cpp` — implement `evaluate_all()`
- `src/sqp_solver/sqp_solver.cpp` — call `_ocproblem.evaluate_all()` in `step()`

---

### BUG-2. `_prev_cost / _prev_viol / _prev_defect` never updated — `src/sqp_solver/sqp_solver.cpp`

**Root cause:** These three members are initialized to `0.0` in `init()` and
never written again. The filter always compares the candidate against 0, so
it always accepts on the first LS iteration regardless of step quality.

**Fix:** After `accept_step()`, update `_prev_*` from the cached `_last_*`
values (populated by `ls_filter()` — see PERF-1):
```cpp
accept_step();
_prev_cost   = _last_cost;
_prev_viol   = _last_viol;
_prev_defect = _last_defect;
```

**Files:**
- `src/sqp_solver/sqp_solver.cpp` — add update of `_prev_*` after `accept_step()`

---

### PERF-1. `ls_filter()` calls `ocp.cost()` twice and does 3 separate traversals — `src/sqp_solver/sqp_ls_filter.cpp`

**Root cause:**
```cpp
if (_ocproblem.cost() < _prev_cost ||
    std::fabs(_ocproblem.cost() - _prev_cost) <= eps || ...   // cost() called TWICE
```
Short-circuit evaluation does not help — both `||` branches for cost call
`ocp.cost()` independently. Each of the three `ocp.*()` calls is a full
N-node traversal, totalling 3 passes per LS iteration.

**Fix:** Cache into locals at the top, and store into members for reuse in
stats (eliminating the stats traversal — see PERF-2):
```cpp
bool SQPSolver::ls_filter() {
    double cur_cost   = _ocproblem.cost();
    double cur_viol   = _ocproblem.constraint_violation();
    double cur_defect = _ocproblem.dynamics_defect();
    _last_cost   = cur_cost;
    _last_viol   = cur_viol;
    _last_defect = cur_defect;
    if (cur_cost   < _prev_cost   || std::fabs(cur_cost   - _prev_cost)   <= eps ||
        cur_viol   < _prev_viol   || std::fabs(cur_viol   - _prev_viol)   <= eps ||
        cur_defect < _prev_defect || std::fabs(cur_defect - _prev_defect) <= eps)
        return true;
    return false;
}
```

Add to `sqp_solver.h`:
```cpp
double _last_cost   = 0.;
double _last_viol   = 0.;
double _last_defect = 0.;
```

**Files:**
- `src/sqp_solver/sqp_ls_filter.cpp` — cache into locals, store in `_last_*`
- `include/sqp_solver/sqp_solver.h` — add `_last_cost / _last_viol / _last_defect`

---

### PERF-2. Stats after `accept_step()` make 3 redundant traversals — `src/sqp_solver/sqp_solver.cpp:34-38`

**Root cause:**
```cpp
accept_step();
_stats.update_cost(_ocproblem.cost());                        // N-node traversal
_stats.update_constraint_violation(_ocproblem.constraint_violation()); // N-node traversal
_stats.update_dynamics_defect(_ocproblem.dynamics_defect());           // N-node traversal
```
These are 3 separate passes reading cached `get_value()` results.
After PERF-1 is done, the accepted values are already in `_last_*`.

**Fix:** Use the cached members instead of re-traversing:
```cpp
accept_step();
_prev_cost   = _last_cost;
_prev_viol   = _last_viol;
_prev_defect = _last_defect;
_stats.update_cost(_last_cost);
_stats.update_constraint_violation(_last_viol);
_stats.update_dynamics_defect(_last_defect);
```

When `LSType::NONE` (no line search), `_last_*` won't have been populated by
`ls_filter()`. Compute once explicitly:
```cpp
if (!_ls_function) {
    _last_cost   = _ocproblem.cost();
    _last_viol   = _ocproblem.constraint_violation();
    _last_defect = _ocproblem.dynamics_defect();
}
```

**Files:**
- `src/sqp_solver/sqp_solver.cpp` — replace 3 `ocp.*()` stats calls with `_last_*`

---

### PERF-3. `get_jac_u()` heap-allocates a `MatrixXd` on every call — `include/trajopt/functions/abstract_function.h:86`

**Root cause:**
```cpp
virtual MatrixXd get_jac_u() const {
    return MatrixXd::Zero(_output_dim, 0);  // heap alloc every call
}
```
Called in `linearize()` for every cost and every constraint per node per SQP
iteration. For state-only functions (the common case) this allocates a
zero-column matrix only to immediately check `cols() == 0` and discard it.

**Fix:** Add a virtual predicate, default `false`, overridden in classes that
have control dependency:
```cpp
// abstract_function.h
virtual bool has_jac_u() const { return false; }

// euler.h, inverse_dynamics.h
bool has_jac_u() const override { return true; }
```

Guard the call sites in `linearize()`:
```cpp
if (cost->has_jac_u() && i < _Nu) {
    MatrixXd Ju = cost->get_jac_u();
    _R[i].noalias() += w * Ju.transpose() * Ju;
    _r[i].noalias() += w * Ju.transpose() * cost_val;
}
```

**Files:**
- `include/trajopt/functions/abstract_function.h` — add `virtual bool has_jac_u() const { return false; }`
- `include/trajopt/constraints/inverse_dynamics.h` — override to `return true`
- `include/trajopt/constraints/integration_schemes/euler.h` — override to `return true`
- `src/sqp_solver/sqp_solver.cpp` — guard `get_jac_u()` in cost and constraint loops

---

### PERF-4. Temporary vector allocations in `step()` — `src/sqp_solver/sqp_solver.cpp:203,210`

**Root cause:**
```cpp
VectorXd scaled_dx = _ls_alpha * _dx[k];   // heap alloc per node per call
VectorXd scaled_du = _ls_alpha * _du[k];   // heap alloc per node per call
```
`step()` is called up to `max_ls_iters` (default 20) times per SQP iteration,
allocating `2N` temporaries each time. The norm computation also creates a
separate temporary: `(_ls_alpha * _dx[k]).cwiseAbs()`.

**Fix:** Pre-allocate scratch vectors alongside `_dx` / `_du` in `init()`:
```cpp
// sqp_solver.h
std::vector<VectorXd> _scaled_dx;
std::vector<VectorXd> _scaled_du;
```

In `step()`:
```cpp
_scaled_dx[k].noalias() = _ls_alpha * _dx[k];
_step_norm = std::max(_step_norm, _scaled_dx[k].cwiseAbs().maxCoeff());
_ocproblem.get_node(k).x_oplus(x_nom[k], _scaled_dx[k], _x_candidate[k]);
```

**Files:**
- `include/sqp_solver/sqp_solver.h` — add `_scaled_dx`, `_scaled_du`
- `src/sqp_solver/sqp_solver.cpp` — initialize in `init()`, use in `step()`

---

### PERF-5. `std::cout` + `std::endl` flush inside innermost loop — `src/sqp_solver/sqp_solver.cpp:266`

**Root cause:**
```cpp
std::cout << "Linearizing :" << cost->get_name() << std::endl;
```
`std::endl` flushes stdout on every cost × every node × every SQP iteration.

**Fix:**
```cpp
DEBUG_PRINT("Linearizing :" << cost->get_name());
```

**Files:**
- `src/sqp_solver/sqp_solver.cpp:266`

---

### CLEANUP-1. Misleading warm-start comment — `src/sqp_solver/sqp_solver.cpp:159`

**Current:**
```cpp
_qp_solver.set_warm_start(false);  // CRITICAL: Enable warm-starting (huge speedup)
```
Comment says "enable" but code sets `false`.

**Fix:**
```cpp
_qp_solver.set_warm_start(false);  // TODO: test enabling warm-start for speedup
```

**Files:**
- `src/sqp_solver/sqp_solver.cpp:159`

---

## TODO List

Work in this order — bugs first, then perf, then cleanup.

- [ ] **BUG-1** — Add `OCP::evaluate_all()` and call it in `step()` after `bind_trajectory()`
- [ ] **BUG-2** — Add `_last_cost/viol/defect` members; update `_prev_*` from them after `accept_step()`
- [ ] **PERF-1** — Cache `ocp.*()` into locals in `ls_filter()`; store in `_last_*`
- [ ] **PERF-2** — Use `_last_*` for stats update; handle `LSType::NONE` case
- [ ] **PERF-3** — Add `has_jac_u()` predicate; override in Euler/InvDyn; guard `get_jac_u()` in `linearize()`
- [ ] **PERF-4** — Pre-allocate `_scaled_dx` / `_scaled_du`; use in `step()`
- [ ] **PERF-5** — Replace `std::cout` with `DEBUG_PRINT` in `linearize()`
- [ ] **CLEANUP-1** — Fix misleading warm-start comment

---

## Files to Modify

| File | Changes |
|------|---------|
| `include/trajopt/ocp.h` | Declare `evaluate_all()` |
| `src/trajopt/ocp.cpp` | Implement `evaluate_all()` |
| `include/trajopt/functions/abstract_function.h` | Add `virtual bool has_jac_u() const { return false; }` |
| `include/trajopt/constraints/inverse_dynamics.h` | Override `has_jac_u()` → `true` |
| `include/trajopt/constraints/integration_schemes/euler.h` | Override `has_jac_u()` → `true` |
| `include/sqp_solver/sqp_solver.h` | Add `_scaled_dx`, `_scaled_du`, `_last_cost`, `_last_viol`, `_last_defect` |
| `src/sqp_solver/sqp_solver.cpp` | Call `evaluate_all()` in `step()`; update `_prev_*` after accept; use `_last_*` for stats; pre-alloc scratch; fix cout; fix comment |
| `src/sqp_solver/sqp_ls_filter.cpp` | Cache into locals; store in `_last_*` |

---

## Verification

After implementing:
```bash
cmake --build /workspace/build -j$(nproc)
ctest --test-dir /workspace/build -V
```
- Check no functional regression (same convergence curve, same final cost)
- With BUG-1+BUG-2 fixed: filter should now reject bad steps — convergence
  behaviour may change, this is expected and correct
- Compare `last_iteration_time_ms` from `SQPstatistics` before/after perf fixes
