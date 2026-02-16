# SQP Solver Pipeline

## Overview

The SQP solver iteratively solves a nonlinear trajectory optimization problem by
approximating it as a sequence of quadratic programs (QPs). Each iteration
follows a fixed pipeline of stages.

## Pipeline

```
for each SQP iteration:

    1. linearize()
    2. populate_qp()
    3. qp_solver.solve()
    4. step()  ──┐
    5. ls_check() │  backtracking loop
       step()  ──┘
    6. accept_step()
    7. stats (cost / violation / defect)
    8. break_criteria()
```

## Stages

### 1. `linearize()`

Rebinds nodes to the **nominal** trajectory, then for each function (dynamics,
costs, constraints) calls:

```cpp
f->evaluate();   // compute f(x_nom, u_nom), store in f->_value
f->jacobian();   // compute ∂f/∂x, ∂f/∂u, store in f->_jacobian

// then read the results:
f->get_value()        // → _value
f->get_jac_x()        // → ∂f/∂x block
f->get_jac_u()        // → ∂f/∂u block
```

`evaluate()` and `jacobian()` are pure triggers — they store results internally.
`get_value()` / `get_jac_*()` are pure readers — no recomputation.

This is the safe reset point: no matter what the previous line search did,
`linearize()` always starts from nominal and produces fresh stored values.

**Produces:** `A, B, b` (dynamics), `Q, q, R, r` (costs), `C, D, lg, ug` (constraints)

### 2. `populate_qp()`

Copies the linearization matrices into the HPIPM QP solver data structures.
Pure data transfer, no computation.

### 3. `qp_solver.solve()`

Solves the structured QP to get the step directions `dx, du`.

### 4. `step()`

Computes the candidate trajectory at the current step size `alpha`:

```
x_candidate[k] = x_nom[k] + alpha * dx[k]   (via manifold oplus)
u_candidate[k] = u_nom[k] + alpha * du[k]
```

**Binds nodes to candidates** so that any subsequent `evaluate()` calls
read from the candidate trajectory.

> **Important:** After `step()`, the nodes point to candidates but all
> stored `_value` / `_jacobian` are **stale** (from the last `linearize()`
> around nominal). Calling `get_value()` without `evaluate()` first returns
> the nominal-point residuals, not candidate residuals.

### 5. Line search (backtracking loop)

`ls_merit()` / `ls_filter()` evaluate acceptance of the candidate step.
Since `step()` already bound nodes to candidates, calling `evaluate()` on
any function now computes on the candidate — no manual binding or restoring
needed.

`ocp.cost()` / `ocp.constraint_violation()` / `ocp.dynamics_defect()` call
`node.calc_cost()` / `node.calc_dynamics_defect()` / `node.calc_constraint_violation()`,
which read `f->get_value()` (the last stored result). Therefore, for the LS
functions to see candidate values, a re-evaluation pass must be triggered
**before** calling these helpers (e.g. via an explicit `evaluate_all()` or
by having `calc_*` call `evaluate()` internally).

If rejected: reduce `alpha *= ls_scale_factor`, call `step()` again
(recomputes candidates at new alpha and rebinds).

If `LSType::NONE`: the loop is skipped entirely, full step is taken.

### 6. `accept_step()`

Swaps the candidate and nominal trajectory vectors (O(1), no copies) and
rebinds nodes to the new nominal.

After this call, `x_traj()`/`u_traj()` contain the accepted step and nodes
point to them.

### 7. Stats

```cpp
_stats.update_cost(_ocproblem.cost());
_stats.update_constraint_violation(_ocproblem.constraint_violation());
_stats.update_dynamics_defect(_ocproblem.dynamics_defect());
```

These call `calc_cost()` / `calc_dynamics_defect()` / `calc_constraint_violation()`
which read `get_value()` — the **last stored** evaluation result. They do not
re-trigger `evaluate()`. After `accept_step()` the nodes are on the new nominal
but `_value` is still from `linearize()` at the previous nominal. The stats
therefore reflect the residuals at the previous nominal point (i.e. the defect
and violation that motivated the step). A fresh `evaluate()` pass would be
needed to get values at the accepted point.

### 8. `break_criteria()`

Checks if the step norm is below tolerance. If yes, the solver has converged.

## Node Binding State Machine

The key invariant is **which trajectory the nodes point to**:

```
linearize()    -> nodes on NOMINAL  (always rebinds at start)
step()         -> nodes on CANDIDATE
accept_step()  -> nodes on NOMINAL  (candidates become nominal via swap)
```

This means:
- `linearize()` can always be called safely — it resets to nominal
- After `step()`, any `evaluate()` call computes on the candidate
- If all candidates are rejected (e.g. to increase regularization), just
  call `linearize()` next — it rebinds to nominal automatically

## Evaluate / Get separation

All functions (`AbstractFunction` subclasses) follow a two-phase pattern:

| Phase | Methods | Effect |
|-------|---------|--------|
| Compute | `evaluate()` → `evaluate_impl()` | Runs computation, writes to `_value` |
| Compute | `jacobian()` → `jacobian_impl()` | Runs computation, writes to `_jacobian` |
| Read | `get_value()` | Returns `_value` (no recomputation) |
| Read | `get_jac_x()`, `get_jac_u()` | Returns blocks of `_jacobian` (no recomputation) |

`allocate_slices()` must resize both `_value` and `_jacobian` during setup.

In `linearize()` both phases are called explicitly. In the line search and
stats helpers, only the read phase is called — so values reflect the point
at which `evaluate()` was **last** called, which is the nominal point from
`linearize()`. Re-evaluating on the candidate requires an explicit `evaluate()`
call after `step()`.
