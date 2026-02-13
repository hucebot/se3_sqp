# SQP Solver Pipeline

## Overview

The SQP solver iteratively solves a nonlinear trajectory optimization problem by
approximating it as a sequence of quadratic programs (QPs). Each iteration
follows a fixed pipeline of 5 stages.

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
    7. break_criteria()
```

## Stages

### 1. `linearize()`

Rebinds nodes to the **nominal** trajectory, then linearizes all dynamics,
costs, and constraints around it. This is the safe reset point — no matter
what the previous line search did, linearize always starts from nominal.

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

### 5. Line search (backtracking loop)

The `solve()` loop calls `step()` then checks acceptance via `_ls_function`
(either `ls_merit()` or `ls_filter()`). Since `step()` already bound the
nodes to candidates, the line search function just evaluates — no binding
or restoring needed.

If rejected: reduce `alpha *= ls_scale_factor`, call `step()` again
(recomputes candidates at new alpha and rebinds).

If `LSType::NONE`: the loop is skipped entirely, full step is taken.

### 6. `accept_step()`

Swaps the candidate and nominal trajectory vectors (O(1), no copies) and
rebinds nodes to the new nominal.

After this call, `x_traj()`/`u_traj()` contain the accepted step and nodes
point to them.

### 7. `break_criteria()`

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
- `ls_merit()`/`ls_filter()` can evaluate directly — nodes are on candidates
- If all candidates are rejected (e.g. to increase regularization), just
  call `linearize()` next — it rebinds to nominal automatically
