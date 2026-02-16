# Multiple Shooting & Dynamics Defect in HPIPM

## How HPIPM handles dynamics
HPIPM's OCP QP structure treats the dynamics slot (`A, B, b`) as a **hard equality constraint** eliminated by the Riccati recursion. The QP solution always satisfies `dx_{k+1} = A_k dx_k + B_k du_k + b_k` exactly — zero residual by construction. There is no slack or relaxation possible in this slot.

## What this means for SQP
In the SQP loop, we linearize: `b_k = dyn->get_value()` (the nonlinear dynamics defect at the current trajectory). HPIPM solves the QP and produces perturbations `dx, du` that **exactly close the linearized gap**. After applying the step via `x_oplus`, any remaining defect is purely from nonlinearity (second-order).

## Initialization determines shooting behavior

- **Infeasible init** (arbitrary states, defect != 0): `b_k` is non-zero. The QP corrects the linearized gap. After `x_oplus`, linearization error creates new (smaller) defect. Over SQP iterations, defect converges to 0. This IS proper multiple shooting.

- **Feasible init** (forward rollout, defect = 0): `b_k = 0` everywhere. Combined with `dx_0 = 0` (fixed initial state), states propagate forward: `dx_1 = B_0 du_0`, `dx_2 = A_1 dx_1 + B_1 du_1`, etc. Defect stays ~0 throughout. This degenerates to **single-shooting-like behavior**.

## Why this matters
1. Single shooting propagates linearization errors — conditioning degrades for long horizons
2. States are not truly independent — Riccati eliminates them, so the optimizer can't explore infeasible trajectories
3. Merit function line search doesn't see dynamics infeasibility (defect is always ~0)
4. Multiple shooting is more robust to initialization and converges better for nonlinear problems
