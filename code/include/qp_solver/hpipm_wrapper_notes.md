# HPIPM Wrapper Notes

## Reference: hucebot/hpipm-cpp
https://github.com/hucebot/hpipm-cpp

### Features our wrapper is missing vs the reference
- **HpipmMode enum**: SpeedAbs, Speed, Balance, Robust (we hardcode ROBUST — slowest)
- **HpipmStatus return**: Success, MaxIterReached, MinStepReached, NaNDetected (our `solve()` returns void)
- **OcpQpIpmSolverSettings**: Full IPM config — mode, iter_max, alpha_min, mu0, tol_*, reg_prim, warm_start, pred_corr, ric_alg, split_step
- **OcpQpIpmSolverStatistics**: Iteration count, max residuals, per-iteration convergence data (18 fields/iter)
- **Constraint masks**: lbx/ubx/lbu/ubu/lg/ug masks for selective constraint activation
- **Soft constraints**: Zl, Zu, zl, zu, idxs, lls, lus (slack penalties)
- **Riccati feedback**: P, p, K, k extraction from workspace
- **Batch data setting**: `d_ocp_qp_set_all()` — single call replaces ~10N individual setter calls
- **Memory reuse**: Only realloc if new size exceeds peak (we malloc/free every time)

### Speed improvements available
1. `d_ocp_qp_set_all` — 1 call instead of ~10N individual setter calls per QP populate
2. Default to SPEED mode instead of ROBUST (fewer IPM iterations)
3. Enable `pred_corr=1` and `ric_alg=1` (prediction-correction + square-root Riccati)
4. Memory reuse pattern (track peak allocation, skip realloc when dimensions unchanged)
5. All hot-path methods already inline (good)

### Settings defaults (reference repo)
```
mode       = Speed        (not Robust!)
iter_max   = 15           (we use 200)
alpha_min  = 1e-8
mu0        = 1e2
tol_*      = 1e-8         (we use 1e-3)
reg_prim   = 1e-12
warm_start = 0            (we enable it — good for SQP)
pred_corr  = 1
ric_alg    = 1
split_step = 0
```

### HPIPM C API functions available (verified in /opt/hpipm/include/)
- `d_ocp_qp_set_all(A**, B**, b**, Q**, S**, R**, q**, r**, idxbx**, lbx**, ubx**, idxbu**, lbu**, ubu**, C**, D**, lg**, ug**, Zl**, Zu**, zl**, zu**, idxs**, idxs_rev**, ls**, us**, qp*)`
- `d_ocp_qp_set_{lbx,ubx,lbu,ubu,lg,ug}_mask(stage, vec*, qp*)`
- `d_ocp_qp_set_{Zl,Zu,zl,zu,lls,lus,idxs,idxs_rev}(stage, vec*, qp*)`
- `d_ocp_qp_ipm_arg_set_{iter_max,alpha_min,mu0,tol_stat,tol_eq,tol_ineq,tol_comp,reg_prim,warm_start,pred_corr,ric_alg,split_step}(val*, arg*)`
- `d_ocp_qp_ipm_get_{status,iter,max_res_stat,max_res_eq,max_res_ineq,max_res_comp,obj,stat,stat_m}(ws*, out*)`
- `d_ocp_qp_ipm_get_ric_{P,p,K,k}(qp*, arg*, ws*, stage, out*)`
- `d_ocp_qp_dim_set_ns(stage, val, dim*)`

---

## Multiple Shooting vs. Single Shooting Analysis

### The issue
HPIPM's `d_ocp_qp` dynamics slot (`A, B, b`) enforces `x_{k+1} = A x_k + B u_k + b` as a **hard equality** via Riccati elimination. The QP solution always satisfies linearized dynamics exactly — zero residual by construction.

### Consequence
With `dx_0 = 0` (fixed initial state), states are determined by controls: `dx_1 = B_0 du_0`, `dx_2 = A_1 dx_1 + B_1 du_1 + b_1`, etc. This is effectively single-shooting behavior wrapped in a multiple-shooting structure.

### When it matters
- **Feasible init** (forward rollout, defect=0): Stays single-shooting-like. Defect ≈ 0 throughout.
- **Infeasible init** (arbitrary states, defect != 0): `b_k` captures nonlinear defect. QP closes the linearized gap. After `x_oplus`, linearization error creates new (smaller) defect. Over SQP iterations, defect → 0. This IS multiple shooting behavior.

### Potential issues with current approach
1. States not truly independent — Riccati eliminates them
2. Linearization errors compound for long horizons (N >> 1)
3. QP conditioning degrades with horizon length
4. Merit function line search doesn't see dynamics infeasibility

### If true multiple shooting is ever needed
Move dynamics from `A/B/b` slot into general constraints (`C/D/lg/ug` with `lg=ug=0`). States become free QP variables. HPIPM won't eliminate them via Riccati. QP is larger but convergence is better for nonlinear problems with long horizons.
