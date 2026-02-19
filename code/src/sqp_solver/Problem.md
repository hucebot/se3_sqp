# KKT Conditions for the SQP QP Subproblem

## 1. QP Subproblem (HPIPM Convention)

At each SQP iteration, the following QP is solved over deviations $\delta x_k$, $\delta u_k$:

$$
\min_{\delta x,\, \delta u} \;\sum_{k=0}^{N-1} \left[
  \frac{1}{2}\,\delta x_k^\top Q_k\,\delta x_k
  \;+\; q_k^\top \delta x_k
  \;+\; \frac{1}{2}\,\delta u_k^\top R_k\,\delta u_k
  \;+\; r_k^\top \delta u_k
  \;+\; \delta u_k^\top S_k\,\delta x_k
\right]
$$

subject to:

- **Dynamics** ($k = 0,\ldots,N_u{-}1$):

$$
\delta x_{k+1} = A_k\,\delta x_k + B_k\,\delta u_k + b_k
$$

- **Inequalities** ($k = 0,\ldots,N{-}1$):

$$
l_k^g \;\le\; C_k\,\delta x_k + D_k\,\delta u_k \;\le\; u_k^g
$$

where:

| Symbol | Meaning |
|--------|---------|
| $Q_k$ | State cost Hessian (Gauss–Newton approximation) |
| $q_k$ | State cost gradient |
| $R_k$ | Control cost Hessian |
| $r_k$ | Control cost gradient |
| $S_k$ | Cost cross-term $\partial^2 \ell / \partial u\,\partial x$ |
| $A_k = \partial f_k / \partial x_k$ | Dynamics state Jacobian at stage $k$ |
| $B_k = \partial f_k / \partial u_k$ | Dynamics control Jacobian at stage $k$ |
| $b_k$ | Dynamics affine (defect) term |
| $C_k,\; D_k$ | Inequality constraint Jacobians |

---

## 2. Lagrangian

Introduce multipliers:

- $\pi_k \in \mathbb{R}^{n_x}$ for dynamics constraint $k$ &ensp;($k = 0,\ldots,N_u{-}1$)
- $\lambda_k^{ug},\;\lambda_k^{lg} \ge 0$ for upper/lower inequality bounds

$$
\mathcal{L} = \sum_{k=0}^{N-1} \left[
  \frac{1}{2}\,\delta x_k^\top Q_k\,\delta x_k
  + q_k^\top \delta x_k
  + \frac{1}{2}\,\delta u_k^\top R_k\,\delta u_k
  + r_k^\top \delta u_k
  + \delta u_k^\top S_k\,\delta x_k
\right]
$$

$$
+\;\sum_{k=0}^{N_u-1} \pi_k^\top \Big(
  A_k\,\delta x_k + B_k\,\delta u_k + b_k - \delta x_{k+1}
\Big)
$$

$$
+\;\sum_{k=0}^{N-1} \Big(\lambda_k^{ug}\Big)^\top \Big(
  C_k\,\delta x_k + D_k\,\delta u_k - u_k^g
\Big)
$$

$$
-\;\sum_{k=0}^{N-1} \Big(\lambda_k^{lg}\Big)^\top \Big(
  C_k\,\delta x_k + D_k\,\delta u_k - l_k^g
\Big)
$$

---

## 3. Stationarity w.r.t. $\delta x_k$

Collect every term in $\mathcal{L}$ containing $\delta x_k$.
The variable $\delta x_k$ appears in three places:

| Source | Term containing $\delta x_k$ | Derivative $\partial\,/\,\partial(\delta x_k)$ |
|--------|------------------------------|------------------------------------------------|
| Cost at stage $k$ | $q_k^\top \delta x_k + \frac{1}{2}\delta x_k^\top Q_k\,\delta x_k + \delta u_k^\top S_k\,\delta x_k$ | $Q_k\,\delta x_k + S_k^\top \delta u_k + q_k$ |
| Dynamics constraint $k$: &ensp; $\delta x_k$ enters as $A_k\,\delta x_k$ | $\pi_k^\top A_k\,\delta x_k$ | $+\,A_k^\top \pi_k$ |
| Dynamics constraint $k{-}1$: &ensp; $\delta x_k$ enters as $-\delta x_{(k{-}1)+1}$ | $-\pi_{k-1}^\top \delta x_k$ | $-\,\pi_{k-1}$ |
| Inequalities at stage $k$ | $(\lambda_k^{ug} - \lambda_k^{lg})^\top C_k\,\delta x_k$ | $C_k^\top (\lambda_k^{ug} - \lambda_k^{lg})$ |

Setting $\partial\mathcal{L}/\partial(\delta x_k) = 0$:

$$
\boxed{
  Q_k\,\delta x_k + S_k^\top \delta u_k + q_k
  \;+\; C_k^\top\!\left(\lambda_k^{ug} - \lambda_k^{lg}\right)
  \;+\; A_k^\top \pi_k
  \;-\; \pi_{k-1}
  \;=\; 0
}
$$

with $\pi_{-1} = 0$ (no dynamics before stage 0), and the $A_k^\top \pi_k$ term absent for $k \ge N_u$ (no dynamics leaving the terminal stage).

---

## 4. Stationarity w.r.t. $\delta u_k$ &ensp;($k < N_u$)

| Source | Derivative $\partial\,/\,\partial(\delta u_k)$ |
|--------|------------------------------------------------|
| Cost | $R_k\,\delta u_k + S_k\,\delta x_k + r_k$ |
| Dynamics constraint $k$ | $+\,B_k^\top \pi_k$ |
| Inequalities | $D_k^\top (\lambda_k^{ug} - \lambda_k^{lg})$ |

$$
\boxed{
  R_k\,\delta u_k + S_k\,\delta x_k + r_k
  \;+\; D_k^\top\!\left(\lambda_k^{ug} - \lambda_k^{lg}\right)
  \;+\; B_k^\top \pi_k
  \;=\; 0
}
$$

---

## 5. HPIPM Sign Convention and the Code

HPIPM's `d_ocp_qp_sol_get_pi(k, ...)` returns the multiplier $\hat{\pi}_k$ with sign such that $\hat{\pi}_k = -\pi_k$ relative to the Lagrangian above. Substituting $\pi_k = -\hat{\pi}_k$ into the stationarity condition for $\delta x_k$:

$$
q_k + C_k^\top(\lambda_k^{ug} - \lambda_k^{lg})
\;-\; A_k^\top \hat{\pi}_k
\;+\; \hat{\pi}_{k-1}
\;=\; 0
$$

Re-reading this as a residual with the code's `π[k]` $\equiv \hat{\pi}_k$, and shifting the index perspective so that the dynamics **arriving** at $x_k$ carry index $k{-}1$ and the dynamics **leaving** $x_k$ carry index $k$:

$$
\nabla_{x_k}\mathcal{L} \;=\;
  q_k
  \;+\; C_k^\top\!\left(\lambda_k^{ug} - \lambda_k^{lg}\right)
  \;+\; A_{k-1}^\top\,\pi_{k-1}
  \;-\; \pi_k
$$

This is exactly what the code computes:

```cpp
VectorXd grad_x = _q[k];                                          // q_k
grad_x.noalias() += _C[k].transpose() * (_lam_ug[k] - _lam_lg[k]); // + C_k^T (λ_ug - λ_lg)
if (k > 0)   grad_x.noalias() += _A[k-1].transpose() * _pi[k-1];   // + A_{k-1}^T π_{k-1}
if (k < _Nu) grad_x.noalias() -= _pi[k];                            // - π_k
```

Similarly for the control:

```cpp
VectorXd grad_u = _r[k];                                          // r_k
grad_u.noalias() += _S[k] * _dx[k];                                // + S_k δx_k
grad_u.noalias() += _D[k].transpose() * (_lam_ug[k] - _lam_lg[k]); // + D_k^T (λ_ug - λ_lg)
grad_u.noalias() += _B[k].transpose() * _pi[k];                    // + B_k^T π_k
```

---

## 6. Physical Interpretation

At optimality, each term in $\nabla_{x_k}\mathcal{L} = 0$ has a clear role:

| Term | Role |
|------|------|
| $q_k$ | **Cost gradient** — the direct incentive to change $x_k$ |
| $C_k^\top(\lambda_k^{ug} - \lambda_k^{lg})$ | **Constraint pressure** — active inequalities push $x_k$ away from bounds |
| $A_{k-1}^\top \pi_{k-1}$ | **Backward adjoint** — how the dynamics *arriving* at $x_k$ (from stage $k{-}1$) propagate the co-state backward through $A_{k-1}$ |
| $-\pi_k$ | **Forward co-state** — the price of the dynamics constraint *leaving* $x_k$ (toward stage $k{+}1$); appears with coefficient $-I$ because $\delta x_k$ enters the constraint $k$ as $-\delta x_{k+1}$ from the next stage's perspective |

At $k = 0$ the backward adjoint is absent (no prior dynamics), hence the `if (k > 0)` guard.
At $k \ge N_u$ the forward co-state is absent (no dynamics leave the terminal stage), hence the `if (k < _Nu)` guard.
