# Future Performance Optimizations (5-11)

This document contains medium and low priority performance optimizations to be implemented after the quick wins (optimizations 1-4).

---

## OPTIMIZATION 5: Devirtualize Cost/Constraint Hot Path [MEDIUM-HIGH]

**Expected Speedup:** 5-10%
**Effort:** High
**Priority:** MEDIUM

### Issue
Every cost and constraint evaluation uses virtual function calls (1,750 per iteration).

### Recommended Fix Options

**Option A: Mark leaf classes `final`** (Easier)
```cpp
class VelocityCost final : public AbstractCost {
    // Compiler can devirtualize calls
};
```

**Option B: CRTP** (More work, better results)
```cpp
template<typename Derived>
class AbstractCostCRTP {
    void evaluate() { static_cast<Derived*>(this)->evaluate_impl(); }
};
```

**Option C: std::variant** (C++17)
Store as `std::variant<VelocityCost, AccelerationCost, ...>` with `std::visit`.

---

## OPTIMIZATION 6: Pre-allocate Temporary Vectors in Constraint Assembly [MEDIUM]

**Expected Speedup:** 5-8%
**Effort:** Low
**Priority:** MEDIUM

### Issue
500 temporary vectors created per iteration in constraint bound computation.

### Fix
```cpp
// Pre-allocate workspace outside loop
VectorXd temp_bound(_max_constraint_dim);

for (auto& con : _ocproblem.get_node(i).get_constraints()) {
    const VectorXd& residual = con->get_value();

    temp_bound.head(nc) = con->get_lower_bound();
    temp_bound.head(nc) -= residual;
    _lg[i].segment(row, nc) = temp_bound.head(nc);

    temp_bound.head(nc) = con->get_upper_bound();
    temp_bound.head(nc) -= residual;
    _ug[i].segment(row, nc) = temp_bound.head(nc);
}
```

**File:** `/workspace/code/src/sqp_solver/sqp_solver.cpp:362-382`

---

## OPTIMIZATION 7: Optimize Constraint Violation Loop [MEDIUM]

**Expected Speedup:** 5-8%
**Effort:** Medium
**Priority:** MEDIUM

### Issue
1,250 temporary allocations per violation check.

### Fix
```cpp
_violation = 0.;
VectorXd lower_viol, upper_viol;  // Pre-allocate

for (auto& constraint: _constraint_list) {
    const VectorXd& val = constraint->get_value();
    const int dim = val.size();

    lower_viol.resize(dim);
    upper_viol.resize(dim);

    lower_viol = constraint->get_lower_bound() - val;
    upper_viol = val - constraint->get_upper_bound();

    for (int i = 0; i < dim; ++i) {
        _violation = std::max(_violation, std::max(lower_viol(i), upper_viol(i)));
    }
}
```

**File:** `/workspace/code/src/trajopt/node.cpp:197-205`

---

## OPTIMIZATION 8: Cache Forward Kinematics Derivatives [MEDIUM]

**Expected Speedup:** 3-5%
**Effort:** Medium
**Priority:** MEDIUM

### Issue
Contact constraints recompute expensive Pinocchio FK derivatives.

### Fix

**Option A:** Extend Node caching
```cpp
// In Node class:
void require_fk_derivatives() {
    if (!(_cache_flags & CACHE_FK_DERIV)) {
        pinocchio::computeForwardKinematicsDerivatives(_model_ptr, _data_ptr, ...);
        _cache_flags |= CACHE_FK_DERIV;
    }
}

// In ContactConstraint::jacobian_impl():
_node->require_fk_derivatives();
```

**Files:**
- `/workspace/code/include/trajopt/node.h`
- `/workspace/code/src/trajopt/constraints/contact_constraint.cpp:78`
- `/workspace/code/src/trajopt/constraints/friction_cone_constraint.cpp:57`

---

## OPTIMIZATION 9: Fix Sequence Copy in ContactScheduler [LOW-MEDIUM]

**Expected Speedup:** 2-3% (if in hot path)
**Effort:** Low
**Priority:** LOW

### Fix
```cpp
// Change from:
Sequence _seq = _sequences[sequence_name];

// To:
const Sequence& _seq = _sequences[sequence_name];
```

**File:** `/workspace/code/src/trajopt/scheduler.cpp:35`

---

## OPTIMIZATION 10: Optimize Vector Insert in ContactScheduler [LOW]

**Expected Speedup:** 1-2%
**Effort:** Low
**Priority:** LOW

### Fix
```cpp
// Pre-calculate total size
size_t total_size = 0;
for (const auto& contact_name : contacts_list) {
    total_size += _contacts[contact_name].size();
}
phase.contact_frames.reserve(total_size);

// Now insert without reallocation
for (const auto& contact_name : contacts_list) {
    const auto& frames = _contacts[contact_name];
    phase.contact_frames.insert(phase.contact_frames.end(), frames.begin(), frames.end());
}
```

**File:** `/workspace/code/src/trajopt/scheduler.cpp:18-21`

---

## OPTIMIZATION 11: Redundant External Force Build in InvDynamics [LOW]

**Expected Speedup:** 1-2%
**Effort:** Low
**Priority:** LOW

### Recommendation
Profile to confirm overhead, then cache `fext` structure across evaluate/jacobian calls.

**File:** `/workspace/code/src/trajopt/constraints/inverse_dynamics.cpp:87-99`

---

## BONUS: Reduce Header Include Bloat

**Impact:** 40% faster compilation (not runtime)
**Effort:** Medium

### Move Pinocchio includes from headers to .cpp files

**File:** `/workspace/code/include/trajopt/node.h:10-15`

Use forward declarations instead of including full headers.

---

## Implementation Priority

After completing optimizations 1-4:

1. **Optimization 6** (Low effort, good gains)
2. **Optimization 7** (Complements opt 2)
3. **Optimization 8** (Medium effort)
4. **Optimization 5** (High effort, consider if further gains needed)
5. **Optimizations 9-11** (Low priority cleanup)
