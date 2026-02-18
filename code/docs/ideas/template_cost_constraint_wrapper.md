# Idea: Template wrapper for Cost / Constraint from AbstractFunction

Instead of writing separate cost and constraint wrapper classes for each function,
use a single template:

```cpp
template <typename Base>
class FunctionWrapper : public Base {
    // Owns an AbstractFunction-derived computation object
    // Delegates allocate_dims, evaluate_impl, jacobian_impl
};

using FrameTranslationCost = FunctionWrapper<AbstractCost>;
using FrameTranslationConstraint = FunctionWrapper<AbstractConstraint>;
```

## Challenge
- Constructor forwarding: AbstractCost takes weight, AbstractConstraint doesn't.
  Could use variadic template forwarding or `if constexpr`.

## Benefit
- Write the delegation logic once, reuse for every function that can be both
  a cost and a constraint.
