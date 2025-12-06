# Convention for Optional Methods (jacobian, hvp)

This document describes the convention for handling optional derivative methods
(`jacobian`, `hvp`, etc.) in the pyapprox typing module.

## Overview

The module uses a **hybrid approach** that balances type safety, runtime
flexibility, and API simplicity:

- **Dynamic Method Binding** for runtime-determined capabilities
- **Protocol Hierarchy** for static type checking at API boundaries
- **Factory Functions with `@overload`** for type-safe construction
- **`hasattr()` Runtime Checks** at consumption points

## When to Use Each Pattern

| Scenario | Pattern | Example |
|----------|---------|---------|
| Capabilities known at compile time | Protocol Hierarchy + Factory | `NumpyFunctionWrapper` |
| Capabilities depend on runtime inputs | Dynamic Binding | `NegativeLogMarginalLikelihoodLoss` |
| User-facing API boundary | Protocol as parameter type | Optimizer `objective` parameter |
| Optimizer consuming objective | `hasattr()` checks | `ScipyTrustConstrOptimizer` |
| Composing objects (kernels) | Dynamic Binding + AND logic | `CompositionKernel` |

### Decision Rules

1. **If capability is inherent to the class** (e.g., `MaternKernel` always has
   `jacobian` for `nu=inf`) → Define method normally on the class

2. **If capability depends on composed/wrapped objects** (e.g., `ProductKernel`
   depends on both component kernels) → Use dynamic binding with
   `_setup_derivative_methods()`

3. **If wrapping for external API** (e.g., NumPy wrapper for scipy) → Use
   protocol hierarchy + factory function

## Pattern 1: Dynamic Method Binding

Use when capabilities depend on runtime inputs (e.g., kernel capabilities).

```python
class MyLoss:
    def __init__(self, component):
        self._component = component
        self._setup_derivative_methods()

    def _setup_derivative_methods(self) -> None:
        """Conditionally expose methods based on component capabilities."""
        if hasattr(self._component, 'jacobian_wrt_params'):
            self.jacobian = self._jacobian
        if hasattr(self._component, 'hvp_wrt_params'):
            self.hvp = self._hvp

    def _jacobian(self, params):
        """Private implementation."""
        ...

    def _hvp(self, params, direction):
        """Private implementation."""
        ...
```

**Key points:**
- Private methods (`_jacobian`, `_hvp`) contain the implementation
- Public methods are conditionally assigned in `_setup_derivative_methods()`
- Consumers check with `hasattr(obj, 'method')`

## Pattern 2: AND Logic for Composition

When composing multiple objects, the composed object only has a capability
if **ALL** components have it.

```python
def _setup_derivative_methods(self) -> None:
    has_jac_1 = hasattr(self._kernel1, 'jacobian_wrt_params')
    has_jac_2 = hasattr(self._kernel2, 'jacobian_wrt_params')

    if has_jac_1 and has_jac_2:
        self.jacobian_wrt_params = self._jacobian_wrt_params
    # Otherwise, method will not exist on this instance
```

**Rationale:** This ensures derivative methods are only available when they
can be correctly computed for the entire composition.

## Pattern 3: Protocol Hierarchy

Use for type-safe API boundaries. Protocols are defined in a hierarchy:

```
FunctionProtocol (base: __call__, nvars, nqoi, bkd)
    └── FunctionWithJacobianProtocol (adds: jacobian)
            └── FunctionWithJacobianAndHVPProtocol (adds: hvp)
            └── FunctionWithJacobianAndWHVPProtocol (adds: whvp)
```

All protocols must be `@runtime_checkable` to support `isinstance()` checks.

**Location:** `pyapprox/typing/interface/functions/protocols/`

## Pattern 4: Factory with @overload

Use when creating wrappers that need type-safe dispatch:

```python
from typing import overload

@overload
def factory(fn: FunctionWithJacobianAndHVPProtocol[Array]) -> WrapperWithHVP[Array]: ...
@overload
def factory(fn: FunctionWithJacobianProtocol[Array]) -> WrapperWithJac[Array]: ...
@overload
def factory(fn: FunctionProtocol[Array]) -> Wrapper[Array]: ...

def factory(fn):
    # Check in order of specificity (most specific first)
    if isinstance(fn, FunctionWithJacobianAndHVPProtocol):
        return WrapperWithHVP(fn)
    if isinstance(fn, FunctionWithJacobianProtocol):
        return WrapperWithJac(fn)
    if isinstance(fn, FunctionProtocol):
        return Wrapper(fn)
    raise TypeError(...)
```

**Location:** `pyapprox/typing/interface/functions/numpy/numpy_function_factory.py`

## Pattern 5: Optimizer Consumption

Optimizers accept the base protocol and use `hasattr()` at runtime:

```python
class Optimizer:
    def __init__(self, objective: ObjectiveProtocol[Array]):
        self._objective = objective

    def minimize(self, init_guess):
        jac = self._jac_callback if hasattr(self._objective, "jacobian") else None
        hessp = self._hvp_callback if hasattr(self._objective, "hvp") else None
        # Pass to underlying optimizer (scipy, etc.)
```

**Important:** No finite difference fallback. Users choose the optimizer based
on available derivatives.

## Naming Conventions

| Item | Convention | Example |
|------|------------|---------|
| Private implementation | `_method` | `_jacobian`, `_hvp` |
| Public method | `method` | `jacobian`, `hvp` |
| Capability check | `hasattr(obj, 'method')` | `hasattr(loss, 'jacobian')` |
| Base protocol | `XxxProtocol` | `FunctionProtocol` |
| Extended protocol | `XxxWith{Cap}Protocol` | `FunctionWithJacobianProtocol` |
| Setup method | `_setup_derivative_methods()` | (standardized name) |

## Documentation Convention

Classes with optional methods must document them in the docstring:

```python
class MyLoss:
    """
    Loss function for optimization.

    Optional Methods
    ----------------
    The following methods are conditionally available:

    - ``jacobian(params)``: Available if kernel has ``jacobian_wrt_params``
    - ``hvp(params, direction)``: Available if kernel has ``hvp_wrt_params``

    Check availability with ``hasattr(loss, 'jacobian')`` / ``hasattr(loss, 'hvp')``.

    Notes
    -----
    This class follows the dynamic binding pattern for optional methods.
    See docs/OPTIONAL_METHODS_CONVENTION.md for details.
    """
```

## Trade-offs

| Aspect | This Convention | Alternative |
|--------|-----------------|-------------|
| Type safety | Medium-High (protocols at boundaries) | Could be higher with wrapper factories everywhere |
| Runtime flexibility | High (dynamic binding works) | Lower if requiring specific types |
| API simplicity | High (single class per concept) | Lower with many wrapper classes |
| Discoverability | Good (docstrings document pattern) | Could add IDE hints |

The hybrid approach optimizes for the common case (dynamic composition) while
preserving type safety at API boundaries where it matters most.

## Key Files

- `pyapprox/typing/interface/functions/protocols/` - Protocol definitions
- `pyapprox/typing/interface/functions/numpy/numpy_function_factory.py` - Factory example
- `pyapprox/typing/surrogates/gaussianprocess/loss.py` - Dynamic binding example
- `pyapprox/typing/surrogates/kernels/composition.py` - AND logic example
- `pyapprox/typing/optimization/minimize/scipy/trust_constr.py` - Consumer example
