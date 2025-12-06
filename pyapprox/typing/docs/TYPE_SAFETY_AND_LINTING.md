# Type Safety and Linting Conventions

This document describes type safety and code quality standards for the `pyapprox.typing` module.

## Table of Contents

1. [Overview](#overview)
2. [Type Annotations](#type-annotations)
3. [Generic Types](#generic-types)
4. [Protocol-Based Interfaces](#protocol-based-interfaces)
5. [Type Checking with mypy](#type-checking-with-mypy)
6. [Linting Standards](#linting-standards)
7. [Common Patterns](#common-patterns)
8. [Examples](#examples)

---

## Overview

The `pyapprox.typing` module enforces strict type safety standards to ensure:
- **Correctness**: Catch type errors at development time, not runtime
- **Clarity**: Make interfaces and contracts explicit
- **Maintainability**: Enable safe refactoring and code evolution
- **IDE Support**: Provide autocomplete and inline documentation

**Key Principle**: Every function, method, and class in `pyapprox.typing` must have complete type annotations.

---

## Type Annotations

### Required Annotations

**All** of the following must have type annotations:
1. Function parameters (except `self` and `cls`)
2. Function return types
3. Class attributes (when not inferred from `__init__`)
4. Generic type parameters

### Examples

#### Function Annotations
```python
# ✅ GOOD - complete annotations
def compute_kernel_matrix(
    kernel: KernelProtocol[Array],
    X: Array,
    *,
    return_gradient: bool = False
) -> Array:
    """Compute kernel matrix K(X, X)."""
    ...

# ❌ BAD - missing annotations
def compute_kernel_matrix(kernel, X, return_gradient=False):
    ...
```

#### Class Annotations
```python
# ✅ GOOD - annotated attributes
class MaternKernel(Kernel[Array], Generic[Array]):
    _nu: float
    _nvars: int
    _log_lenscale: LogHyperParameter
    _bkd: Backend[Array]

    def __init__(
        self,
        nu: float,
        length_scale: List[float],
        bounds: Tuple[float, float],
        nvars: int,
        bkd: Backend[Array]
    ) -> None:
        ...

# ❌ BAD - no annotations
class MaternKernel:
    def __init__(self, nu, length_scale, bounds, nvars, bkd):
        self._nu = nu
        ...
```

### Type Hint Imports

Always import type hints from `typing`:

```python
from typing import (
    Generic,
    List,
    Tuple,
    Optional,
    Union,
    Protocol,
    runtime_checkable,
)
```

For array types, use protocols:

```python
from pyapprox.typing.util.backends.protocols import Array, Backend
```

---

## Generic Types

### Generic Classes

Use `Generic[Array]` for backend-agnostic classes:

```python
from typing import Generic
from pyapprox.typing.util.backends.protocols import Array, Backend

class MyKernel(Generic[Array]):
    """Backend-agnostic kernel implementation."""

    def __init__(self, bkd: Backend[Array]) -> None:
        self._bkd = bkd

    def __call__(self, X: Array) -> Array:
        # Works with both NumPy and PyTorch
        return self._bkd.matmul(X, X.T)
```

### Generic Type Variables

Define type variables for protocols:

```python
from typing import TypeVar, Generic

Array = TypeVar('Array')  # Array type (np.ndarray or torch.Tensor)

class Kernel(Protocol, Generic[Array]):
    def __call__(self, X1: Array, X2: Array) -> Array:
        ...
```

### Constraints

Use type variable constraints when needed:

```python
from typing import TypeVar
import numpy as np
import torch

ArrayType = TypeVar('ArrayType', np.ndarray, torch.Tensor)

def process_array(arr: ArrayType) -> ArrayType:
    # Only accepts NumPy or PyTorch arrays
    ...
```

---

## Protocol-Based Interfaces

### Defining Protocols

Use `@runtime_checkable` for protocols that may be checked at runtime:

```python
from typing import Protocol, runtime_checkable, Generic

@runtime_checkable
class KernelProtocol(Protocol, Generic[Array]):
    """Protocol for kernel implementations."""

    def __call__(self, X1: Array, X2: Array | None = None) -> Array:
        """Evaluate kernel matrix."""
        ...

    def nvars(self) -> int:
        """Number of input variables."""
        ...

    def bkd(self) -> Backend[Array]:
        """Return backend."""
        ...
```

### Protocol Composition

Compose protocols for richer interfaces:

```python
class KernelWithJacobianProtocol(
    KernelProtocol[Array],
    KernelHasJacobianProtocol[Array],
    Protocol
):
    """Protocol for kernels with Jacobian support."""
    pass  # Inherits all methods from both protocols
```

### Why Protocols?

**Protocols** (PEP 544) enable structural subtyping:
- Classes don't need to explicitly inherit
- Duck typing with type safety
- Easier to retrofit existing code
- More flexible than ABC (Abstract Base Classes)

---

## Type Checking with mypy

### Running mypy

Type check the typing module with mypy:

```bash
# Check specific file
mypy pyapprox/typing/surrogates/kernels/matern.py

# Check entire module
mypy pyapprox/typing/

# Check with strict mode
mypy --strict pyapprox/typing/surrogates/kernels/
```

### mypy Configuration

Create `pyproject.toml` or `mypy.ini`:

```toml
[tool.mypy]
python_version = "3.9"
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = true
disallow_incomplete_defs = true
check_untyped_defs = true
disallow_untyped_calls = true
warn_redundant_casts = true
warn_unused_ignores = true
strict_equality = true
```

### Common mypy Errors and Fixes

#### Error: Missing return type
```python
# ❌ BAD
def compute_value(x):
    return x * 2

# ✅ GOOD
def compute_value(x: float) -> float:
    return x * 2
```

#### Error: Incompatible types
```python
# ❌ BAD
def process(arr: Array) -> int:
    return arr  # Array is not int!

# ✅ GOOD
def process(arr: Array) -> Array:
    return arr
```

#### Error: Need type annotation for generic
```python
# ❌ BAD
class MyKernel(Kernel):
    ...

# ✅ GOOD
class MyKernel(Kernel[Array], Generic[Array]):
    ...
```

### Type Ignores (Last Resort)

Use `# type: ignore` sparingly and with comments:

```python
# Sometimes unavoidable with external libraries
result = scipy.optimize.minimize(...)  # type: ignore[attr-defined]  # scipy stubs incomplete
```

---

## Linting Standards

### Tools

Use these linting tools on `pyapprox.typing`:

1. **mypy**: Type checking
2. **ruff**: Fast Python linter (replaces flake8, isort, etc.)
3. **black**: Code formatting (optional, for consistency)

### Running Linters

```bash
# Type check
mypy pyapprox/typing/

# Lint code
ruff check pyapprox/typing/

# Format code (optional)
black pyapprox/typing/
```

### ruff Configuration

Create `pyproject.toml`:

```toml
[tool.ruff]
line-length = 88
target-version = "py39"

[tool.ruff.lint]
select = [
    "E",   # pycodestyle errors
    "W",   # pycodestyle warnings
    "F",   # pyflakes
    "I",   # isort
    "N",   # pep8-naming
    "UP",  # pyupgrade
    "B",   # flake8-bugbear
    "C4",  # flake8-comprehensions
]
ignore = [
    "E501",  # line too long (handled by formatter)
]

[tool.ruff.lint.per-file-ignores]
"__init__.py" = ["F401"]  # Allow unused imports in __init__.py
```

### Code Style Requirements

1. **Line length**: Maximum 88 characters (black default)
2. **Imports**: Organized with isort-compatible grouping:
   ```python
   # Standard library
   import math
   from typing import Generic, List

   # Third-party
   import numpy as np
   import torch

   # Local imports
   from pyapprox.typing.util.backends.protocols import Array, Backend
   ```

3. **Naming conventions**:
   - Classes: `PascalCase`
   - Functions/methods: `snake_case`
   - Constants: `UPPER_SNAKE_CASE`
   - Private attributes: `_leading_underscore`

4. **Docstrings**: NumPy-style docstrings for all public APIs:
   ```python
   def compute_kernel(X: Array) -> Array:
       """
       Compute kernel matrix.

       Parameters
       ----------
       X : Array
           Input data, shape (nvars, nsamples).

       Returns
       -------
       K : Array
           Kernel matrix, shape (nsamples, nsamples).
       """
       ...
   ```

---

## Common Patterns

### Pattern 1: Backend-Agnostic Functions

```python
from typing import Generic
from pyapprox.typing.util.backends.protocols import Array, Backend

class MyClass(Generic[Array]):
    def __init__(self, bkd: Backend[Array]) -> None:
        self._bkd = bkd

    def process(self, X: Array) -> Array:
        # Use backend methods
        return self._bkd.matmul(X, X.T)
```

### Pattern 2: Protocol Implementation

```python
from typing import Generic
from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.interface.functions.protocols import FunctionProtocol

class MyFunction(FunctionProtocol[Array], Generic[Array]):
    """Implements FunctionProtocol."""

    def __init__(self, nvars: int, nqoi: int, bkd: Backend[Array]) -> None:
        self._nvars = nvars
        self._nqoi = nqoi
        self._bkd = bkd

    def nvars(self) -> int:
        return self._nvars

    def nqoi(self) -> int:
        return self._nqoi

    def bkd(self) -> Backend[Array]:
        return self._bkd

    def __call__(self, X: Array) -> Array:
        # Implementation
        ...
```

### Pattern 3: Type Guards

Use type guards for runtime checks:

```python
from typing import TypeGuard

def is_numpy_array(arr: Array) -> TypeGuard[np.ndarray]:
    """Check if array is NumPy."""
    return isinstance(arr, np.ndarray)

def process(arr: Array) -> None:
    if is_numpy_array(arr):
        # mypy knows arr is np.ndarray here
        print(arr.dtype)
```

### Pattern 4: Union Types for Flexibility

```python
from typing import Union

def get_bounds(hyp: HyperParameter) -> Union[Array, Tuple[float, float]]:
    """Return bounds as array or tuple."""
    ...
```

### Pattern 5: Optional Parameters

```python
from typing import Optional

def kernel_matrix(
    X1: Array,
    X2: Optional[Array] = None
) -> Array:
    """Compute K(X1, X2). If X2 is None, compute K(X1, X1)."""
    if X2 is None:
        X2 = X1
    ...
```

---

## Examples

### Example 1: Type-Safe Kernel

```python
from typing import Generic, Tuple
from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.surrogates.kernels.protocols import (
    KernelWithJacobianAndParameterJacobianProtocol
)
from pyapprox.typing.util.hyperparameter import HyperParameterList

class MyKernel(
    KernelWithJacobianAndParameterJacobianProtocol[Array],
    Generic[Array]
):
    """
    Type-safe kernel implementation.

    Parameters
    ----------
    nvars : int
        Number of input variables.
    bkd : Backend[Array]
        Backend for array operations.
    """

    def __init__(self, nvars: int, bkd: Backend[Array]) -> None:
        self._nvars = nvars
        self._bkd = bkd
        self._hyp_list = HyperParameterList(bkd)

    def __call__(
        self,
        X1: Array,
        X2: Array | None = None
    ) -> Array:
        """
        Evaluate kernel matrix.

        Parameters
        ----------
        X1 : Array
            First input, shape (nvars, n1).
        X2 : Array, optional
            Second input, shape (nvars, n2). If None, use X1.

        Returns
        -------
        K : Array
            Kernel matrix, shape (n1, n2).
        """
        if X2 is None:
            X2 = X1
        # Implementation
        ...

    def jacobian(self, X1: Array, X2: Array) -> Array:
        """
        Compute spatial Jacobian.

        Returns
        -------
        jac : Array
            Jacobian, shape (n1, n2, nvars).
        """
        ...

    def jacobian_wrt_params(self, X: Array) -> Array:
        """
        Compute parameter Jacobian.

        Returns
        -------
        jac : Array
            Jacobian, shape (n, n, nparams).
        """
        ...

    def nvars(self) -> int:
        return self._nvars

    def bkd(self) -> Backend[Array]:
        return self._bkd

    def hyp_list(self) -> HyperParameterList:
        return self._hyp_list

    def diag(self, X: Array) -> Array:
        """
        Compute diagonal of kernel matrix.

        Returns
        -------
        diag : Array
            Diagonal values, shape (nsamples,).
        """
        ...
```

### Example 2: Type-Safe Testing

```python
from typing import Generic
from pyapprox.typing.util.backends.protocols import Array, Backend

class TestMyKernel(Generic[Array], unittest.TestCase):
    """Backend-agnostic kernel tests."""

    def bkd(self) -> Backend[Array]:
        """Override in subclasses."""
        raise NotImplementedError

    def setUp(self) -> None:
        self._bkd = self.bkd()
        self._kernel = MyKernel(nvars=2, bkd=self._bkd)

    def test_kernel_matrix(self) -> None:
        """Test kernel matrix computation."""
        X: Array = self._bkd.random.randn(2, 10)
        K: Array = self._kernel(X)

        # Type-safe assertions
        self.assertEqual(K.shape, (10, 10))

class TestMyKernelNumpy(TestMyKernel[np.ndarray]):
    """NumPy backend tests."""

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()

class TestMyKernelTorch(TestMyKernel[torch.Tensor]):
    """PyTorch backend tests."""

    def bkd(self) -> TorchBkd:
        return TorchBkd()
```

---

## Summary

**Type Safety Checklist**:
- ✅ All functions have parameter and return type annotations
- ✅ Generic classes use `Generic[Array]`
- ✅ Protocols define interface contracts
- ✅ Code passes `mypy --strict` without errors
- ✅ Code passes `ruff check` without warnings
- ✅ NumPy-style docstrings for all public APIs
- ✅ Consistent naming conventions
- ✅ Type ignores are minimal and documented

**Benefits**:
- Catch bugs before runtime
- Better IDE autocomplete
- Easier refactoring
- Self-documenting code
- Backend portability (NumPy ↔ PyTorch)
