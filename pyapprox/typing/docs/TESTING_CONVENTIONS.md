# Testing Conventions for PyApprox Typing Module

This document describes the testing conventions used throughout the `pyapprox.typing` module to ensure consistent, backend-agnostic testing across NumPy and PyTorch implementations.

## Table of Contents

- [Overview](#overview)
- [Backend-Agnostic Testing Pattern](#backend-agnostic-testing-pattern)
- [Excluding Base Classes from Test Discovery](#excluding-base-classes-from-test-discovery)
- [Backend-Only Code in Tests](#backend-only-code-in-tests)
- [Test Structure Examples](#test-structure-examples)
- [Running Tests](#running-tests)
- [Best Practices](#best-practices)

---

## Overview

The PyApprox typing module supports multiple numerical backends (NumPy and PyTorch) through a unified backend protocol interface. To ensure all implementations work correctly with both backends, we use a **Generic Base Class Pattern** where:

1. **Base test class** defines all test logic using backend protocols (Generic over `Array`)
2. **Concrete test classes** inherit from the base and specify the backend (NumPy or Torch)
3. **`__test__ = False`** attribute on base classes prevents them from running
4. **Shared `load_tests`** function provides unittest compatibility

This approach ensures:
- **No code duplication**: Test logic written once, runs on all backends
- **Type safety**: Generic typing ensures backend-agnostic code
- **Automatic coverage**: Adding a new backend only requires one new subclass
- **Clean test output**: No errors from abstract base classes

---

## Backend-Agnostic Testing Pattern

### Basic Structure

```python
import unittest
from typing import Any, Generic

import torch
from numpy.typing import NDArray

from pyapprox.typing.util.backends.numpy import NumpyBkd
from pyapprox.typing.util.backends.torch import TorchBkd
from pyapprox.typing.util.backends.protocols import Array


class TestMyClass(Generic[Array], unittest.TestCase):
    """Base class for MyClass tests - defines test logic."""

    __test__ = False  # Prevents pytest/unittest from running this class

    def bkd(self):
        """Abstract method - must be implemented by subclasses."""
        raise NotImplementedError

    def setUp(self):
        """Initialize backend and test fixtures."""
        self._bkd = self.bkd()
        # Set up test data using self._bkd

    def test_feature_x(self):
        """Test feature X using backend operations."""
        X = self._bkd.array([[1.0, 2.0, 3.0]])
        result = my_function(X)

        # Use backend comparison methods
        expected = self._bkd.array([[2.0, 4.0, 6.0]])
        self.assertTrue(
            self._bkd.allclose(result, expected)
        )


class TestMyClassNumpy(TestMyClass[NDArray[Any]]):
    """Test MyClass with NumPy backend."""

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestMyClassTorch(TestMyClass[torch.Tensor]):
    """Test MyClass with PyTorch backend."""

    def bkd(self) -> TorchBkd:
        return TorchBkd()
```

### Key Principles

1. **Generic Base Class**: Use `Generic[Array]` for the base test class
2. **`__test__ = False`**: Add this attribute to base classes to exclude from discovery
3. **Abstract `bkd()` method**: Forces subclasses to specify backend
4. **Backend operations only**: All array operations go through `self._bkd`
5. **Concrete subclasses**: One per backend, implementing only `bkd()`

---

## Excluding Base Classes from Test Discovery

### Using `__test__ = False` (Primary Method)

The simplest and most reliable way to exclude base test classes is using the `__test__ = False` class attribute:

```python
class TestMyClass(Generic[Array], unittest.TestCase):
    """Base class - will not be run directly."""

    __test__ = False  # This line excludes the class from test discovery

    def bkd(self):
        raise NotImplementedError

    def test_something(self):
        # Test logic here
        pass


class TestMyClassNumpy(TestMyClass[NDArray[Any]]):
    """Concrete class - WILL be run (inherits from base but doesn't have __test__ = False in __dict__)."""

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()
```

### Shared `load_tests` for unittest (Required for unittest discovery)

For unittest compatibility, import the shared `load_tests` function at the end of each test file:

```python
from pyapprox.typing.util.test_utils import load_tests


if __name__ == "__main__":
    unittest.main()
```

The shared `load_tests` function in `pyapprox/typing/util/test_utils.py` handles exclusion:

```python
"""Shared test utilities for the typing module."""

import unittest


def load_tests(loader, tests, pattern):
    """
    Exclude base classes from unittest discovery.

    This function is called automatically by unittest's test discovery.
    We check __dict__ directly to avoid inheritance issues where derived
    classes would inherit __test__ = False from their parent.
    """
    suite = unittest.TestSuite()
    for group in tests:
        for test in group:
            # Check __dict__ directly to avoid inheritance issues
            if test.__class__.__dict__.get('__test__', True):
                suite.addTest(test)
    return suite
```

### pytest Compatibility via conftest.py

The `conftest.py` file in `pyapprox/typing/` handles pytest's test collection:

```python
"""Pytest configuration for typing module tests."""

import unittest


def pytest_pycollect_makeitem(collector, name, obj):
    """
    Custom collection hook to handle __test__ = False inheritance.

    When a base class has __test__ = False, derived classes inherit this
    attribute via normal Python inheritance. This hook detects when a class
    has inherited __test__ = False (i.e., it's not in the class's own __dict__)
    and resets it to True so pytest will collect and run those tests.
    """
    if isinstance(obj, type) and issubclass(obj, unittest.TestCase):
        # If __test__ is not in this class's __dict__ but is False (inherited)
        if '__test__' not in obj.__dict__ and getattr(obj, '__test__', True) is False:
            obj.__test__ = True
    return None
```

### Why This Approach?

- **`__test__ = False`**: Standard pytest convention, works out of the box
- **Shared `load_tests`**: Centralizes unittest exclusion logic, no boilerplate
- **`conftest.py` hook**: Handles inherited `__test__ = False` for derived classes
- **Clean output**: No skipped tests or errors from base classes in either framework

---

## Backend-Only Code in Tests

### DO: Use Backend Methods

```python
# GOOD - uses backend methods
X = self._bkd.array([[1.0, 2.0, 3.0]])
result = self._bkd.sum(X)
expected = self._bkd.array([[6.0]])

self.assertTrue(
    self._bkd.allclose(result, expected)
)
```

### DON'T: Use numpy/torch Directly

```python
# BAD - uses numpy directly
import numpy as np
X = self._bkd.array([[1.0, 2.0, 3.0]])
result = my_function(X)

# This will fail for Torch backend!
np.testing.assert_allclose(
    self._bkd.to_numpy(result),
    np.array([[6.0]])
)
```

### Correct Patterns

#### Array Creation
```python
# Use backend array creation
X = self._bkd.array([[1.0, 2.0, 3.0]])
zeros = self._bkd.zeros((3, 3))
ones = self._bkd.ones((2, 2))
```

#### Comparisons
```python
# Use backend comparison methods
self.assertTrue(self._bkd.allclose(a, b, rtol=1e-6, atol=1e-7))
self.assertTrue(self._bkd.all_bool(values > 0))
```

#### Shape/Type Checks
```python
# Use standard Python/unittest assertions
self.assertEqual(result.shape, (3, 3))
self.assertGreater(value, 0.0)
```

#### Converting to Native Types (when necessary)
```python
# Only convert when absolutely necessary
value_python = self._bkd.to_numpy(value)[0, 0]
self.assertGreater(value_python, 0.5)
```

---

## Test Structure Examples

### Example 1: Testing a Kernel

```python
"""
Tests for MyKernel.
"""

import unittest
from typing import Any, Generic

import torch
from numpy.typing import NDArray

from pyapprox.typing.util.backends.numpy import NumpyBkd
from pyapprox.typing.util.backends.torch import TorchBkd
from pyapprox.typing.util.backends.protocols import Array
from pyapprox.typing.surrogates.kernels.my_kernel import MyKernel
from pyapprox.typing.util.test_utils import load_tests


class TestMyKernel(Generic[Array], unittest.TestCase):
    """Base class for MyKernel tests."""

    __test__ = False

    def bkd(self):
        raise NotImplementedError

    def setUp(self):
        self._bkd = self.bkd()
        self._kernel = MyKernel([1.0, 1.0], (0.1, 10.0), 2, self._bkd)

    def test_kernel_matrix_shape(self):
        """Test kernel matrix has correct shape."""
        X1 = self._bkd.array([[0.0, 1.0], [0.0, 0.0]])
        X2 = self._bkd.array([[0.5], [0.5]])

        K = self._kernel(X1, X2)

        # Shape should be (n1, n2) = (2, 1)
        self.assertEqual(K.shape, (2, 1))

    def test_kernel_symmetry(self):
        """Test that K(X1, X2) = K(X2, X1).T."""
        X1 = self._bkd.array([[0.0, 1.0], [0.0, 0.0]])
        X2 = self._bkd.array([[0.5], [0.5]])

        K12 = self._kernel(X1, X2)
        K21 = self._kernel(X2, X1)

        self.assertTrue(
            self._bkd.allclose(K12, K21.T, rtol=1e-10)
        )

    def test_positive_definite(self):
        """Test kernel is positive definite."""
        X = self._bkd.array([[0.0, 0.5, 1.0], [0.0, 0.0, 0.0]])
        K = self._kernel(X, X)

        # All eigenvalues should be positive
        eigenvalues = self._bkd.eigvalsh(K)
        self.assertTrue(
            self._bkd.all_bool(eigenvalues > -1e-10)
        )


class TestMyKernelNumpy(TestMyKernel[NDArray[Any]]):
    """Test MyKernel with NumPy backend."""

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestMyKernelTorch(TestMyKernel[torch.Tensor]):
    """Test MyKernel with PyTorch backend."""

    def bkd(self) -> TorchBkd:
        return TorchBkd()


if __name__ == "__main__":
    unittest.main()
```

---

## Running Tests

### With pytest (Recommended)

```bash
# Run all typing module tests
pytest pyapprox/typing -v

# Run tests with short traceback
pytest pyapprox/typing --tb=short -q

# Run specific test file
pytest pyapprox/typing/surrogates/kernels/tests/test_matern.py -v

# Run specific test class
pytest pyapprox/typing/surrogates/kernels/tests/test_matern.py::TestMaternKernelNumpy -v

# Run specific test
pytest pyapprox/typing/surrogates/kernels/tests/test_matern.py::TestMaternKernelNumpy::test_kernel_symmetry
```

### With unittest

```bash
# Run all tests in a module
python -m unittest pyapprox.typing.surrogates.kernels.tests.test_matern

# Run specific test class
python -m unittest pyapprox.typing.surrogates.kernels.tests.test_matern.TestMaternKernelNumpy

# Run from file directly
python pyapprox/typing/surrogates/kernels/tests/test_matern.py
```

### With conda environment

```bash
# Run with conda environment
conda run -n linalg python -m pytest pyapprox/typing -v
```

### Expected Output

```
$ pytest pyapprox/typing --tb=no -q
..........................................................................
624 passed in 6.42s
```

---

## Best Practices

### 1. Always Use Backend Methods

Bad:
```python
import numpy as np
result = np.sum(array)
```

Good:
```python
result = self._bkd.sum(array)
```

### 2. Never Import numpy/torch Directly in Tests

Bad:
```python
import numpy as np
import torch
```

Good:
```python
import torch  # Only for type annotations
from numpy.typing import NDArray  # Only for type annotations
```

The imports should only be used for:
- Type annotations: `NDArray[Any]`, `torch.Tensor`
- Backend class imports: `NumpyBkd`, `TorchBkd`

### 3. Use Appropriate Tolerances

Different backends have different precision:
- **NumPy**: float64 by default (high precision)
- **PyTorch**: float64 when set (we use `torch.set_default_dtype(torch.float64)`)

Use relaxed tolerances for cross-backend compatibility:

```python
# Good - relaxed tolerance
self.assertTrue(
    self._bkd.allclose(result, expected, rtol=1e-6, atol=1e-7)
)

# Too strict - may fail intermittently
self.assertTrue(
    self._bkd.allclose(result, expected, rtol=1e-10)
)
```

### 4. Test Backend Consistency

Always verify backend is propagated correctly:

```python
def test_backend_consistency(self):
    """Test that backend is correctly propagated."""
    obj = MyClass(self._bkd)
    self.assertIs(obj.bkd(), self._bkd)
```

### 5. Document Test Classes

Provide clear docstrings:

```python
class TestMyKernel(Generic[Array], unittest.TestCase):
    """
    Base class for MyKernel tests.

    Tests kernel evaluation, symmetry, positive definiteness,
    and Jacobian computations across multiple backends.
    """
```

### 6. One Test File Per Class/Module

Follow this structure:
```
tests/
├── test_matern.py          # Tests MaternKernel classes
├── test_scalings.py        # Tests PolynomialScaling
├── test_composition.py     # Tests ProductKernel, SumKernel
└── test_plot.py           # Tests plotting wrappers
```

### 7. Use Descriptive Test Names

```python
# Good - clear what is being tested
def test_kernel_matrix_shape(self):
def test_kernel_symmetry(self):
def test_constant_scaling_jacobian(self):

# Bad - unclear what is being tested
def test_kernel(self):
def test_1(self):
```

### 8. Include Type Annotations

```python
class TestMyKernelNumpy(TestMyKernel[NDArray[Any]]):
    """Test MyKernel with NumPy backend."""

    def bkd(self) -> NumpyBkd:  # Return type annotation
        return NumpyBkd()
```

---

## Common Backend Methods

Here are the most commonly used backend protocol methods:

### Array Creation
```python
self._bkd.array([[1.0, 2.0]])
self._bkd.zeros((3, 3))
self._bkd.ones((2, 2))
self._bkd.eye(5)
self._bkd.linspace(0.0, 1.0, 100)
```

### Array Operations
```python
self._bkd.sum(array)
self._bkd.mean(array)
self._bkd.std(array)
self._bkd.maximum(a, b)
self._bkd.minimum(a, b)
self._bkd.sqrt(array)
self._bkd.exp(array)
self._bkd.log(array)
```

### Linear Algebra
```python
self._bkd.matmul(A, B)
self._bkd.cholesky(K)
self._bkd.solve_triangular(L, y, lower=True)
self._bkd.eigvalsh(K)
```

### Comparison
```python
self._bkd.allclose(a, b, rtol=1e-6, atol=1e-7)
self._bkd.all_bool(array > 0)
self._bkd.any_bool(array < 0)
```

### Array Manipulation
```python
self._bkd.reshape(array, (3, 3))
self._bkd.hstack([a, b])
self._bkd.vstack([a, b])
self._bkd.flatten(array)
self._bkd.tile(array, (2, 3))
```

### Conversion
```python
self._bkd.to_numpy(torch_array)  # Convert to numpy for assertions
```

---

## File Template

Use this template when creating new test files:

```python
"""
Tests for [ClassName].
"""

import unittest
from typing import Any, Generic

import torch
from numpy.typing import NDArray

from pyapprox.typing.util.backends.numpy import NumpyBkd
from pyapprox.typing.util.backends.torch import TorchBkd
from pyapprox.typing.util.backends.protocols import Array
from pyapprox.typing.[module.path] import [ClassName]
from pyapprox.typing.util.test_utils import load_tests


class Test[ClassName](Generic[Array], unittest.TestCase):
    """Base class for [ClassName] tests."""

    __test__ = False

    def bkd(self):
        raise NotImplementedError

    def setUp(self):
        self._bkd = self.bkd()
        # Initialize test fixtures

    def test_feature_x(self):
        """Test feature X."""
        # Test implementation using self._bkd


class Test[ClassName]Numpy(Test[ClassName][NDArray[Any]]):
    """Test [ClassName] with NumPy backend."""

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class Test[ClassName]Torch(Test[ClassName][torch.Tensor]):
    """Test [ClassName] with PyTorch backend."""

    def bkd(self) -> TorchBkd:
        return TorchBkd()


if __name__ == "__main__":
    unittest.main()
```

---

## Summary

**Key Takeaways**:

1. Use Generic base class with `Generic[Array]`
2. Add `__test__ = False` to base classes to exclude them from discovery
3. Import shared `load_tests` from `pyapprox.typing.util.test_utils`
4. Use backend methods exclusively (never numpy/torch directly)
5. Import torch unconditionally (assumed installed)
6. Run with `pytest` or `unittest` - both are supported
7. Use relaxed tolerances (1e-6) for cross-backend compatibility
8. Test with both NumPy and Torch backends
9. One concrete test class per backend

Following these conventions ensures consistent, maintainable, backend-agnostic testing throughout the PyApprox typing module.
