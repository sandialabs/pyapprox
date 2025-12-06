# Testing Conventions for PyApprox Typing Module

This document describes the testing conventions used throughout the `pyapprox.typing` module to ensure consistent, backend-agnostic testing across NumPy and PyTorch implementations.

## Table of Contents

- [Overview](#overview)
- [Backend-Agnostic Testing Pattern](#backend-agnostic-testing-pattern)
- [Using load_tests to Skip Base Classes](#using-load_tests-to-skip-base-classes)
- [Backend-Only Code in Tests](#backend-only-code-in-tests)
- [Test Structure Examples](#test-structure-examples)
- [Running Tests](#running-tests)
- [Best Practices](#best-practices)

---

## Overview

The PyApprox typing module supports multiple numerical backends (NumPy and PyTorch) through a unified backend protocol interface. To ensure all implementations work correctly with both backends, we use a **Generic Base Class Pattern** where:

1. **Base test class** defines all test logic using backend protocols (Generic over `Array`)
2. **Concrete test classes** inherit from the base and specify the backend (NumPy or Torch)
3. **`load_tests` function** ensures only concrete classes run (skips base class)

This approach ensures:
- **No code duplication**: Test logic written once, runs on all backends
- **Type safety**: Generic typing ensures backend-agnostic code
- **Automatic coverage**: Adding a new backend only requires one new subclass

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
2. **Abstract `bkd()` method**: Forces subclasses to specify backend
3. **Backend operations only**: All array operations go through `self._bkd`
4. **Concrete subclasses**: One per backend, implementing only `bkd()`

---

## Using load_tests to Skip Base Classes

When using `unittest` (recommended), add a `load_tests` function to skip base class tests:

```python
def load_tests(loader, tests, pattern):
    """
    Custom test loader to skip base class tests.

    Only run tests from concrete implementation classes (Numpy, Torch),
    not from the generic base classes.
    """
    suite = unittest.TestSuite()

    # Only add tests from concrete Numpy and Torch classes
    for test in tests:
        for test_case in test:
            test_class_name = test_case.__class__.__name__
            if 'Numpy' in test_class_name or 'Torch' in test_class_name:
                suite.addTest(test_case)

    return suite
```

### Why `load_tests`?

- **Prevents base class execution**: Base classes have `NotImplementedError` in `bkd()`
- **Works with unittest**: Standard Python unittest discovery
- **Clean output**: No skipped tests or errors from base classes
- **Automatic**: No need to manually skip in `setUp()`

### Alternative Approaches (Not Recommended)

❌ **Using `skipTest` in `setUp()`**:
```python
def setUp(self):
    if self.__class__.__name__ == 'TestMyClass':
        self.skipTest("Base class - tests only run on Numpy/Torch subclasses")
    # ... rest of setUp
```
- **Problem**: Shows skipped tests in output (clutter)
- **Problem**: Requires boilerplate in every base class

❌ **Using pytest fixtures**:
- **Problem**: Assumes pytest, but we use unittest
- **Problem**: More complex setup for simple backend switching

✅ **Use `load_tests`**: Clean, standard, works with unittest

---

## Backend-Only Code in Tests

### DO: Use Backend Methods

```python
# ✅ GOOD - uses backend methods
X = self._bkd.array([[1.0, 2.0, 3.0]])
result = self._bkd.sum(X)
expected = self._bkd.array([[6.0]])

self.assertTrue(
    self._bkd.allclose(result, expected)
)
```

### DON'T: Use numpy/torch Directly

```python
# ❌ BAD - uses numpy directly
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
# ✅ Use backend array creation
X = self._bkd.array([[1.0, 2.0, 3.0]])
zeros = self._bkd.zeros((3, 3))
ones = self._bkd.ones((2, 2))
```

#### Comparisons
```python
# ✅ Use backend comparison methods
self.assertTrue(self._bkd.allclose(a, b, rtol=1e-6, atol=1e-7))
self.assertTrue(self._bkd.all_bool(values > 0))
```

#### Shape/Type Checks
```python
# ✅ Use standard Python/unittest assertions
self.assertEqual(result.shape, (3, 3))
self.assertGreater(value, 0.0)
```

#### Converting to Native Types (when necessary)
```python
# ✅ Only convert when absolutely necessary
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


def load_tests(loader, tests, pattern):
    """Skip base class tests."""
    suite = unittest.TestSuite()
    for test in tests:
        for test_case in test:
            test_class_name = test_case.__class__.__name__
            if 'Numpy' in test_class_name or 'Torch' in test_class_name:
                suite.addTest(test_case)
    return suite


class TestMyKernel(Generic[Array], unittest.TestCase):
    """Base class for MyKernel tests."""

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

### Example 2: Testing a Scaling Function

```python
"""
Tests for PolynomialScaling.
"""

import unittest
from typing import Any, Generic

import torch
from numpy.typing import NDArray

from pyapprox.typing.util.backends.numpy import NumpyBkd
from pyapprox.typing.util.backends.torch import TorchBkd
from pyapprox.typing.util.backends.protocols import Array
from pyapprox.typing.surrogates.kernels.scalings import PolynomialScaling


def load_tests(loader, tests, pattern):
    """Skip base class tests."""
    suite = unittest.TestSuite()
    for test in tests:
        for test_case in test:
            test_class_name = test_case.__class__.__name__
            if 'Numpy' in test_class_name or 'Torch' in test_class_name:
                suite.addTest(test_case)
    return suite


class TestPolynomialScaling(Generic[Array], unittest.TestCase):
    """Base class for PolynomialScaling tests."""

    def bkd(self):
        raise NotImplementedError

    def setUp(self):
        self._bkd = self.bkd()
        self._bounds = (0.1, 2.0)

    def test_constant_scaling(self):
        """Test degree 0 (constant) scaling."""
        scaling = PolynomialScaling([0.8], self._bounds, self._bkd, nvars=1)

        X = self._bkd.array([[-1.0, 0.0, 1.0]])
        rho = scaling(X)

        # All values should be 0.8
        expected = self._bkd.ones((3, 1)) * 0.8
        self.assertTrue(
            self._bkd.allclose(rho, expected)
        )

    def test_linear_scaling_jacobian(self):
        """Test spatial Jacobian for linear scaling."""
        scaling = PolynomialScaling([0.9, 0.1], self._bounds, self._bkd)

        X = self._bkd.array([[-1.0, 0.0, 1.0]])
        jac_x = scaling.jacobian(X)

        # ∂ρ/∂x = 0.1 (constant slope)
        expected = self._bkd.ones((3, 1)) * 0.1
        self.assertTrue(
            self._bkd.allclose(jac_x, expected, rtol=1e-6, atol=1e-7)
        )


class TestPolynomialScalingNumpy(TestPolynomialScaling[NDArray[Any]]):
    """Test PolynomialScaling with NumPy backend."""

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestPolynomialScalingTorch(TestPolynomialScaling[torch.Tensor]):
    """Test PolynomialScaling with PyTorch backend."""

    def bkd(self) -> TorchBkd:
        return TorchBkd()


if __name__ == "__main__":
    unittest.main()
```

---

## Running Tests

### With unittest (Recommended)

```bash
# Run all tests in a module
python -m unittest pyapprox.typing.surrogates.kernels.tests.test_my_kernel

# Run multiple test modules
python -m unittest \
    pyapprox.typing.surrogates.kernels.tests.test_my_kernel \
    pyapprox.typing.surrogates.kernels.tests.test_scalings

# Run specific test class (both backends)
python -m unittest pyapprox.typing.surrogates.kernels.tests.test_my_kernel.TestMyKernelNumpy

# Run from file directly
python pyapprox/typing/surrogates/kernels/tests/test_my_kernel.py
```

### With pytest (Also Supported)

```bash
# Run all tests in a module
pytest pyapprox/typing/surrogates/kernels/tests/test_my_kernel.py

# Run with verbose output
pytest pyapprox/typing/surrogates/kernels/tests/test_my_kernel.py -v

# Run specific test
pytest pyapprox/typing/surrogates/kernels/tests/test_my_kernel.py::TestMyKernelNumpy::test_kernel_symmetry
```

**Note**: When using pytest, the `load_tests` function is ignored. Base class tests may run but will fail with `NotImplementedError` - this is expected and can be filtered.

### Expected Output

```
$ python -m unittest pyapprox.typing.surrogates.kernels.tests.test_my_kernel
......................................................................
----------------------------------------------------------------------
Ran 66 tests in 0.549s

OK
```

---

## Best Practices

### 1. Always Use Backend Methods

❌ **Bad**:
```python
import numpy as np
result = np.sum(array)
```

✅ **Good**:
```python
result = self._bkd.sum(array)
```

### 2. Never Import numpy/torch Directly in Tests

❌ **Bad**:
```python
import numpy as np
import torch
```

✅ **Good**:
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
- **PyTorch**: float32 by default (lower precision)

Use relaxed tolerances for cross-backend compatibility:

```python
# ✅ Good - relaxed tolerance for float32
self.assertTrue(
    self._bkd.allclose(result, expected, rtol=1e-6, atol=1e-7)
)

# ❌ Too strict - may fail on torch
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
├── test_matern.py          # Tests MaternKernel
├── test_scalings.py        # Tests PolynomialScaling
├── test_composition.py     # Tests ProductKernel, SumKernel
└── test_plot.py           # Tests plotting wrappers
```

### 7. Use Descriptive Test Names

```python
# ✅ Good - clear what is being tested
def test_kernel_matrix_shape(self):
def test_kernel_symmetry(self):
def test_constant_scaling_jacobian(self):

# ❌ Bad - unclear what is being tested
def test_kernel(self):
def test_1(self):
```

### 8. Include Type Annotations

```python
class TestMyKernelNumpy(TestMyKernel[NDArray[Any]]):
    """Test MyKernel with NumPy backend."""

    def bkd(self) -> NumpyBkd:  # ✅ Return type annotation
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


def load_tests(loader, tests, pattern):
    """
    Custom test loader to skip base class tests.

    Only run tests from concrete implementation classes (Numpy, Torch),
    not from the generic base classes.
    """
    suite = unittest.TestSuite()

    # Only add tests from concrete Numpy and Torch classes
    for test in tests:
        for test_case in test:
            test_class_name = test_case.__class__.__name__
            if 'Numpy' in test_class_name or 'Torch' in test_class_name:
                suite.addTest(test_case)

    return suite


class Test[ClassName](Generic[Array], unittest.TestCase):
    """Base class for [ClassName] tests."""

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

1. ✅ Use Generic base class with `Generic[Array]`
2. ✅ Add `load_tests` function to skip base class tests
3. ✅ Use backend methods exclusively (never numpy/torch directly)
4. ✅ Import torch unconditionally (assumed installed)
5. ✅ Run with `python -m unittest` for clean output
6. ✅ Use relaxed tolerances for cross-backend compatibility
7. ✅ Test with both NumPy and Torch backends
8. ✅ One concrete test class per backend

Following these conventions ensures consistent, maintainable, backend-agnostic testing throughout the PyApprox typing module.
