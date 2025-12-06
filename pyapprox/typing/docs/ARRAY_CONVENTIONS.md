# Array Shape Conventions

This document describes the conventions for array shapes used throughout the `pyapprox.typing` module.

## Table of Contents

1. [Overview](#overview)
2. [Input Arrays (X)](#input-arrays-x)
3. [Output Arrays (y)](#output-arrays-y)
4. [Jacobian Arrays](#jacobian-arrays)
5. [When to Use 1D vs 2D Arrays](#when-to-use-1d-vs-2d-arrays)
6. [Random Variables Convention](#random-variables-convention)
7. [Examples](#examples)
8. [Backend Compatibility](#backend-compatibility)

---

## Overview

The `pyapprox.typing` module follows consistent conventions for array shapes to ensure clarity and interoperability across all components (kernels, functions, optimizers, etc.).

**Key Principles**:
- **Samples are columns** for input arrays X
- **Variables/dimensions are rows** for input arrays X
- **Samples are rows** for output arrays y (when shape matters)
- **Outputs/quantities-of-interest are columns** for output arrays y
- **1D arrays** are used when the array structure cannot be 2D (e.g., hyperparameters, single sample evaluation)

---

## Input Arrays (X)

### Shape Convention

Input arrays **X** have shape **(nvars, nsamples)** where:
- **nvars**: Number of input variables (dimensions)
- **nsamples**: Number of samples (data points, realizations)

**Each column is one sample**, and **each row is one variable**.

### Interpretation

```
X = [[x1_sample1, x1_sample2, x1_sample3, ...],
     [x2_sample1, x2_sample2, x2_sample3, ...],
     [x3_sample1, x3_sample2, x3_sample3, ...],
     ...]
```

- **X[:, i]**: The i-th sample (column vector of length nvars)
- **X[j, :]**: The j-th variable across all samples (row vector of length nsamples)

### Rationale

This convention is standard in many scientific computing libraries and has several benefits:
1. **Natural matrix operations**: When X is a design matrix, X.T @ X forms the Gram matrix
2. **Kernel evaluations**: K(X1, X2) naturally computes all pairwise kernel values
3. **Broadcasting**: Many operations broadcast naturally over samples

### Examples

#### Single Sample in 2D
```python
# One sample with x1=1.0, x2=2.0
X = bkd.array([[1.0], [2.0]])  # Shape: (2, 1)
```

#### Multiple Samples in 1D
```python
# Three samples: x=0.0, x=1.0, x=2.0
X = bkd.array([[0.0, 1.0, 2.0]])  # Shape: (1, 3)
```

#### Multiple Samples in 3D
```python
# Two samples in 3D space
X = bkd.array([
    [1.0, 2.0],   # x1 values
    [3.0, 4.0],   # x2 values
    [5.0, 6.0]    # x3 values
])  # Shape: (3, 2)
```

---

## Output Arrays (y)

### Shape Convention

Output arrays **y** have shape **(nsamples, nqoi)** where:
- **nsamples**: Number of samples (evaluations, data points)
- **nqoi**: Number of quantities of interest (output dimensions)

**Each row is one sample's output**, and **each column is one output variable**.

### Interpretation

```
y = [[y1_sample1, y2_sample1, y3_sample1, ...],
     [y1_sample2, y2_sample2, y3_sample2, ...],
     [y1_sample3, y2_sample3, y3_sample3, ...],
     ...]
```

- **y[i, :]**: The i-th sample's outputs (row vector of length nqoi)
- **y[:, j]**: The j-th output across all samples (column vector of length nsamples)

### Rationale

This convention differs from the input convention for practical reasons:
1. **Observation matrix**: y is typically thought of as an observation matrix where each row is an observation
2. **Regression target**: Matches convention in GP regression and sklearn
3. **Statistical interpretation**: Each row is a realization, each column is a variable being measured

### Examples

#### Single Output, Multiple Samples
```python
# Three samples with scalar output
y = bkd.array([[1.0], [2.0], [3.0]])  # Shape: (3, 1)
```

#### Multiple Outputs, Multiple Samples
```python
# Two samples with 3 outputs each
y = bkd.array([
    [1.0, 2.0, 3.0],  # Sample 1: [y1, y2, y3]
    [4.0, 5.0, 6.0]   # Sample 2: [y1, y2, y3]
])  # Shape: (2, 3)
```

### Special Case: Function Protocol

When implementing `FunctionProtocol`, the `__call__` method returns:
- **Shape**: (nqoi, nsamples) to match the **transpose** of the output convention
- **Reason**: Consistency with gradient shapes and kernel evaluation patterns

```python
class MyFunction(FunctionProtocol[Array]):
    def __call__(self, X: Array) -> Array:
        """
        Evaluate function.

        Parameters
        ----------
        X : Array
            Input samples, shape (nvars, nsamples)

        Returns
        -------
        Array
            Output values, shape (nqoi, nsamples)
        """
        # ...
```

**This means**:
- **Training data y**: shape (nsamples, nqoi)
- **Function evaluation f(X)**: shape (nqoi, nsamples)

To convert between conventions, use transpose:
```python
# Function evaluation
f_X = my_function(X)  # Shape: (nqoi, nsamples)

# Convert to training data format
y = bkd.transpose(f_X)  # Shape: (nsamples, nqoi)
```

---

## Jacobian Arrays

### Shape Convention

**Critical Rule**: Jacobians are **always 2D arrays**, even when nqoi=1 or nvars=1.

Jacobian arrays have shape **(nqoi, nvars)** where:
- **nqoi**: Number of outputs (quantities of interest)
- **nvars**: Number of input variables

**We never use 1D gradients.** All derivatives are expressed as Jacobian matrices.

### Function Jacobian: ∂f/∂x

For a function f: R^nvars → R^nqoi, the Jacobian is:

```python
jac = function.jacobian(X)  # Shape: (nqoi, nvars, nsamples)
```

**Per-sample Jacobian**: For a single sample, shape is **(nqoi, nvars)**:
```python
jac_i = jac[:, :, i]  # Shape: (nqoi, nvars)
```

**Special Cases**:
- Scalar function (nqoi=1): Shape is **(1, nvars)** NOT (nvars,)
- Single variable (nvars=1): Shape is **(nqoi, 1)** NOT (nqoi,)

### Kernel Jacobian: ∂K/∂x

For kernel spatial derivatives:

```python
jac = kernel.jacobian(X1, X2)  # Shape: (nsamples1, nsamples2, nvars)
```

The last dimension is the gradient direction. For a single kernel value K(x1, x2):
```python
grad_x1 = jac[i, j, :]  # Shape: (nvars,) - gradient w.r.t. first argument
```

**Note**: This is the one exception where we extract a 1D array, but the full Jacobian is always 3D.

### Parameter Jacobian: ∂f/∂θ

For derivatives with respect to hyperparameters:

```python
jac_params = kernel.jacobian_wrt_params(X)  # Shape: (nsamples, nsamples, nparams)
```

For a scalar loss function L(θ):
```python
jac = loss.jacobian(params)  # Shape: (1, nparams) NOT (nparams,)
```

**Always 2D**: Even though the loss is scalar, the Jacobian must be 2D.

### Examples

#### Scalar Function Jacobian
```python
# f: R^3 → R (scalar output)
class MyScalarFunction(FunctionProtocol[Array]):
    def nvars(self) -> int:
        return 3

    def nqoi(self) -> int:
        return 1

    def jacobian(self, X: Array) -> Array:
        """
        Compute Jacobian.

        Parameters
        ----------
        X : Array, shape (3, nsamples)

        Returns
        -------
        Array, shape (1, 3, nsamples)  # NOT (3, nsamples)!
        """
        # Compute gradient
        grad = compute_gradient(X)  # Shape: (3, nsamples)

        # Reshape to (1, 3, nsamples) for Jacobian format
        return bkd.reshape(grad, (1, 3, X.shape[1]))
```

#### Multi-Output Function Jacobian
```python
# f: R^2 → R^3 (3 outputs)
def jacobian(self, X: Array) -> Array:
    """
    Returns
    -------
    Array, shape (3, 2, nsamples)
    """
    # jac[i, j, k] = ∂f_i/∂x_j at sample k
```

#### Loss Function Jacobian
```python
class MyLoss(FunctionWithJacobianProtocol[Array]):
    def nvars(self) -> int:
        return nparams  # Number of parameters

    def nqoi(self) -> int:
        return 1  # Scalar loss

    def jacobian(self, params: Array) -> Array:
        """
        Compute loss gradient w.r.t. parameters.

        Parameters
        ----------
        params : Array, shape (nparams,)

        Returns
        -------
        Array, shape (1, nparams)  # NOT (nparams,)!
        """
        grad = compute_gradient(params)  # Shape may vary

        # Ensure 2D output
        return bkd.reshape(grad, (1, self.nvars()))
```

### Jacobian Naming Conventions

**Critical Rule**: Method names indicate which variable the derivative is with respect to.

#### Primary Jacobian: `jacobian()`

The `jacobian()` method (without suffix) computes derivatives **with respect to the primary variables** - those that are varied most often:

- **Kernels**: `jacobian(X1, X2)` → ∂K/∂X (spatial derivatives)
- **Functions**: `jacobian(X)` → ∂f/∂X (function derivatives)
- **Surrogates/GPs**: `jacobian(X)` → ∂f*/∂X (predictive derivatives)

```python
class MaternKernel:
    def jacobian(self, X1: Array, X2: Array) -> Array:
        """
        Spatial Jacobian: ∂K/∂X1.

        Primary variables are the input points X.
        """
        ...

class MyFunction:
    def jacobian(self, X: Array) -> Array:
        """
        Function Jacobian: ∂f/∂X.

        Primary variables are the inputs X.
        """
        ...
```

#### Parameter Jacobian: `jacobian_wrt_params()`

The `jacobian_wrt_params()` method computes derivatives **with respect to hyperparameters** - those that are optimized less frequently:

- **Kernels**: `jacobian_wrt_params(X)` → ∂K/∂θ (kernel hyperparameters)
- **Surrogates**: `jacobian_wrt_params(X)` → ∂f*/∂θ (model hyperparameters)
- **Scaling functions**: `jacobian_wrt_params(X)` → ∂ρ/∂c (scaling coefficients)

```python
class MaternKernel:
    def jacobian_wrt_params(self, X: Array) -> Array:
        """
        Parameter Jacobian: ∂K/∂θ where θ = log(length_scale).

        Secondary variables are the hyperparameters.
        """
        ...
```

#### Multi-Variable Classes: Explicit Naming

For classes with multiple variable types (e.g., implicit functions with state and parameters), **always use explicit names**:

- `jacobian_wrt_state()` → ∂f/∂u (state variables)
- `jacobian_wrt_params()` → ∂f/∂p (parameters)
- `jacobian_wrt_control()` → ∂f/∂c (control inputs)

```python
class ImplicitFunction:
    """Implicit function: F(u, p) = 0."""

    def jacobian_wrt_state(self, u: Array, p: Array) -> Array:
        """Jacobian w.r.t. state: ∂F/∂u."""
        ...

    def jacobian_wrt_params(self, u: Array, p: Array) -> Array:
        """Jacobian w.r.t. parameters: ∂F/∂p."""
        ...
```

**Rationale**:
- **Primary** = what users vary most (inputs, test points)
- **Parameters** = what is optimized/tuned (hyperparameters, coefficients)
- **Explicit naming** prevents ambiguity when multiple variable types exist

### Why Always 2D?

**Consistency**: All Jacobians have the same structural interpretation regardless of dimensions:
- **Row i**: Derivatives of output i
- **Column j**: Derivatives with respect to input j

**Type Safety**: 2D shape prevents ambiguity and makes code more robust:
```python
# ❌ BAD - ambiguous shape
grad = loss.jacobian(params)  # Shape: (nparams,) - is this a gradient or a vector?

# ✅ GOOD - unambiguous
jac = loss.jacobian(params)  # Shape: (1, nparams) - clearly a Jacobian row
```

**Matrix Operations**: 2D Jacobians work naturally with matrix operations:
```python
# Hessian-vector product: J.T @ J @ v
H_v = jac.T @ jac @ v  # Works naturally when jac is (1, nparams)
```

---

## When to Use 1D vs 2D Arrays

### Use 2D Arrays When:

- **Inputs X**: Always use (nvars, nsamples) even for single sample or single variable
- **Outputs y**: Always use (nsamples, nqoi) for training data
- **Function evaluations**: Always use (nqoi, nsamples) when implementing FunctionProtocol

### Use 1D Arrays When:

1. **Hyperparameters**: Shape (nparams,) - a flat list of parameter values
   ```python
   params = bkd.array([0.5, 1.0, 2.0])  # Shape: (3,)
   ```

2. **Diagonal elements**: Shape (n,) - diagonal of a matrix
   ```python
   diag = kernel.diag(X)  # Shape: (nsamples,)
   ```

3. **Index arrays**: Shape (n,) - indices for selection
   ```python
   indices = bkd.array([0, 2, 5])  # Shape: (3,)
   ```

### Always Use 2D Arrays For:

1. **Bounds**: Shape (nparams, 2) - each row is [lower, upper]
   ```python
   bounds = bkd.array([[0.1, 10.0], [0.5, 5.0]])  # Shape: (2, 2)
   ```

2. **Jacobians**: Shape (nqoi, nvars) - even when nqoi=1 or nvars=1
   ```python
   # Scalar function (nqoi=1)
   jac = loss.jacobian(X)  # Shape: (1, nvars) NOT (nvars,)

   # Single variable (nvars=1)
   jac = kernel.jacobian(X)  # Shape: (nqoi, 1) NOT (nqoi,)
   ```

**Important**: We **never** use 1D gradients. All derivatives are expressed as Jacobian matrices (2D arrays).

### Reshaping Guidelines

**DON'T** use 1D arrays where 2D is expected. Instead, use `reshape`:

```python
# ❌ BAD - 1D array for input
X = bkd.array([1.0, 2.0, 3.0])  # Shape: (3,) - ambiguous!

# ✅ GOOD - 2D array with explicit shape
X = bkd.array([[1.0, 2.0, 3.0]])  # Shape: (1, 3) - 1 variable, 3 samples

# ✅ GOOD - Alternative using reshape
X = bkd.reshape(bkd.array([1.0, 2.0, 3.0]), (1, 3))
```

**Ambiguity Problem**: A 1D array of length n could mean:
- n samples of a 1D variable? → Should be (1, n)
- 1 sample of an n-dimensional variable? → Should be (n, 1)

Always use 2D to remove ambiguity.

---

## Random Variables Convention

When working with random variables, the conventions extend naturally:

### Input Random Variables

If X represents samples from a random vector with **nvars** marginal distributions:
- **X**: shape (nvars, nsamples)
- **X[i, :]**: nsamples realizations of the i-th marginal
- **X[:, j]**: One realization of the joint random vector

### Output Random Variables

If evaluating a function of random inputs, outputs are also random:
- **y = f(X)**: shape (nsamples, nqoi)
- **y[i, :]**: One realization of the output random vector
- **y[:, j]**: nsamples realizations of the j-th output marginal

### Example: Monte Carlo Sampling

```python
# Sample from random vector X with 2 marginals
nsamples = 1000
X = bkd.array([
    bkd.random.normal(0, 1, nsamples),  # Marginal 1
    bkd.random.normal(0, 1, nsamples)   # Marginal 2
])  # Shape: (2, 1000)

# Evaluate function (random outputs)
y = my_function(X)  # Shape: (nqoi, 1000)

# Convert to training format
y_train = bkd.transpose(y)  # Shape: (1000, nqoi)

# Each row of y_train is one realization of the output random vector
# Each column of y_train is the marginal distribution of one output
```

---

## Examples

### Example 1: Kernel Evaluation

```python
from pyapprox.typing.surrogates.kernels import MaternKernel
from pyapprox.typing.util.backends.numpy import NumpyBkd

bkd = NumpyBkd()
kernel = MaternKernel(2.5, [1.0, 1.0], (0.1, 10.0), 2, bkd)

# Input: 5 samples in 2D
X1 = bkd.random.randn(2, 5)  # Shape: (2, 5)

# Input: 3 samples in 2D
X2 = bkd.random.randn(2, 3)  # Shape: (2, 3)

# Kernel matrix
K = kernel(X1, X2)  # Shape: (5, 3)
# K[i, j] = k(X1[:, i], X2[:, j])
```

### Example 2: Gaussian Process Regression

```python
from pyapprox.typing.surrogates.gaussianprocess import ExactGaussianProcess

# Training data
X_train = bkd.random.randn(2, 100)  # Shape: (2, 100) - 100 samples in 2D
y_train = bkd.random.randn(100, 1)  # Shape: (100, 1) - 100 scalar outputs

# Fit GP
gp = ExactGaussianProcess(kernel, mean, noise_variance)
gp.fit(X_train, y_train)

# Test data
X_test = bkd.random.randn(2, 20)  # Shape: (2, 20) - 20 samples in 2D

# Predict
y_pred = gp.predict(X_test)  # Shape: (20, 1)
```

### Example 3: Multi-Output Function

```python
from pyapprox.typing.interface.functions.protocols import FunctionProtocol

class MyMultiOutputFunction(FunctionProtocol[Array]):
    def nvars(self) -> int:
        return 2

    def nqoi(self) -> int:
        return 3

    def __call__(self, X: Array) -> Array:
        """
        Evaluate function.

        Parameters
        ----------
        X : Array, shape (2, nsamples)

        Returns
        -------
        Array, shape (3, nsamples)
        """
        # Each output is a function of the inputs
        y1 = X[0, :] + X[1, :]     # Sum
        y2 = X[0, :] * X[1, :]     # Product
        y3 = X[0, :] ** 2 + X[1, :] ** 2  # Sum of squares

        return self._bkd.stack([y1, y2, y3], axis=0)  # Shape: (3, nsamples)

# Usage
f = MyMultiOutputFunction(bkd)
X = bkd.array([[1.0, 2.0], [3.0, 4.0]])  # Shape: (2, 2) - 2 samples
y = f(X)  # Shape: (3, 2)

# Convert to training format
y_train = bkd.transpose(y)  # Shape: (2, 3)
```

### Example 4: Hyperparameter Optimization

```python
from pyapprox.typing.util.hyperparameter import HyperParameterList

# Create hyperparameters (1D array)
hyp_list = HyperParameterList()
# ... add hyperparameters ...

# Get active values
params = hyp_list.get_active_values()  # Shape: (nactive,) - 1D

# Set new values
new_params = bkd.array([0.5, 1.0, 2.0])  # Shape: (3,) - 1D
hyp_list.set_active_values(new_params)

# Get bounds
bounds = hyp_list.get_bounds()  # List of tuples [(lb1, ub1), (lb2, ub2), ...]
```

---

## Backend Compatibility

All array conventions work seamlessly with both NumPy and PyTorch backends:

### NumPy Backend
```python
from pyapprox.typing.util.backends.numpy import NumpyBkd
import numpy as np

bkd = NumpyBkd()
X = bkd.array([[1.0, 2.0], [3.0, 4.0]])  # Returns np.ndarray
```

### PyTorch Backend
```python
from pyapprox.typing.util.backends.torch import TorchBkd
import torch

bkd = TorchBkd()
X = bkd.array([[1.0, 2.0], [3.0, 4.0]])  # Returns torch.Tensor
```

### Key Backend Methods

- `bkd.array()`: Create array from list
- `bkd.reshape(X, shape)`: Reshape array
- `bkd.transpose(X)`: Transpose array
- `bkd.stack(arrays, axis)`: Stack arrays along axis
- `bkd.to_numpy(X)`: Convert to NumPy (for plotting, assertions)

**Important**: Always use backend methods, never create arrays directly with `np.array()` or `torch.tensor()` in generic code.

---

## Summary

| Array Type | Shape | Samples Are | Variables Are |
|------------|-------|-------------|---------------|
| Input X | (nvars, nsamples) | Columns | Rows |
| Output y (training) | (nsamples, nqoi) | Rows | Columns |
| Function f(X) | (nqoi, nsamples) | Columns | Rows |
| Hyperparameters | (nparams,) | N/A (1D) | N/A (1D) |
| Kernel K(X1, X2) | (nsamples1, nsamples2) | Both dims | N/A |

**Key Takeaways**:
1. **X is always (nvars, nsamples)** - samples are columns
2. **y training data is (nsamples, nqoi)** - samples are rows
3. **FunctionProtocol returns (nqoi, nsamples)** - transpose of y
4. **Use 2D arrays** unless the data is inherently 1D (hyperparameters, indices)
5. **Transpose** to convert between training format and function format
6. **Random variable samples** follow the same conventions
