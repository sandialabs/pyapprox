# PyApprox Benchmark Conventions

This document defines the conventions for creating and using benchmarks in `pyapprox_benchmarks`.

## Overview

Benchmarks are **fixed problem instances** with known ground truth values. They serve to:
- Validate algorithm implementations
- Provide reproducible test cases
- Enable comparisons across methods

Benchmarks differ from the reusable function implementations in `pyapprox.interface.functions` - they are pre-configured instances with specific parameters and domain bounds.

## Benchmark Categories

| Category | Description | Ground Truth | Example |
|----------|-------------|--------------|---------|
| `sensitivity` | Sensitivity analysis problems | Sobol indices, mean, variance | `ishigami_3d` |
| `optimization` | Unconstrained/constrained optimization | Global minimum, minimizers | `rosenbrock_2d` |
| `quadrature` | Numerical integration | Analytical integral | `genz_oscillatory_2d` |
| `multifidelity` | Multi-model ensembles | Correlations, costs | `polynomial_ensemble_5model` |
| `ode` | ODE systems | Initial conditions, nominal params | `lotka_volterra_3species` |

## Core Protocols

### BenchmarkProtocol

The base protocol for all benchmarks:

```python
@runtime_checkable
class BenchmarkProtocol(Protocol, Generic[Array]):
    def name(self) -> str: ...
    def function(self) -> FunctionProtocol[Array]: ...
    def ground_truth(self) -> GroundTruthProtocol: ...
    def domain(self) -> DomainProtocol[Array]: ...
```

### BenchmarkWithPriorProtocol

For UQ problems requiring probability distributions:

```python
@runtime_checkable
class BenchmarkWithPriorProtocol(Protocol, Generic[Array]):
    # ... all BenchmarkProtocol methods plus:
    def prior(self) -> Any: ...  # Distribution for sampling
```

### ConstrainedBenchmarkProtocol

For constrained optimization:

```python
@runtime_checkable
class ConstrainedBenchmarkProtocol(Protocol, Generic[Array]):
    # ... all BenchmarkProtocol methods plus:
    def constraints(self) -> Sequence[ConstraintProtocol[Array]]: ...
```

## Category-Specific Requirements

### Algebraic Benchmarks (sensitivity, optimization, quadrature)

Standard benchmarks provide a callable `function()`:

```python
def function(self) -> FunctionProtocol[Array]:
    """Return callable function.

    The returned function supports:
    - __call__(samples: Array) -> Array  # shape: (nvars, N) -> (nqoi, N)
    - nvars() -> int
    - nqoi() -> int
    - bkd() -> Backend[Array]

    May also support jacobian(), hvp(), whvp() if available.
    """
```

**Ground Truth Types:**

| Type | Required Fields | Optional Fields |
|------|-----------------|-----------------|
| `SensitivityGroundTruth` | - | mean, variance, main_effects, total_effects, sobol_indices |
| `OptimizationGroundTruth` | - | global_minimum, global_minimizers, local_minima |
| `QuadratureGroundTruth` | - | integral, integral_formula |

### ODE Benchmarks

ODE benchmarks provide a `residual()` instead of `function()`:

```python
def residual(self) -> ODEResidualProtocol[Array]:
    """Return ODE residual defining dy/dt = f(y, t; p).

    The residual supports:
    - __call__(state, time) -> Array  # RHS evaluation
    - jacobian(state, time) -> Array  # State Jacobian
    - set_param(param) -> None        # Set parameters
    - param_jacobian(state, time) -> Array  # Parameter Jacobian
    """

def time_config(self) -> ODETimeConfig:
    """Return time integration settings (init_time, final_time, deltat)."""

def nstates(self) -> int:
    """Return number of state variables."""

def nparams(self) -> int:
    """Return number of parameters."""
```

**ODEGroundTruth Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `nstates` | int | Number of state variables |
| `nparams` | int | Number of parameters |
| `initial_condition` | ndarray | Default initial state, shape (nstates,) |
| `nominal_parameters` | ndarray | Nominal parameter values, shape (nparams,) |
| `init_time` | float | Start time |
| `final_time` | float | End time |
| `deltat` | float | Time step |

### Multifidelity Benchmarks

Multifidelity benchmarks provide an `ensemble()` of models:

```python
def ensemble(self) -> Sequence[FunctionProtocol[Array]]:
    """Return ordered list of models (highest to lowest fidelity)."""
```

**MultifidelityGroundTruth Fields:**
- `high_fidelity_mean`, `high_fidelity_variance`
- `model_correlations` - Correlation matrix between models
- `model_costs` - Relative computational costs

## Registry System

All benchmarks must be registered with the `BenchmarkRegistry`:

```python
from pyapprox_benchmarks.registry import BenchmarkRegistry

@BenchmarkRegistry.register(
    "my_benchmark_name",
    category="sensitivity",  # or optimization, quadrature, ode, multifidelity
    description="Short description for catalog"
)
def _my_benchmark_factory(bkd: Backend[Array]) -> Benchmark[Array]:
    return my_benchmark_instance(bkd)
```

**Usage:**

```python
from pyapprox_benchmarks.registry import BenchmarkRegistry
from pyapprox.util.backends.numpy import NumpyBkd

bkd = NumpyBkd()
benchmark = BenchmarkRegistry.get("ishigami_3d", bkd)

# List available benchmarks
BenchmarkRegistry.list_all()
BenchmarkRegistry.list_category("sensitivity")
BenchmarkRegistry.categories()
```

## Naming Conventions

| Element | Convention | Examples |
|---------|------------|----------|
| Benchmark names | `lowercase_with_underscores` | `ishigami_3d`, `rosenbrock_10d` |
| Dimension suffix | `_{dimension}d` when relevant | `genz_oscillatory_2d` |
| Factory functions | `_name_factory` (private) | `_ishigami_factory` |
| Public creators | `name()` function | `ishigami_3d(bkd)` |

## File Organization

```
pyapprox/typing/benchmarks/
├── __init__.py              # Public exports
├── protocols.py             # Protocol definitions
├── benchmark.py             # Base classes (BoxDomain, Benchmark)
├── ground_truth.py          # Ground truth dataclasses
├── registry.py              # BenchmarkRegistry
├── CONVENTIONS.md           # This file
├── functions/               # Function implementations
│   ├── algebraic/           # Ishigami, Rosenbrock, etc.
│   ├── genz/                # Genz integration test functions
│   ├── multifidelity/       # Polynomial ensembles
│   └── ode/                 # ODE benchmark wrapper
├── instances/               # Pre-configured instances
│   ├── sensitivity.py       # ishigami_3d, sobol_g_*
│   ├── optimization.py      # rosenbrock_*, branin_2d
│   ├── quadrature.py        # genz_*
│   ├── multifidelity.py     # polynomial_ensemble_*
│   └── ode.py               # lotka_volterra_*, etc.
└── tests/
```

## Creating a New Benchmark

### Step 1: Implement the Function

Create the function class in `functions/[category]/`:

```python
# functions/algebraic/my_function.py
from typing import Generic
from pyapprox.util.backends.protocols import Array, Backend

class MyFunction(Generic[Array]):
    def __init__(self, a: float, b: float, bkd: Backend[Array]) -> None:
        self._a = a
        self._b = b
        self._bkd = bkd

    def bkd(self) -> Backend[Array]:
        return self._bkd

    def nvars(self) -> int:
        return 2

    def nqoi(self) -> int:
        return 1

    def __call__(self, samples: Array) -> Array:
        # samples shape: (nvars, nsamples)
        x = samples[0:1, :]
        y = samples[1:2, :]
        return self._a * x**2 + self._b * y**2
```

### Step 2: Create Instance Factory

Create the pre-configured instance in `instances/[category].py`:

```python
# instances/optimization.py
from pyapprox_benchmarks.benchmark import Benchmark, BoxDomain
from pyapprox_benchmarks.ground_truth import OptimizationGroundTruth
from pyapprox_benchmarks.registry import BenchmarkRegistry
from pyapprox_benchmarks.functions.algebraic.my_function import MyFunction

def my_function_2d(bkd: Backend[Array]) -> Benchmark[Array, OptimizationGroundTruth]:
    """Create 2D my_function benchmark with standard parameters."""
    return Benchmark(
        _name="my_function_2d",
        _function=MyFunction(a=1.0, b=2.0, bkd=bkd),
        _domain=BoxDomain(
            _bounds=bkd.array([[-5.0, 5.0], [-5.0, 5.0]]),
            _bkd=bkd,
        ),
        _ground_truth=OptimizationGroundTruth(
            global_minimum=0.0,
            global_minimizers=bkd.array([[0.0], [0.0]]),
        ),
        _description="Simple quadratic function",
        _reference="Your paper (2025)",
    )

@BenchmarkRegistry.register(
    "my_function_2d",
    category="optimization",
    description="Simple 2D quadratic optimization test"
)
def _my_function_2d_factory(bkd: Backend[Array]) -> Benchmark[Array, OptimizationGroundTruth]:
    return my_function_2d(bkd)
```

### Step 3: Export in `__init__.py`

Add exports to the appropriate `__init__.py` files:

```python
# functions/algebraic/__init__.py
from .my_function import MyFunction

# instances/__init__.py
from .optimization import my_function_2d

# benchmarks/__init__.py
from .functions.algebraic import MyFunction
from .instances import my_function_2d
```

### Step 4: Add Tests

Create dual-backend tests in `tests/test_my_function.py`:

```python
class TestMyFunction:
    def test_output_shape(self, bkd):
        benchmark = my_function_2d(bkd)
        samples = bkd.array([[0.0, 1.0], [0.0, 1.0]])  # (2, 2)
        result = benchmark.function()(samples)
        assert result.shape == (1, 2)

    def test_global_minimum(self, bkd):
        benchmark = my_function_2d(bkd)
        gt = benchmark.ground_truth()
        minimizer = gt.global_minimizers
        result = benchmark.function()(minimizer)
        bkd.assert_allclose(result[0, 0], gt.global_minimum)
```

## Mathematical Notation

For consistency with tutorials, use notation from `pyapprox/tutorials/CONVENTIONS.md`:

| Symbol | Meaning |
|--------|---------|
| $d_\theta$ | Number of parameters (maps to `nparams()`) |
| $d_x$ | Number of input variables (maps to `nvars()`) |
| $d_y$ | Number of outputs (maps to `nqoi()`) |
| $N$ | Number of samples (**reserved** - do not use for other purposes) |
| $\boldsymbol{\theta}$ | Parameter vector |
| $\mathbf{x}$ | Input vector |

## Array Shape Conventions

All arrays follow pyapprox conventions:

| Array | Shape | Description |
|-------|-------|-------------|
| Input samples | `(nvars, nsamples)` | Samples as columns |
| Output values | `(nqoi, nsamples)` | QoIs as rows |
| Single sample | `(nvars, 1)` | Always 2D |
| Bounds | `(nvars, 2)` | [lower, upper] per row |
| Jacobian | `(nqoi, nvars)` | Single sample |
| Jacobian batch | `(nsamples, nqoi, nvars)` | Multiple samples |
