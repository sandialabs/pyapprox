# PyApprox Benchmark Conventions

This document defines the conventions for creating and using benchmarks in `pyapprox_benchmarks`.

## Terminology

| Term | Definition |
|------|-----------|
| **Function** | Reusable computational object. Pure math, no domain, no prior. Methods: `__call__`, `jacobian`, `hvp`, etc. Lives in `functions/`. |
| **Problem** | One or more functions bound to the additional data needed for a specific computational task (domain, prior, constraints, noise model, design space, etc.). No known answers. Lives in `problems/`. |
| **Benchmark** | Problem + typed ground truth methods. Each benchmark is a standalone class exposing exactly the ground truths it supports. Lives in a use-case directory (`sensitivity/`, `optimization/`, etc.). |

**Key rules**:
- No base class for Problem or Benchmark — each is standalone with typed methods
- Code is only reused when multiple problems share the *same* structure AND *same* semantic role
- If there is no ground truth, it is a Problem, not a Benchmark
- Ground truths are atomic typed methods on the benchmark class (not a monolithic dataclass)
- Benchmarks construct their Problem internally in `__init__` — one name per benchmark, no factory functions
- Every Problem class has `name()` and `description()` methods

## Problem Classes

Each problem type is a standalone class with exactly the fields it needs.

| Problem class | Used by | Module |
|---------------|---------|--------|
| `FunctionOverDomainProblem` | Quadrature, unconstrained optimization | `problems/function_over_domain` |
| `ForwardUQProblem` | Sensitivity analysis, forward UQ | `problems/forward_uq` |
| `ConstrainedOptimizationProblem` | Constrained optimization | `problems/optimization` |
| `BayesianInferenceProblem` | Inference, KL-OED | `problems/inverse` |
| `GoalOrientedInferenceProblem` | Prediction OED | `problems/inverse` |
| `KLOEDProblem` | KL-based OED | `problems/oed` |
| `PredictionOEDProblem` | Prediction OED | `problems/oed` |

## Benchmark Classes (standalone, self-constructing)

Each benchmark is a class whose `__init__` takes the backend and configuration parameters, builds the Problem internally, and stores ground truth values. Users construct by instantiation: `RosenbrockBenchmark(bkd, nvars=2)`.

```python
# Example: sensitivity benchmark
class IshigamiBenchmark(Generic[Array]):
    def __init__(self, bkd: Backend[Array], a: float = 7.0, b: float = 0.1) -> None:
        func = IshigamiFunction(bkd, a=a, b=b)
        prior = IndependentJoint([UniformMarginal(-pi, pi, bkd)] * 3, bkd)
        self._problem = ForwardUQProblem("ishigami", func, prior)
        self._mean = ...
        self._main_effects = ...

    def problem(self) -> ForwardUQProblem: return self._problem
    def mean(self) -> float: return self._mean
    def main_effects(self) -> Array: return self._main_effects

# Example: optimization benchmark
class RosenbrockBenchmark(Generic[Array]):
    def __init__(self, bkd: Backend[Array], nvars: int = 2) -> None:
        func = RosenbrockFunction(bkd, nvars=nvars)
        domain = BoxDomain(bkd.array([[-5.0, 5.0]] * nvars), bkd)
        self._problem = FunctionOverDomainProblem("rosenbrock", func, domain)
        self._global_minimum = 0.0
        self._global_minimizers = bkd.ones((nvars, 1))

    def problem(self) -> FunctionOverDomainProblem: return self._problem
    def global_minimum(self) -> float: return self._global_minimum
    def global_minimizers(self) -> Array: return self._global_minimizers
```

Usage:
```python
bm = RosenbrockBenchmark(bkd, nvars=2)
bm.problem().function()(samples)     # typed access
bm.problem().domain().bounds()
bm.global_minimum()                   # typed ground truth
```

## Directory Structure

Organized by **primary intended use case**:

```
pyapprox_benchmarks/
├── __init__.py
├── protocols.py                # DomainProtocol, ConstraintProtocol
├── problems/                   # Problem classes
│   ├── function_over_domain.py #   FunctionOverDomainProblem
│   ├── forward_uq.py           #   ForwardUQProblem
│   ├── optimization.py         #   ConstrainedOptimizationProblem
│   ├── inverse/                #   BayesianInferenceProblem, GaussianInferenceProblem
│   └── oed/                    #   KLOEDProblem, PredictionOEDProblem
├── functions/                  # Reusable function implementations
│   ├── algebraic/
│   ├── genz/
│   ├── multifidelity/
│   └── ode/
├── sensitivity/                # Sensitivity benchmarks
├── optimization/               # Optimization benchmarks
├── quadrature/                 # Quadrature benchmarks
├── statest/                    # Multifidelity variance reduction benchmarks
├── expdesign/                  # OED benchmarks
├── ode/                        # ODE problems (no ground truth)
└── pde/                        # PDE problems/benchmarks
```

### Placement rules
- Benchmark goes in directory matching its **original/canonical use case**
- Ishigami -> `sensitivity/` even though usable for quadrature
- ODE/PDE instances without analytical ground truth -> `ode/`/`pde/` as Problems
- ODE/PDE instances designed for a specific use case (e.g., advdiff for OED) -> that use-case directory

## Protocols

Only two structural protocols ship with the package:

- `DomainProtocol` — input domain specification (`bounds()`, `nvars()`, `bkd()`)
- `ConstraintProtocol` — single constraint function (`__call__()`, `constraint_type()`)

No benchmark-level protocols. Users define their own query protocols if needed:
```python
@runtime_checkable
class HasMeanAndMainEffects(Protocol, Generic[Array]):
    def mean(self) -> float: ...
    def main_effects(self) -> Array: ...
```

## Mathematical Notation

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
