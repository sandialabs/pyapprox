[![Tests](https://github.com/sandialabs/pyapprox/actions/workflows/tests.yml/badge.svg)](https://github.com/sandialabs/pyapprox/actions/workflows/tests.yml)
[![Lint](https://github.com/sandialabs/pyapprox/actions/workflows/lint.yml/badge.svg)](https://github.com/sandialabs/pyapprox/actions/workflows/lint.yml)

# PyApprox

## Documentation

Online documentation can be found at [PyApprox](https://sandialabs.github.io/pyapprox/).

## Description

PyApprox provides flexible and efficient tools for high-dimensional approximation, uncertainty quantification, and decision-making under uncertainty. It implements methods addressing various issues surrounding high-dimensional parameter spaces and limited evaluations of expensive simulation models, with the goal of facilitating simulation-aided knowledge discovery, prediction, and design.

Tools are provided for:

1. **Surrogate modeling** — polynomial chaos expansions (least squares, compressive sensing, interpolation), Gaussian process regression (single- and multi-output, DAG-structured), low-rank tensor decompositions (function trains), and sparse grid interpolation
2. **Multi-fidelity estimation** — approximate control variates (ACV), multi-level Monte Carlo (MLMC), multi-fidelity Monte Carlo (MFMC), MLBLUE, and group ACV
3. **Bayesian experimental design** — KL-based and goal-oriented optimal experimental design with gradient-based optimization
4. **Bayesian inference** — MCMC sampling and conjugate posterior analysis
5. **Sensitivity analysis** — Sobol indices, Morris screening, and surrogate-based sensitivity
6. **Probability and risk** — random variable transformations, risk measures, and random field representations (KLE)
7. **PDE solvers** — collocation and Galerkin finite-element methods for advection-diffusion-reaction, Helmholtz, Stokes, elasticity, and more
8. **Optimization** — implicit function differentiation, adjoint methods, and design under uncertainty

All code is fully typed, supports dual backends (NumPy and PyTorch), and preserves PyTorch autograd computation graphs for automatic differentiation.

## Installation

### From source (recommended for development)

```bash
git clone https://github.com/sandialabs/pyapprox.git
cd pyapprox
pip install -e ".[test]"
```

### Optional dependency groups

```bash
pip install -e ".[fem]"       # Finite element (scikit-fem)
pip install -e ".[umbridge]"  # UMBridge model interface
pip install -e ".[numba]"     # Numba JIT acceleration
pip install -e ".[parallel]"  # Parallel execution (joblib, mpire)
pip install -e ".[cvxpy]"     # Convex optimization
pip install -e ".[test]"      # Testing tools
pip install -e ".[lint]"      # Linting (mypy, ruff)
pip install -e ".[docs]"      # Documentation (quarto)
pip install -e ".[all]"       # Everything
```

### Using conda

```bash
conda env create -f environment.yml
conda activate pyapprox
pip install -e .
```

## Running Tests

```bash
pytest pyapprox -v --tb=short
```

The `-v` flag enables verbose output and `--tb=short` abbreviates tracebacks on failures for readability.

Some tests are marked as slow and are skipped by default. To include them:

```bash
PYAPPROX_RUN_SLOW=1 pytest pyapprox -v --tb=short          # include slow tests (>5s)
PYAPPROX_RUN_SLOWER=1 pytest pyapprox -v --tb=short         # include slower tests (>30s)
PYAPPROX_RUN_SLOWEST=1 pytest pyapprox -v --tb=short        # include all tests
```

To run *only* a specific tier (skipping fast and other tiers), combine the environment variable with `-m`:

```bash
PYAPPROX_RUN_SLOW=1 pytest pyapprox -v --tb=short -m slow       # only @slow_test
PYAPPROX_RUN_SLOWER=1 pytest pyapprox -v --tb=short -m slower    # only @slower_test
PYAPPROX_RUN_SLOWEST=1 pytest pyapprox -v --tb=short -m slowest  # only @slowest_test
```

## Acknowledgements

This research was developed with funding from the Defense Advanced Research Projects Agency (DARPA). The views, opinions and/or findings expressed are those of the author and should not be interpreted as representing the official views or policies of the Department of Defense or the U.S. Government.
