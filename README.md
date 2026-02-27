[![Tests](https://github.com/sandialabs/pyapprox/actions/workflows/tests.yml/badge.svg)](https://github.com/sandialabs/pyapprox/actions/workflows/tests.yml)
[![Lint](https://github.com/sandialabs/pyapprox/actions/workflows/lint.yml/badge.svg)](https://github.com/sandialabs/pyapprox/actions/workflows/lint.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

# PyApprox

PyApprox provides flexible and efficient tools for high-dimensional approximation, uncertainty quantification, and decision-making under uncertainty. It implements methods addressing various issues surrounding high-dimensional parameter spaces and limited evaluations of expensive simulation models, with the goal of facilitating simulation-aided knowledge discovery, prediction, and design.

**[Documentation](https://sandialabs.github.io/pyapprox/)** | **[Tutorials](https://sandialabs.github.io/pyapprox/)** | **[Paper](https://doi.org/10.1016/j.envsoft.2023.105825)**

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

## Quick Start

```python
import numpy as np
from pyapprox.util.backends.numpy import NumpyBkd
from pyapprox.interface.functions.fromcallable.function import FunctionFromCallable
from pyapprox.probability import UniformMarginal, IndependentJoint
from pyapprox.surrogates.sparsegrids import create_basis_factories
from pyapprox.surrogates.sparsegrids.isotropic_fitter import IsotropicSparseGridFitter
from pyapprox.surrogates.sparsegrids.subspace_factory import TensorProductSubspaceFactory
from pyapprox.surrogates.affine.indices import LinearGrowthRule

bkd = NumpyBkd()

# Define a 2D function using the FunctionProtocol
def target(samples):
    x, y = samples[0], samples[1]
    return bkd.reshape(x**3 + x*y + y**2, (1, -1))

func = FunctionFromCallable(1, 2, target, bkd)

# Build a sparse grid surrogate
marginals = [UniformMarginal(-1.0, 1.0, bkd) for _ in range(2)]
joint = IndependentJoint(marginals, bkd)
factories = create_basis_factories(joint.marginals(), bkd, "gauss")
growth = LinearGrowthRule(scale=1, shift=1)
tp_factory = TensorProductSubspaceFactory(bkd, factories, growth)
fitter = IsotropicSparseGridFitter(bkd, tp_factory, level=3)
samples = fitter.get_samples()
result = fitter.fit(func(samples))
surrogate = result.surrogate

# Evaluate surrogate at new points
test_pts = joint.rvs(100)
approx_values = surrogate(test_pts)
```

## Requirements

- Python >= 3.10
- NumPy >= 2.0, SciPy >= 1.11, PyTorch >= 2.5
- matplotlib, sympy, networkx

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

To run tests in parallel (requires `pytest-xdist`):

```bash
pytest pyapprox -v --tb=short -n auto    # use all CPUs
pytest pyapprox -v --tb=short -n 4       # use 4 workers
```

## Building Documentation

The tutorial site is built with [Quarto](https://quarto.org/). Install it, then:

```bash
cd tutorials
./build.sh                    # build with cached results (freeze)
./build.sh --execute          # force re-execute all code
./build.sh --no-execute       # skip execution, use cache only
./build.sh -j auto            # parallel execution (auto-detect CPUs)
./build.sh -j 4               # parallel execution with 4 workers
./build.sh --notebooks        # also generate downloadable .ipynb files
./build.sh --serve            # start local server after build
./build.sh --skip=pacv_usage  # skip a specific tutorial
```

Output is written to `tutorials/library/_site/`.

## Linting

```bash
ruff check pyapprox/          # style and import checks
mypy pyapprox/                # static type checking
```

## Contributing

Contributions are welcome. Please:

1. Fork the repository and create a feature branch
2. Ensure all tests pass (including slow): `PYAPPROX_RUN_SLOWEST=1 pytest pyapprox -v --tb=short`
3. Ensure no lint errors: `ruff check pyapprox/`
4. Submit a pull request

## Citation

If you use PyApprox in your research, please cite:

```bibtex
@article{JAKEMAN2023105825,
  title = {PyApprox: A software package for sensitivity analysis, Bayesian inference,
           optimal experimental design, and multi-fidelity uncertainty quantification
           and surrogate modeling},
  author = {J.D. Jakeman},
  journal = {Environmental Modelling \& Software},
  volume = {170},
  pages = {105825},
  year = {2023},
  doi = {10.1016/j.envsoft.2023.105825}
}
```

## License

PyApprox is licensed under the [MIT License](LICENSE).

## Acknowledgements

This research was developed with funding from the Defense Advanced Research Projects Agency (DARPA), the U.S. Department of Energy Office of Science Advanced Scientific Computing Research (ASCR) program, and the Sandia National Laboratories Laboratory Directed Research and Development (LDRD) program. The views, opinions and/or findings expressed are those of the author and should not be interpreted as representing the official views or policies of the Department of Defense or the U.S. Government.
