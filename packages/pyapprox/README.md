# PyApprox

[![PyPI](https://img.shields.io/pypi/v/pyapprox.svg)](https://pypi.org/project/pyapprox/)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/sandialabs/pyapprox/blob/main/LICENSE)

High-dimensional function approximation, uncertainty quantification, and
experimental design.

PyApprox provides tools for high-dimensional approximation, uncertainty
quantification, and decision-making under uncertainty, aimed at problems
with large parameter spaces and expensive simulation models.

**[Documentation](https://sandialabs.github.io/pyapprox/)** |
**[Repository](https://github.com/sandialabs/pyapprox)** |
**[Paper](https://doi.org/10.1016/j.envsoft.2023.105825)**

## What is in this package

- **Surrogate modeling** — polynomial chaos expansions, Gaussian processes
  (single- and multi-output, DAG-structured), function trains, sparse grids
- **Multi-fidelity estimation** — approximate control variates, MLMC, MFMC,
  MLBLUE, and group ACV
- **Bayesian experimental design** — KL-based and goal-oriented optimal
  design with gradient-based optimization
- **Bayesian inference** — MCMC sampling and conjugate posterior analysis
- **Sensitivity analysis** — Sobol indices, Morris screening
- **Probability and risk** — variable transformations, risk measures, and
  random field representations (KLE)
- **PDE solvers** — collocation and Galerkin finite elements for
  advection-diffusion-reaction, Helmholtz, Stokes, and elasticity
- **Optimization** — implicit function differentiation and adjoint methods

All code is fully typed, supports NumPy and PyTorch backends, and preserves
PyTorch autograd computation graphs for automatic differentiation.

## Installation

```bash
pip install pyapprox                    # core library
pip install "pyapprox[runtime-extras]"  # plus all runtime extras
```

Individual extras are also available: `fem` (scikit-fem), `umbridge`,
`numba`, `parallel`, and `cvxpy`.

## Quick start

```python
from pyapprox.util.backends.numpy import NumpyBkd
from pyapprox.interface.functions.fromcallable.function import (
    FunctionFromCallable,
)
from pyapprox.probability import UniformMarginal, IndependentJoint
from pyapprox.surrogates.sparsegrids import create_basis_factories
from pyapprox.surrogates.sparsegrids.isotropic_fitter import (
    IsotropicSparseGridFitter,
)
from pyapprox.surrogates.sparsegrids.subspace_factory import (
    TensorProductSubspaceFactory,
)
from pyapprox.surrogates.affine.indices import LinearGrowthRule

bkd = NumpyBkd()


def target(samples):
    x, y = samples[0], samples[1]
    return bkd.reshape(x**3 + x * y + y**2, (1, -1))


func = FunctionFromCallable(1, 2, target, bkd)

marginals = [UniformMarginal(-1.0, 1.0, bkd) for _ in range(2)]
joint = IndependentJoint(marginals, bkd)
factories = create_basis_factories(joint.marginals(), bkd, "gauss")
growth = LinearGrowthRule(scale=1, shift=1)
tp_factory = TensorProductSubspaceFactory(bkd, factories, growth)
fitter = IsotropicSparseGridFitter(bkd, tp_factory, level=3)
samples = fitter.get_samples()
surrogate = fitter.fit(func(samples)).surrogate

approx_values = surrogate(joint.rvs(100))
```

## Related packages

`pyapprox` is developed in a monorepo alongside
[`pyapprox-benchmarks`](https://pypi.org/project/pyapprox-benchmarks/)
(benchmark problems) and
[`pyapprox-tutorials`](https://pypi.org/project/pyapprox-tutorials/)
(tutorial helpers). See the
[repository](https://github.com/sandialabs/pyapprox) for development setup.

## Citation

```bibtex
@article{JAKEMAN2023105825,
  title = {PyApprox: A software package for sensitivity analysis, Bayesian
           inference, optimal experimental design, and multi-fidelity
           uncertainty quantification and surrogate modeling},
  author = {J.D. Jakeman},
  journal = {Environmental Modelling \& Software},
  volume = {170},
  pages = {105825},
  year = {2023},
  doi = {10.1016/j.envsoft.2023.105825}
}
```

## License

MIT. See [LICENSE](https://github.com/sandialabs/pyapprox/blob/main/LICENSE).
