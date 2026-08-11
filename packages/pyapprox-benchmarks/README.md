# PyApprox Benchmarks

[![PyPI](https://img.shields.io/pypi/v/pyapprox-benchmarks.svg)](https://pypi.org/project/pyapprox-benchmarks/)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/sandialabs/pyapprox/blob/main/LICENSE)

Benchmark problems for the [PyApprox](https://pypi.org/project/pyapprox/)
library.

This package collects reference problems used to test and compare
uncertainty quantification, surrogate modeling, and experimental design
methods.

**[Documentation](https://sandialabs.github.io/pyapprox/)** |
**[Repository](https://github.com/sandialabs/pyapprox)**

## Functions, problems, and benchmarks

The package is organized in three layers, and the distinction is worth
knowing before reaching for something:

| Layer | What it is | Where |
|-------|-----------|-------|
| **Function** | Pure math — no domain, no prior. Evaluates, and supplies derivatives where available. | `functions/` |
| **Problem** | One or more functions bound to what a task needs: domain, prior, constraints, noise model, design space. **No known answers.** | `problems/` |
| **Benchmark** | A problem plus typed ground-truth methods — the reference values a method is scored against. | use-case directories |

The dividing line is the ground truth: if there is no known answer, it is a
problem, not a benchmark. A benchmark constructs its problem internally, so
you instantiate one name and ask it for both:

```python
benchmark = IshigamiBenchmark(bkd)
problem = benchmark.problem()   # the function, domain, and prior
exact = benchmark.mean()        # the ground truth to score against
```

Problems cover forward UQ, quadrature, constrained optimization, Bayesian
inference, and optimal experimental design. Benchmarks are grouped by the
use case they serve: `sensitivity/`, `quadrature/`, `optimization/`,
`expdesign/`, `statest/` (statistical estimation), `pde/`, and `ode/`.
Full conventions are in `CONVENTIONS.md` in the package source.

## Installation

```bash
pip install pyapprox-benchmarks
```

Some PDE benchmarks require a finite element backend:

```bash
pip install "pyapprox-benchmarks[fem]"
```

Installing this package pulls in `pyapprox` as a dependency.

## Related packages

Developed in a monorepo alongside
[`pyapprox`](https://pypi.org/project/pyapprox/) (the core library) and
[`pyapprox-tutorials`](https://pypi.org/project/pyapprox-tutorials/)
(tutorial helpers). See the
[repository](https://github.com/sandialabs/pyapprox) for development setup.

## License

MIT. See [LICENSE](https://github.com/sandialabs/pyapprox/blob/main/LICENSE).
