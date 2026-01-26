# PyApprox Tutorial Conventions

This document defines mathematical notation, writing style, and code conventions for all tutorials in `pyapprox.typing.tutorials`.

## Purpose

Ensure consistency across all tutorials with:
- Mathematical notation standards
- Code-math correspondence
- LaTeX macros auto-included in all tutorials
- Writing style guidelines

## Mathematical Notation

### Vectors and Matrices

| Notation | Meaning | LaTeX Macro |
|----------|---------|-------------|
| $\mathbf{x}$ | Vector (bold lowercase) | `\vc{x}` |
| $\mathbf{A}$ | Matrix (bold uppercase) | `\mt{A}` |
| $\mathcal{T}$ | Tensor (calligraphic) | `\ts{T}` |
| $\boldsymbol{\theta}$ | Parameter vector | `\params` |

### Dimension Naming

| Math | Code Variable | PyApprox Method | Description |
|------|---------------|-----------------|-------------|
| $d_\theta$ | `num_params` | `nparams()` | Number of parameters |
| $d_x$ | `nvars` | `nvars()` | Number of input variables |
| $d_y$ | `nqoi` | `nqoi()` | Number of outputs/QoIs |
| $N$ | `num_samples` | - | Number of samples (**RESERVED**) |

**Critical:** $N$ must ONLY be used for number of samples throughout all tutorials.

### Sample Notation

| Notation | Meaning | Example |
|----------|---------|---------|
| $N$ | Number of samples | $N = 1000$ |
| $\theta^{(k)}$ | $k$-th sample/realization of $\theta$ | $k = 1, \ldots, N$ |
| $\mathbf{x}^{(k)}$ | $k$-th sample of vector $\mathbf{x}$ | |
| $\theta_i$ | $i$-th component of $\boldsymbol{\theta}$ | |
| $\theta_i^{(k)}$ | $i$-th component of $k$-th sample | |
| $\boldsymbol{\Theta}$ | Matrix of samples | $\boldsymbol{\Theta} = [\boldsymbol{\theta}^{(1)}, \ldots, \boldsymbol{\theta}^{(N)}] \in \mathbb{R}^{d_\theta \times N}$ |

### Random Variable vs Realization

- Default to realization notation (most UQ literature does this)
- When distinction is needed, use tilde for random variable: $\tilde{\theta}$ vs $\theta$
- Macro: `\rv{theta}` produces $\tilde{\theta}$

### Indexing Convention

- **1-indexed by default** for samples ($k = 1, \ldots, N$) and components ($i = 1, \ldots, d_\theta$)
- **Exception:** 0-indexed when mathematical convention dictates (e.g., polynomial degree in PCE where index equals degree of basis term)

### Function Argument Convention

Use comma separation for all function arguments:
- Correct: $f(\mathbf{x}, t, \boldsymbol{\theta})$
- Incorrect: $f(\mathbf{x}; t; \boldsymbol{\theta})$ (do NOT use semicolons)

### Function Types

| Type | Math Notation | PyApprox Protocol |
|------|---------------|-------------------|
| Parametric model | $f(\boldsymbol{\theta})$ | `FunctionProtocol` |
| Spatial function | $f(\mathbf{x})$ | `FunctionProtocol` |
| Parameterized function | $f(\mathbf{x}, \boldsymbol{\theta})$ | `ParameterizedFunctionProtocol` |
| ODE right-hand side | $\dot{u} = f(u, t)$ | `GalerkinPhysicsProtocol` |

### QoI Chain Notation

For uncertainty propagation through models:

| Symbol | Role | Description |
|--------|------|-------------|
| $f$ | Forward model | ODE/PDE solver: $f(\mathbf{x}, t, \boldsymbol{\theta})$ |
| $g$ | Functional | Extracts QoI from full solution: $g: \mathcal{F} \to \mathbb{R}^{d_y}$ |
| $q$ | QoI map | Composed parameter-to-QoI: $q(\boldsymbol{\theta}) = g(f(\cdot, \cdot, \boldsymbol{\theta}))$ |

**Example for Lotka-Volterra:**
- $f(t, \boldsymbol{\theta})$: ODE solution giving population densities over time
- $g$: Extract maximum predator population
- $q(\boldsymbol{\theta}) = \max_t f_{\text{predator}}(t, \boldsymbol{\theta})$

### Surrogate Notation

| Notation | Meaning | LaTeX Macro |
|----------|---------|-------------|
| $f_N$ | Surrogate trained on $N$ samples | `\surrogate{N}` or `\surr` |

The subscript $N$ indicates dependence on training sample size.

### Jacobian Naming

| Math | Code Method | Description |
|------|-------------|-------------|
| $\partial f/\partial \mathbf{x}$ | `jacobian()` | w.r.t. primary variables |
| $\partial f/\partial \boldsymbol{\theta}$ | `jacobian_wrt_params()` | w.r.t. parameters |

### Probability Notation

| Notation | Meaning | LaTeX Macro |
|----------|---------|-------------|
| $\mathbb{E}_\theta[f(\theta)]$ | Expectation w.r.t. $\theta$ | `\E_\theta[f(\theta)]` |
| $\mathbb{V}_X[g(X)]$ | Variance w.r.t. $X$ | `\Var_X[g(X)]` |
| $\mathrm{Cov}_\theta(X, Y)$ | Covariance w.r.t. $\theta$ | `\Cov_\theta(X, Y)` |
| $p(\theta)$ | PDF | `\pdf(\theta)` |
| $\mathcal{N}(\mu, \sigma^2)$ | Normal distribution | `\normal(\mu, \sigma^2)` |
| $\mathcal{U}(a, b)$ | Uniform distribution | `\uniform(a, b)` |

**Convention:** Always use subscripts on $\E$, $\Var$, and $\Cov$ to indicate which random variable the statistic is computed with respect to. This is critical when multiple random variables are present:

- $\E_\theta[q(\theta)]$ - expectation of QoI over parameter uncertainty
- $\E_X[f(X, \theta)]$ - expectation over input $X$ for fixed $\theta$
- $\E_{\theta, X}[f(X, \theta)]$ - expectation over both (joint)
- $\Var_\theta[\E_X[f(X, \theta)]]$ - variance of conditional expectation (law of total variance)

**Covariance subscript convention:**

- $\Cov_\theta(X, Y)$ - covariance where both $X$ and $Y$ depend on the same random variable $\theta$
- $\Cov_{\theta_1, \theta_2}(X, Y)$ - covariance where $X$ depends on $\theta_1$ and $Y$ depends on $\theta_2$ (joint distribution)
- For variance as a special case: $\Var_\theta[X] = \Cov_\theta(X, X)$

### Sensitivity Analysis

| Notation | Meaning | LaTeX Macro |
|----------|---------|-------------|
| $S_i$ | First-order Sobol index | `\Sobol{i}` |
| $S_i^T$ | Total-order Sobol index | `\SobolT{i}` |

## Array Shape Conventions

All arrays follow PyApprox conventions from `CLAUDE.md`:

| Array | Shape | Note |
|-------|-------|------|
| Input samples | `(nvars, nsamples)` | Samples as columns |
| Output values | `(nqoi, nsamples)` | QoIs as rows |
| Single sample | `(nvars, 1)` | Always 2D |
| Jacobian (single) | `(nqoi, nvars)` | Always 2D |
| Jacobian (batch) | `(nsamples, nqoi, nvars)` | 3D |
| Hyperparameters | `(nparams,)` | ONLY exception: 1D |

**Critical Rules:**
- Single samples are always 2D: `(nvars, 1)` not `(nvars,)`
- Jacobians are never 1D: `(1, nvars)` for scalar functions
- Only hyperparameters use 1D arrays

## LaTeX Macros

All tutorials automatically include macros from `_macros.tex` (PDF) or `_macros_html.tex` (HTML).

See `_macros.tex` for the complete list. Key macros:

```latex
% Vectors
\params       % boldsymbol theta
\inputs       % mathbf x
\outputs      % mathbf y

% Dimensions
\dparams      % d_theta
\dinputs      % d_x
\doutputs     % d_y
\nsamples     % N (RESERVED for samples)

% Probability
\E            % expectation
\Var          % variance
\normal       % N for normal distribution

% Sensitivity
\Sobol{i}     % S_i
\SobolT{i}    % S_i^T

% Surrogates
\surr         % f_N
\surrogate{M} % f_M
```

## Acronym Policy

Define each acronym at first use in each tutorial. Standard acronyms:

| Acronym | Full Form |
|---------|-----------|
| UQ | Uncertainty Quantification |
| MC | Monte Carlo |
| PCE | Polynomial Chaos Expansion |
| GP | Gaussian Process |
| QoI | Quantity of Interest |
| PDF | Probability Density Function |
| CDF | Cumulative Distribution Function |
| ODE | Ordinary Differential Equation |
| PDE | Partial Differential Equation |
| MCMC | Markov Chain Monte Carlo |
| SA | Sensitivity Analysis |
| OUU | Optimization Under Uncertainty |
| AVaR | Average Value at Risk |
| HMC | Hamiltonian Monte Carlo |

## Tutorial Structure

Each tutorial should follow this template (target: 200-400 lines):

### YAML Frontmatter

```yaml
---
title: "Tutorial Title"
subtitle: "PyApprox Tutorial Library"
description: "One-line description for catalog"
prerequisites:
  - prereq_tutorial_name
tags:
  - tag1
  - tag2
format:
  html:
    code-fold: false
    code-tools: true
    toc: true
execute:
  echo: true
  warning: false
jupyter: python3
---
```

### Content Sections

1. **Learning Objectives** (3-5 bullet points)
2. **Prerequisites** (narrative form, link to prerequisite tutorials)
3. **Conceptual Introduction** (1-2 paragraphs, math where needed)
4. **Worked Example** (Lotka-Volterra based when applicable)
5. **Key Takeaways** (bullet summary)
6. **Exercises** (2-4 exercises, labeled by difficulty)
7. **Next Steps** (links to related tutorials)

### Exercise Difficulty Labels

- **(Easy)** - Direct application of tutorial content
- **(Medium)** - Requires synthesis or small modifications
- **(Challenge)** - Requires deeper understanding or research

## Writing Style

### Tone
- Direct and technical
- Avoid excessive hedging ("might", "could potentially")
- Use active voice where possible
- No emojis unless explicitly requested

### Code Comments
- Explain the "why" not the "what"
- Keep comments concise
- Use type hints in function signatures

### Math-Code Correspondence
When showing math and code together, use consistent variable names:

```python
# Mathematical model: q(theta) = E[f(X; theta)]
# where X ~ p(x) is the input distribution

def estimate_qoi_mean(theta: Array, samples: Array) -> float:
    """Estimate E[f(X; theta)] via Monte Carlo.

    Parameters
    ----------
    theta : Array
        Parameter vector, shape (d_theta, 1).
    samples : Array
        Input samples X^{(k)}, shape (d_x, N).

    Returns
    -------
    float
        Monte Carlo estimate of the mean.
    """
    # f_values shape: (1, N) - one QoI per sample
    f_values = model(samples, theta)
    return float(bkd.mean(f_values))
```

## Code Conventions

### Imports

Standard import order:
```python
# Standard library
from typing import Tuple

# Third-party
import numpy as np

# pyapprox
from pyapprox.typing.util.backends.numpy import NumpyBkd
from pyapprox.typing.benchmarks import lotka_volterra_3species
```

### Backend Usage

Always use backend-agnostic operations:
```python
bkd = NumpyBkd()
# Good: backend method
samples = bkd.array([[0.3, 0.5], [0.4, 0.6]])
result = bkd.sum(values)

# Bad: numpy directly (unless showing comparison)
samples = np.array([[0.3, 0.5], [0.4, 0.6]])
```

### Benchmark Usage

For UQ tutorials, use benchmark instances:
```python
from pyapprox.typing.benchmarks import lotka_volterra_3species
from pyapprox.typing.util.backends.numpy import NumpyBkd

bkd = NumpyBkd()
benchmark = lotka_volterra_3species(bkd)

# Access components
prior = benchmark.prior()
domain = benchmark.domain()
gt = benchmark.ground_truth()
```

Tutorial Naming
No Numeric Prefixes
Tutorial filenames must NOT include numeric prefixes or ordering identifiers:

Correct: gp_integration_moments.qmd, gp_integration_sensitivity.qmd
Incorrect: 01_gp_integration_moments.qmd, 02_gp_integration_sensitivity.qmd

Rationale: Tutorial order may change as new material is added. Numeric prefixes create maintenance burden and misleading implied order. Use descriptive names only; ordering is controlled via _quarto.yml navigation configuration.

## File Organization

```
pyapprox/typing/tutorials/
├── CONVENTIONS.md           # This file
├── _macros.tex              # LaTeX macros for PDF
├── _macros_html.tex         # MathJax macros for HTML
├── library/
│   ├── _quarto.yml          # Quarto project config
│   ├── index.qmd            # Library catalog
│   ├── setup_environment.qmd
│   ├── uq_problem_framing.qmd
│   └── ...                  # Other tutorials
├── workshops/
│   ├── index.qmd            # Workshop catalog
│   └── intro_to_uq_2025/
│       ├── _quarto.yml
│       ├── index.qmd
│       └── workshop.yml
└── scripts/
    └── generate_workshop_index.py
```

## Related Documents

- `pyapprox/typing/benchmarks/CONVENTIONS.md` - Benchmark interface conventions
- `CLAUDE.md` - Project-wide coding conventions
- `pyapprox/typing/quarto-docs/developer-guide/array-conventions.qmd` - Array shape documentation
