# PyApprox Tutorial Conventions

This document defines mathematical notation, writing style, code conventions, and tutorial structure for all tutorials in `pyapprox.typing.tutorials`.

## Purpose

Ensure consistency across all tutorials with:
- Mathematical notation standards
- Code-math correspondence
- LaTeX macros auto-included in all tutorials
- Writing style guidelines
- Tutorial structure for workshop composition

---

## Tutorial Design Principles

### Atomicity: One Tutorial, One Focus

Each tutorial should have a **singular pedagogical objective**. This enables:
- Flexible workshop composition
- Clear prerequisite chains
- Focused learning outcomes
- Manageable maintenance

**Rule of thumb:** If a tutorial has more than one learning objective that could stand alone, split it.

---

## Tutorial Types

Tutorial types are **guidelines, not rigid categories**. Use the `tutorial_type` tag to indicate primary focus, but types are extensible—new types can be added as needed.

### Core Types

| Type | Purpose | Math | Code | Derivations |
|------|---------|------|------|-------------|
| **Concept** | Understand what and why | Key definitions, intuition | Illustrative only | No |
| **Usage** | Learn API and workflows | Minimal recap | Heavy, working examples | No |
| **Analysis** | Deep theory, proofs | Full derivations, theorems | Optional (verify theory) | Yes |

### Type Definitions

#### Concept Tutorials (`concept`)
- **Purpose:** Build intuition for what a method does and why it matters
- **Audience:** Users new to the topic
- **Math content:** Key definitions, important equations, geometric/visual intuition
- **Code content:** Illustrative figures only (use `#| echo: false`)
- **What to exclude:** Full derivations, proofs, convergence analysis (→ Analysis tutorial)
- **Example:** `sobol_indices_concept.qmd` — defines $S_i$, explains variance decomposition visually

#### Usage Tutorials (`usage`)
- **Purpose:** Teach how to use PyApprox for a task
- **Audience:** Practitioners who want working code
- **Math content:** Brief recap with links to Concept tutorial
- **Code content:** Complete, runnable examples with API patterns
- **Prerequisites:** Should link to corresponding Concept tutorial
- **Example:** `sobol_indices_usage.qmd` — shows `SampleBasedSensitivityAnalysis` API

#### Analysis Tutorials (`analysis`)
- **Purpose:** Rigorous mathematical treatment, derivations, theoretical properties
- **Audience:** Advanced users, researchers, those who want to understand deeply
- **Math content:** Full derivations, theorems, proofs, error bounds, convergence rates
- **Code content:** Optional—only to verify theoretical results numerically
- **Prerequisites:** Should link to corresponding Concept tutorial
- **Example:** `sobol_estimator_analysis.qmd` — derives estimator, proves consistency, shows convergence rates

### Extending Tutorial Types

The type system is extensible. To add a new type:

1. Define it in your tutorial's YAML frontmatter
2. Document its purpose in a comment or this file
3. Add appropriate limits to `scripts/estimate_tutorial_time.py`

Example custom types:
- `workshop` — Curated sequence for live instruction
- `exercise` — Problem sets with solutions
- `reference` — API reference documentation
- `case_study` — Extended real-world example

---

## Tutorial Dependencies

### Flexible Prerequisite Chains

Tutorials declare prerequisites in YAML frontmatter. The dependency graph is flexible:

```yaml
prerequisites:
  - sobol_indices_concept      # Concept → Usage is typical
  - monte_carlo_basics         # Cross-topic dependencies allowed
```

### Typical Patterns

```
Concept ──→ Usage
   │
   └──→ Analysis
```

But patterns can vary:
- Analysis may depend on multiple Concepts
- Usage may depend on multiple other Usage tutorials
- New tutorial types slot into the graph naturally

### Dependency Rules

1. **Usage** tutorials should list relevant **Concept** as prerequisite
2. **Analysis** tutorials should list relevant **Concept** as prerequisite
3. Cross-topic dependencies are allowed and encouraged
4. Circular dependencies are forbidden

---

## Figures in Tutorials

### Figure Sources

Tutorials can include figures from two sources:

**1. Generated figures (hidden code)**
```python
#| echo: false
#| fig-cap: "First four Legendre polynomials"
#| label: fig-legendre-basis
import numpy as np
import matplotlib.pyplot as plt
# ... plotting code ...
plt.show()
```

**2. Static images (uploaded files)**
```markdown
![ANOVA variance decomposition](figures/anova_decomposition.png){#fig-anova width=80%}
```

### Code Block Classification

| Block Type | Syntax | Counts Toward Limit | Time Weight |
|------------|--------|---------------------|-------------|
| Visible code | ```` ```{python} ```` | Yes | 2.5 min |
| Hidden code | `#| echo: false` | No | 0.5 min |
| Static image | `![](path)` | No | 1.5 min |

**Key principle:** Hidden code blocks generate figures without adding cognitive load. They don't count toward the code block limit.

### Figure Organization

Store static images in `library/figures/`:

```
library/
├── figures/
│   ├── anova_decomposition.png
│   ├── sobol_variance_partition.svg
│   └── ...
└── *.qmd
```

### Figure Styling

- **Width:** Use `width=80%` for most figures, `width=60%` for smaller diagrams
- **Captions:** Always include `fig-cap` for generated or `![Caption]` for static
- **Labels:** Use `#| label: fig-name` or `{#fig-name}` for cross-references
- **Format:** Prefer SVG for diagrams, PNG for plots with many points

### Cross-Referencing Figures

```markdown
As shown in @fig-legendre-basis, the polynomials are orthogonal.
```

---

## Tutorial Size and Time Constraints

### The 10-Minute Rule

**Target: Tutorials should be presentable in 5-10 minutes.**

Rationale:
- Matches typical attention spans
- Allows 6-8 tutorials per 90-minute workshop session
- Forces tight focus on a single concept
- Enables flexible mix-and-match workshop composition

### Time Targets by Type

| Type | Target | Suggested Limit | With Exercises | Max Lines |
|------|--------|-----------------|----------------|-----------|
| **Concept** | 5-10 min | 15 min | 15-20 min | 150 |
| **Usage** | 5-10 min | 15 min | 20-30 min | 200 |
| **Analysis** | 10-20 min | 30 min | 30-45 min | 350 |

### Extended Time Exception

Analysis tutorials with indivisible derivations may exceed the suggested limit. When this occurs:

1. Document the reason in YAML frontmatter:
   ```yaml
   extended_time_reason: "Complete derivation of estimator consistency cannot be meaningfully split"
   ```
2. The script will flag but not fail these tutorials
3. Consider whether the derivation can be split into lemmas across multiple tutorials

### Content Limits by Type

| Type | Visible Code | Hidden Code | Equations | Total Figures |
|------|--------------|-------------|-----------|---------------|
| **Concept** | 2 | unlimited | 6 | 6 |
| **Usage** | 8 | 3 | 3 | 6 |
| **Analysis** | 6 | 3 | 15 | 8 |

### Time Budget per Content Element

| Content Element | Time Allowance |
|-----------------|----------------|
| Key equation + explanation | 1-1.5 min |
| Derivation step | 1-1.5 min |
| Visible code block + discussion | 1.5-2 min |
| Hidden code (figure only) | 0.5 min |
| Figure interpretation | 1-1.5 min |
| Prose paragraph | 0.3 min |

### Time Estimation Script

Use the provided script to estimate tutorial presentation time:

```bash
# Single tutorial
python scripts/estimate_tutorial_time.py library/my_tutorial.qmd

# All tutorials
python scripts/estimate_tutorial_time.py library/

# Verbose breakdown
python scripts/estimate_tutorial_time.py library/my_tutorial.qmd --verbose

# CI check (exit 1 if limits exceeded)
python scripts/estimate_tutorial_time.py library/ --check
```

See `scripts/estimate_tutorial_time.py` for implementation details.

### Should I Split This Tutorial?

Ask yourself:

1. **Is estimated time > 10 min?** → Consider splitting (unless Analysis with justification)
2. **Does it have multiple independent learning objectives?** → Split
3. **Does it mix definitions AND derivations?** → Split into Concept + Analysis
4. **Does it mix theory AND heavy API usage?** → Split into Concept/Analysis + Usage
5. **Could someone skip a section entirely?** → That section is a separate tutorial

---

## YAML Frontmatter

### Required Fields

```yaml
---
title: "Tutorial Title"
subtitle: "PyApprox Tutorial Library"
description: "One-line description for catalog"
tutorial_type: concept              # concept | usage | analysis | (extensible)
topic: sensitivity_analysis         # Grouping key for related tutorials
difficulty: intermediate            # beginner | intermediate | advanced
estimated_time: 15                  # Presentation time in minutes
prerequisites:
  - prereq_tutorial_name
tags:
  - sensitivity-analysis
  - sobol-indices
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

### Optional Fields

```yaml
estimated_time_exercises: 25        # Including hands-on time
extended_time_reason: "..."         # Required if estimated_time > suggested limit
```

---

## Content Guidelines by Type

### Concept Tutorials

**Goal:** Reader understands what the method does and why it's useful.

**Structure:**
1. Learning Objectives (3-4 bullets)
2. Prerequisites (with links)
3. Motivation — Why does this problem matter?
4. Key Definitions — Core equations, notation
5. Visual Intuition — Figures illustrating the concept
6. Simple Example (optional) — Minimal worked example
7. Key Takeaways
8. Exercises (conceptual, not coding)
9. Next Steps → links to Usage and/or Analysis

**Style:**
- Focus on intuition over rigor
- Use figures liberally (hidden code is fine)
- Equations should be "memorable" definitions, not derivations
- Keep code hidden unless it teaches something

**What belongs elsewhere:**
- Full derivations → Analysis tutorial
- API patterns → Usage tutorial
- Convergence proofs → Analysis tutorial

### Usage Tutorials

**Goal:** Reader can use PyApprox to accomplish a task.

**Structure:**
1. Learning Objectives (3-4 bullets)
2. Prerequisites → link to Concept tutorial
3. Quick Recap (1 paragraph summary of concept)
4. Setup — Imports, backend, data
5. Basic Usage — Minimal working example
6. Common Patterns — Variations, options, configurations
7. Interpretation — How to read the outputs
8. Troubleshooting / Common Errors
9. Key Takeaways
10. Exercises (coding)
11. Next Steps

**Style:**
- Code-first approach
- Every code block should be runnable
- Explain API choices, not math
- Type hints in all function signatures
- Show output and explain it

### Analysis Tutorials

**Goal:** Reader understands the mathematical foundations deeply.

**Structure:**
1. Learning Objectives (3-4 bullets)
2. Prerequisites → link to Concept tutorial
3. Mathematical Setup — Notation, assumptions
4. Main Results — Theorems, lemmas
5. Derivations — Step-by-step proofs
6. Theoretical Properties — Convergence, error bounds
7. Numerical Verification (optional) — Code that confirms theory
8. Discussion — Implications, limitations
9. Key Takeaways
10. Exercises (proofs, extensions)
11. References

**Style:**
- Academic tone, rigorous
- Number important equations for reference
- Proofs can use collapsible blocks if lengthy
- Code is optional—only to verify theory numerically
- Include references to literature

---

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
- **Exception:** 0-indexed when mathematical convention dictates (e.g., polynomial degree in PCE)

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

### Surrogate Notation

| Notation | Meaning | LaTeX Macro |
|----------|---------|-------------|
| $f_N$ | Surrogate trained on $N$ samples | `\surrogate{N}` or `\surr` |

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

**Convention:** Always use subscripts on $\E$, $\Var$, and $\Cov$ to indicate which random variable the statistic is computed with respect to.

### Sensitivity Analysis

| Notation | Meaning | LaTeX Macro |
|----------|---------|-------------|
| $S_i$ | First-order Sobol index | `\Sobol{i}` |
| $S_i^T$ | Total-order Sobol index | `\SobolT{i}` |

---

## Array Shape Conventions

All arrays follow PyApprox conventions:

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

---

## LaTeX Macros

All tutorials automatically include macros from `_macros.tex` (PDF) or `_macros_html.tex` (HTML).

Key macros:

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

---

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
| SA | Sensitivity Analysis |

---

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
samples = bkd.array([[0.3, 0.5], [0.4, 0.6]])
result = bkd.sum(values)
```

### Benchmark Usage

For UQ tutorials, use benchmark instances:
```python
from pyapprox.typing.benchmarks import lotka_volterra_3species
from pyapprox.typing.util.backends.numpy import NumpyBkd

bkd = NumpyBkd()
benchmark = lotka_volterra_3species(bkd)
```

---

## Writing Style

### Tone
- Direct and technical
- Avoid excessive hedging ("might", "could potentially")
- Use active voice where possible
- No emojis

### Code Comments
- Explain the "why" not the "what"
- Keep comments concise
- Use type hints in function signatures

---

## Tutorial Naming

**No Numeric Prefixes:** Tutorial filenames must NOT include numeric prefixes:

- Correct: `gp_integration_moments.qmd`
- Incorrect: `01_gp_integration_moments.qmd`

Ordering is controlled via `_quarto.yml` navigation configuration.

---

## File Organization

```
pyapprox/typing/tutorials/
├── CONVENTIONS.md           # This file
├── BUILD.md                 # Build and deployment guide
├── _macros.tex              # LaTeX macros for PDF
├── _macros_html.tex         # MathJax macros for HTML
├── scripts/
│   ├── estimate_tutorial_time.py   # Time estimation tool
│   └── generate_workshop_index.py
├── library/
│   ├── _quarto.yml
│   ├── index.qmd
│   ├── figures/             # Static images
│   └── *.qmd                # Tutorial files
└── workshops/
    ├── index.qmd
    └── */                   # Workshop directories
```

---

## Related Documents

- `BUILD.md` - Building and deploying tutorials
- `scripts/estimate_tutorial_time.py` - Tutorial time estimation
- `pyapprox/typing/benchmarks/CONVENTIONS.md` - Benchmark conventions
- `CLAUDE.md` - Project-wide coding conventions
