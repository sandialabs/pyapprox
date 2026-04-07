# BOED Tutorial Structure  <!-- v2: session-2 tutorials added -->

## Dependency Graph

```
experimental_design_intro  (existing)
    │
    ├── boed_kl_concept              concept    ~8 min   What is EIG-BOED?
    │       ├── boed_kl_estimator    analysis   ~10 min  Double-loop estimator derivation
    │       ├── boed_kl_gradients   analysis   ~15 min  Reparameterization trick + C1/C2/C3 gradient
    │       ├── boed_kl_usage       usage      ~10 min  KLOEDDiagnostics API + convergence
    │       └── boed_kl_qmc         usage      ~10 min  MC vs Halton QMC convergence
    │
    └── boed_pred_concept            concept    ~8 min   What is goal-oriented / risk-aware OED?
            ├── boed_pred_gaussian_analysis  analysis ~25 min  Posterior expressions, entropic risk, AVaR
            │       └── boed_pred_lognormal_analysis analysis ~15 min  Lognormal std-dev derivations
            └── boed_pred_usage      usage      ~12 min  PredOEDDiagnostics API + convergence
```

## Source-to-Tutorial Mapping

| Source file                                | Tutorial(s)                                          |
|--------------------------------------------|------------------------------------------------------|
| plot_bayesoed4param_formulation.py         | boed_kl_concept.qmd + boed_kl_estimator.qmd + boed_kl_gradients.qmd |
| plot_bayesoed4param_verification.py        | boed_kl_usage.qmd                                    |
| plot_bayesoed4pred_formulation.py          | boed_pred_concept.qmd                                |
| plot_bayesoed4pred_gaussian_expressions.py | boed_pred_gaussian_analysis.qmd                      |
| plot_bayesoed4pred_lognormal_expressions.py| boed_pred_lognormal_analysis.qmd                     |
| plot_bayesoed4pred_verification.py         | boed_pred_usage.qmd                                  |

## New Typing API Required

### Current API
```python
from pyapprox_benchmarks.instances.oed.linear_gaussian import (
    build_linear_gaussian_kl_benchmark,
)
from pyapprox_benchmarks.instances.oed.linear_gaussian_pred import (
    build_linear_gaussian_pred_benchmark,
)
from pyapprox.expdesign.diagnostics import KLOEDDiagnostics
from pyapprox.expdesign.diagnostics import (
    PredictionOEDDiagnostics,
    create_prediction_oed_diagnostics,
)
```
