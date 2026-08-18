# Migrating from PyApprox 1.x to 2.0

PyApprox 2.0 is a ground-up redesign, not an incremental release. The
repository became a monorepo publishing **three PyPI packages**, the entire
API moved from functional factories to typed, backend-generic classes, and
most subpackages were renamed or reorganized. Code written against 1.x will
not run against 2.0 without changes. This guide covers everything a 1.x user
needs: environment requirements, the package split, the new backend system,
an old→new import map, and area-by-area before/after examples.

This guide applies to `pyapprox` 2.0.0 (released 2026-06-07). The last 1.x
release was 1.0.3.

## At a glance

| | v1.x | v2.0 |
|---|---|---|
| Python | >= 3.6 | **>= 3.11** (3.11 / 3.12 / 3.13) |
| NumPy | >= 1.16 (NumPy 1.x) | **>= 2.0** |
| PyPI packages | `pyapprox` | `pyapprox`, `pyapprox-benchmarks`, `pyapprox-tutorials` |
| Build | setuptools + Cython (compiled extensions) | hatchling + hatch-vcs (pure Python; no compiler needed) |
| Array handling | plain NumPy arrays | backend objects (`NumpyBkd` / `TorchBkd`) passed explicitly |
| API style | functional factories (`setup_benchmark`, `get_estimator`, `approximate`) | direct class instantiation |
| Docs/tutorials | Sphinx + sphinx-gallery | Quarto site built from `pyapprox-tutorials` |

## 1. Requirements and installation

Python 3.11+ and NumPy 2.0+ are now required, so most users will need a
fresh environment rather than an in-place upgrade:

```bash
pip install pyapprox              # core library
pip install pyapprox-benchmarks   # benchmark functions (was pyapprox.benchmarks)
pip install pyapprox-tutorials    # tutorial figure/code helpers
```

Cython compiled extensions are gone entirely — installation no longer needs
a C compiler. Performance-critical kernels now use optional Numba
implementations with pure-Python fallbacks.

Optional extras were restructured. The single v1 `docs` extra became:

| Extra | Provides |
|---|---|
| `pyapprox[fem]` | finite-element models (scikit-fem >= 9.0) |
| `pyapprox[umbridge]` | UM-Bridge model interface |
| `pyapprox[numba]` | Numba JIT acceleration |
| `pyapprox[parallel]` | joblib / mpire parallel evaluation |
| `pyapprox[cvxpy]` | convex-optimization-based methods |
| `pyapprox[rol]` | Rapid Optimization Library bindings |
| `pyapprox[runtime-extras]` | all runtime extras above |
| `pyapprox[dev]` | everything, plus test/lint/docs tooling |

Dependency changes to be aware of: `scikit-learn` is **no longer a
dependency** (code paths that relied on it were removed or rewritten);
`torch >= 2.0` and `networkx >= 3.0` are core dependencies; `numba`,
`scikit-fem`, and `umbridge` moved from core to extras.

Versioning is now derived from git tags (`hatch-vcs`). Installing from an
untagged source checkout produces a local version like `2.0.1.dev3+g1234abc`.

## 2. The three-package split

The v1 `pyapprox.benchmarks` subpackage and the tutorial gallery moved out
of the core library:

| v1 location | v2 package | v2 import name |
|---|---|---|
| `pyapprox.*` (core) | `pyapprox` | `pyapprox` |
| `pyapprox.benchmarks` | `pyapprox-benchmarks` | `pyapprox_benchmarks` |
| `tutorials/` gallery | `pyapprox-tutorials` | `pyapprox_tutorials` |

Note the underscore/hyphen distinction: install `pyapprox-benchmarks`,
import `pyapprox_benchmarks`.

## 3. The backend system — the core paradigm shift

In v1, everything consumed and returned plain NumPy arrays. In v2, nearly
every class is generic over an array type and takes an explicit backend
object (`bkd`) that supplies array creation and linear algebra. Two backends
ship with 2.0:

```python
from pyapprox.util.backends.numpy import NumpyBkd   # NumPy ndarrays
from pyapprox.util.backends.torch import TorchBkd   # torch.Tensors (GPU-capable)

bkd = NumpyBkd()
x = bkd.array([[1.0, 2.0, 3.0]])   # create arrays through the backend
```

Practical consequences when migrating:

1. Most constructors now require a `bkd` argument. Create one `NumpyBkd()`
   (or `TorchBkd()`) near the top of your script and pass it everywhere.
2. Objects built on different backends do not mix; pick one per workflow.
3. Swapping `NumpyBkd` for `TorchBkd` is how you get autograd/GPU support —
   the v1 pattern of separate torch-specific modules (e.g. `autogp`) is gone.

## 4. Subpackage renames

Top-level map from v1 to v2 (`pyapprox.` prefix omitted):

| v1 subpackage | v2 subpackage | Notes |
|---|---|---|
| `variables` | `probability` | new marginal/joint class hierarchy (see §5.1) |
| `multifidelity` | `statest` | "statistical estimation"; factory removed (see §5.2) |
| `bayes` | `inverse` | Bayesian inference |
| `analysis` | `sensitivity` | variance-based SA, active subspaces |
| `benchmarks` | — | moved to the `pyapprox-benchmarks` package |
| `cython` | — | removed (optional Numba kernels instead) |
| `surrogates` | `surrogates` | kept, but internally reorganized (see §5.3) |
| `expdesign` | `expdesign` | kept, greatly expanded |
| `optimization` | `optimization` | kept, reorganized |
| `interface` | `interface` | kept; function protocols added (see §5.5) |
| `pde` | `pde` | kept; adds `collocation`, `field_maps`, `zoo`, ... |
| `util` | `util` | slimmed down; adds `util.backends` |
| — | `ode` | new: ODE steppers/operators |
| — | `generative` | new: generative models incl. flow matching |
| — | `risk` | new top-level home for risk measures |

The v1 habit of `import pyapprox as pya; pya.<thing>` no longer works — the
root `pyapprox` namespace exposes only `__version__`. Import from the
specific subpackage.

## 5. Area-by-area migration

### 5.1 Random variables → `pyapprox.probability`

`IndependentMarginalsVariable` no longer exists. Marginals are now explicit
classes (implementing `MarginalProtocol`), scipy frozen distributions are
wrapped rather than consumed directly, and joints take a backend.

v1:

```python
from scipy import stats
from pyapprox.variables.joint import IndependentMarginalsVariable

variable = IndependentMarginalsVariable([stats.uniform(0, 1), stats.norm(0, 1)])
samples = variable.rvs(100)
```

v2:

```python
from scipy import stats
from pyapprox.util.backends.numpy import NumpyBkd
from pyapprox.probability import IndependentJoint
from pyapprox.probability.univariate import ScipyContinuousMarginal

bkd = NumpyBkd()
joint = IndependentJoint(
    [ScipyContinuousMarginal(stats.uniform(0, 1), bkd),
     ScipyContinuousMarginal(stats.norm(0, 1), bkd)],
    bkd,
)
samples = joint.rvs(100)   # shape (nvars, nsamples), as in v1
```

For common distributions, prefer the native analytic marginals — they carry
exact PDFs/CDFs/Jacobians and hyperparameters for optimization:
`GaussianMarginal(mean, stdev, bkd)`, `UniformMarginal`, `BetaMarginal`,
`GammaMarginal`, plus `ScipyDiscreteMarginal`, `CustomDiscreteMarginal`, and
`DiscreteChebyshevMarginal`. `GroupIndependentJoint` handles grouped
independence. Variable transforms live under
`pyapprox.probability.transforms` (`GaussianTransform`,
`IndependentGaussianTransform`, `NatafTransform`, `RosenblattTransform`);
the v1 `AffineTransform` class has no same-named v2 equivalent.

### 5.2 Multifidelity estimation → `pyapprox.statest`

The string-keyed factory (`get_estimator`,
`numerically_compute_estimator_variance`, `compare_estimator_variances`) is
gone. Instantiate estimator classes directly, with an explicit statistic
object that owns the backend.

v1:

```python
from pyapprox.multifidelity.factory import get_estimator

est = get_estimator("mfmc", stat, costs)
```

v2:

```python
from pyapprox.util.backends.numpy import NumpyBkd
from pyapprox.statest import MFMCEstimator, MultiOutputMean

bkd = NumpyBkd()
stat = MultiOutputMean(nqoi, bkd)     # what to estimate
est = MFMCEstimator(stat, costs)      # backend inherited from stat
```

Available estimators in `pyapprox.statest`: `MCEstimator`, `CVEstimator`,
`ACVEstimator`, `GMFEstimator`, `GISEstimator`, `GRDEstimator`,
`MFMCEstimator`, `MLMCEstimator`, `MLBLUEEstimator`, the group-ACV family
(`GroupACVEstimatorIS`, `GroupACVEstimatorNested`), and the AETC family
(`AETC`, `AETCMC`, `AETCBLUE`). Statistics: `MultiOutputMean`,
`MultiOutputVariance`, `MultiOutputMeanAndVariance`. Model-subset selection,
which v1 handled via `max_nmodels`, is now explicit via strategy classes
(`AllModelsStrategy`, `FixedSubsetStrategy`, `AllSubsetsStrategy`, ...).
Plotting helpers (`plot_allocation`, `plot_estimator_variance_reductions`)
are re-exported from `pyapprox.statest`.

### 5.3 Surrogates

The umbrella functions `approximate(...)` and `adaptive_approximate(...)`
were removed along with the string-keyed method selection. Each surrogate
type now has its own fitter/model classes:

| v1 | v2 |
|---|---|
| `surrogates.approximate` / `adaptive_approximate` | per-method fitter classes (below) |
| `surrogates.interp.adaptive_sparse_grid` | `surrogates.sparsegrids` — `SingleFidelityAdaptiveSparseGridFitter`, `MultiFidelityAdaptiveSparseGridFitter` + basis factories (`ClenshawCurtisLagrangeFactory`, `LejaLagrangeFactory`, `GaussLagrangeFactory`, `PiecewiseFactory`) |
| `surrogates.polychaos.gpc.PolynomialChaosExpansion` | `surrogates.affine.expansions.pce` |
| `surrogates.gaussianprocess.gaussian_process.GaussianProcess` | `surrogates.gaussianprocess` — `ExactGaussianProcess(kernel, nvars, bkd)`, `TorchExactGaussianProcess`, `VariationalGaussianProcess`, `DeepGaussianProcess` |
| `surrogates.autogp` (torch GPs) | merged into `surrogates.gaussianprocess` (use `TorchBkd` / torch variants) |
| `pde.karhunen_loeve_expansion` | `surrogates.kle` |
| `surrogates.orthopoly` | folded into `surrogates.affine` and `surrogates.quadrature` |
| `surrogates.function_train` | `surrogates.functiontrain` |

GP adaptive sampling (`CholeskySampler`, `IVARSampler`, ...) moved to
`surrogates.gaussianprocess.adaptive` with an `AdaptiveGPBuilder`.

### 5.4 Benchmarks → `pyapprox_benchmarks`

`setup_benchmark("name")`, `list_benchmarks()`, and the `Benchmark` result
object are gone. Benchmarks are now concrete function/benchmark classes in
the separate `pyapprox-benchmarks` package, implementing the same function
protocols as the rest of v2.

v1:

```python
from pyapprox.benchmarks import setup_benchmark

benchmark = setup_benchmark("ishigami", a=7, b=0.1)
values = benchmark.fun(samples)
```

v2:

```python
from pyapprox.util.backends.numpy import NumpyBkd
from pyapprox_benchmarks import IshigamiFunction

bkd = NumpyBkd()
fun = IshigamiFunction(bkd, a=7.0, b=0.1)
values = fun(samples)          # jacobians via fun.jacobian(...) where implemented
```

Available names include `IshigamiFunction`, `RosenbrockFunction`,
`SobolGFunction`, `BraninFunction`, the Genz family (`CornerPeakFunction`,
`GaussianPeakFunction`, `OscillatoryFunction`, `ProductPeakFunction`),
`ODEQoIFunction`, and benchmark bundles such as `BraninBenchmark` and
`RosenbrockBenchmark`.

### 5.5 Model interface

`pyapprox.interface.wrappers` is class-based:
`FiniteDifferenceWrapper`. Evaluation counts and wall times are recorded
by `FunctionTimer`/`TimedFunction` in
`pyapprox.interface.functions.timing`. Functions you pass to pyapprox
algorithms should satisfy the protocols in
`pyapprox.interface.functions.protocols` (`FunctionProtocol`, etc.);
`FunctionFromCallable` adapts a plain Python callable. UM-Bridge support
sits behind the `umbridge` extra (`pyapprox.interface.umbridge`,
`UMBridgeModel`), and parallel evaluation lives in
`pyapprox.interface.parallel`.

### 5.6 Sensitivity analysis and inference

`pyapprox.analysis.sensitivity_analysis` → `pyapprox.sensitivity`
(variance-based methods under `sensitivity.variance_based`). Active
subspaces also live under `sensitivity`. `pyapprox.bayes` →
`pyapprox.inverse`. The v1 `analysis.convergence_studies` and
`analysis.parameter_sweeps` helpers were not carried over.

## 6. Removed with no direct replacement

- `pyapprox.cython.*` — all compiled kernels (superseded by optional Numba).
- Top-level convenience namespace (`pya.<symbol>`).
- `setup_benchmark` / `list_benchmarks` string registry.
- `get_estimator` and companion factory functions.
- `approximate` / `adaptive_approximate` umbrella functions.
- scikit-learn-based code paths.
- `analysis.convergence_studies`, `analysis.parameter_sweeps`.

## 7. Common errors and fixes

| Symptom | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: pyapprox.variables` (or `.multifidelity`, `.bayes`, `.analysis`) | subpackage renamed | see §4 map |
| `ModuleNotFoundError: pyapprox.benchmarks` | moved to separate package | `pip install pyapprox-benchmarks`; import `pyapprox_benchmarks` |
| `ImportError: cannot import name 'IndependentMarginalsVariable'` | class removed | `IndependentJoint` + marginal classes (§5.1) |
| `ImportError: cannot import name 'get_estimator'` | factory removed | instantiate estimator classes (§5.2) |
| `AttributeError: module 'pyapprox' has no attribute ...` | root namespace emptied | import from the specific subpackage |
| `TypeError: ... missing 1 required positional argument: 'bkd'` | new backend system | create `NumpyBkd()` and pass it (§3) |
| pip resolves pyapprox 1.0.3 instead of 2.0 | running Python < 3.11 | upgrade to Python 3.11+ |
| NumPy 1.x conflicts in existing env | v2 requires numpy>=2.0 | build a fresh environment |

## 8. Suggested migration procedure

1. Create a fresh environment with Python >= 3.11; `pip install pyapprox
   pyapprox-benchmarks`.
2. Fix imports mechanically using the §4 table (`variables` → `probability`,
   `multifidelity` → `statest`, `bayes` → `inverse`, `analysis` →
   `sensitivity`, `benchmarks` → `pyapprox_benchmarks`).
3. Introduce a backend: `bkd = NumpyBkd()` at the top of each script; thread
   it into constructors as errors surface.
4. Replace factory calls with class instantiation (§5.1, §5.2, §5.4).
5. Consult the [tutorial gallery](https://sandialabs.github.io/pyapprox/)
   for a working v2 example of each method — every major workflow has one.
6. If something you used has no v2 equivalent, pin `pyapprox==1.0.3` for
   that code path and open an issue at
   https://github.com/sandialabs/pyapprox/issues so it can be prioritized.
