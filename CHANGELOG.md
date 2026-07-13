# Changelog

All notable changes to PyApprox are documented in this file. The format is
based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this
project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
Entries are added under **Unreleased** as user-facing changes merge; the
section is renamed to the version number when a release is tagged (see
[docs/RELEASING.md](docs/RELEASING.md)).

## [Unreleased]

### Added

- `PeriodicStructuredMesh1D` in `pyapprox.pde.galerkin.mesh`: 1D Galerkin
  mesh with endpoint-identified (periodic) topology; requires the `fem`
  extra.
- Operator-inference PDE problems in `pyapprox_benchmarks.problems.pde`:
  periodic viscous Burgers and homogeneous Chafee–Infante.
- `GalerkinModel` accepts `method="implicit_midpoint"` for transient
  solves.

### Changed

- `adjust_sign_svd` now requires the `bkd` argument (keyword-only); the
  implicit NumPy fallback was removed.
- `Backend.any_bool` / `Backend.all_bool` no longer accept a `keepdims`
  argument (it could not affect the scalar result and returned the wrong
  type on the NumPy backend when set).

### Fixed

- `TorchBkd.prod` with `axis=None, keepdims=True` raised `RuntimeError`;
  it now matches NumPy semantics.

## [2.0.0] - 2026-06-07

Ground-up redesign of the library. See the
[v1 → v2 migration guide](docs/migrations/v1-to-v2.md); the summary below
is the headline view.

### Changed (breaking)

- Minimum Python raised from 3.6 to 3.11; NumPy 2.0+ now required.
- Repository restructured as a monorepo publishing three packages:
  `pyapprox` (core), `pyapprox-benchmarks`, `pyapprox-tutorials`.
- All array handling now flows through explicit backend objects
  (`NumpyBkd`, `TorchBkd` in `pyapprox.util.backends`); most classes are
  generic over the array type and take a `bkd` argument.
- Subpackages renamed: `variables` → `probability`, `multifidelity` →
  `statest`, `bayes` → `inverse`, `analysis` → `sensitivity`.
- Functional factories replaced by direct class instantiation:
  `setup_benchmark` → benchmark classes in `pyapprox_benchmarks`;
  `get_estimator` → estimator classes in `pyapprox.statest`;
  `approximate`/`adaptive_approximate` → per-method fitter classes.
- `IndependentMarginalsVariable` replaced by `IndependentJoint` plus
  explicit marginal classes implementing `MarginalProtocol`.
- `GaussianProcess` replaced by `ExactGaussianProcess`,
  `TorchExactGaussianProcess`, `VariationalGaussianProcess`, and
  `DeepGaussianProcess`.
- Build system moved from setuptools+Cython to hatchling+hatch-vcs;
  versions are derived from git tags.
- Docs/tutorials moved from Sphinx/sphinx-gallery to a Quarto site built
  from `pyapprox-tutorials`.

### Added

- Backend abstraction with interchangeable NumPy and Torch (GPU-capable)
  backends.
- New subpackages: `ode` (steppers/operators), `generative` (including
  flow matching), `risk`, `statest.groupacv` (group ACV estimators),
  `statest.aetc` (AETC family), `surrogates.kle`, `surrogates.supn`,
  `surrogates.mfnets`, GP adaptive-sampling builders.
- Optional extras: `fem`, `umbridge`, `numba`, `parallel`, `cvxpy`, `rol`,
  `runtime-extras`, `dev`.
- Typed public API (`Typing :: Typed`), mypy-strict codebase, import-linter
  architecture checks.

### Removed

- Compiled Cython extensions (replaced by optional Numba kernels).
- `scikit-learn` dependency and code paths.
- Top-level `pyapprox` convenience namespace (root exposes `__version__`
  only).
- `setup_benchmark`, `list_benchmarks`, `get_estimator`,
  `compare_estimator_variances`, `approximate`, `adaptive_approximate`.
- `analysis.convergence_studies`, `analysis.parameter_sweeps`.

## [1.0.3] - and earlier

Final 1.x releases (1.0, 1.0.2, 1.0.3) predate this changelog. See the
[legacy-master branch](https://github.com/sandialabs/pyapprox/tree/legacy-master)
for the 1.x codebase.

[Unreleased]: https://github.com/sandialabs/pyapprox/compare/v2.0.0...HEAD
[2.0.0]: https://github.com/sandialabs/pyapprox/compare/legacy-master...v2.0.0
