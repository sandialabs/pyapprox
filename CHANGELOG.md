# Changelog

All notable changes to PyApprox are documented in this file. The format is
based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this
project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
Entries are added under **Unreleased** as user-facing changes merge; the
section is renamed to the version number when a release is tagged (see
[docs/RELEASING.md](docs/RELEASING.md)).

## [Unreleased]

### Added

- 3D linear elasticity and hyperelasticity in the Galerkin module
  (hex/tet, degree 1–2): Newton solves via a new analytical neo-Hookean
  tangent, 3D manufactured solutions, component-wise Dirichlet BCs
  (symmetry/roller conditions), and 3D Neumann/Robin traction support.
- Dimension-generic elasticity post-processing (`strain_from_displacement`,
  `stress_from_strain`, `von_mises_stress`, `integrate`) with an explicit
  plane-stress/plane-strain/3d assumption, replacing the 2D-only `*_2d`
  functions.
- `LatinHypercubeSampler` in `pyapprox.util.sampling`: Latin hypercube
  designs via `scipy.stats.qmc` with optional scrambling, strength-2
  (orthogonal-array) designs, discrepancy optimization, and inverse-CDF
  transformation. Designs are not extensible, so `sample()` is
  single-shot per design (call `reset()` for a new design).
- `TruncatedPivotedQRFactorizer` in `pyapprox.util.linalg`: matrix-free
  truncated column-pivoted QR (Householder with Businger–Golub pivoting)
  that matches scipy's `geqp3` pivot order but runs only `npivots`
  reflector steps; NumPy and Torch backends plus an optional Numba fast
  path.
- `RidgeCVFitter` in `pyapprox.surrogates.affine.expansions.fitters`:
  ridge regression selecting the regularization strength via fast
  leave-one-out / leave-many-out cross-validation (hat-matrix formula,
  no per-fold refitting).
- `Backend.cond`: backend-generic matrix condition number (NumPy and
  Torch), with `ord` typed like `norm`.
- `PCAFunctionEncoder.fit_from_data` accepts `center=False` for
  uncentered POD.
- `RandomizedSVD` accepts an optional `seed` drawn from a local
  `RandomState` (reproducible without perturbing the global NumPy RNG).
- `AdvectionDiffusionReaction` (Galerkin) exposes `stiffness_forms()`,
  `forcing_form()`, `reaction_form()`, and `reaction_jacobian_form()` so
  consumers (e.g. hyper-reduction) can assemble the weak form on
  element-restricted bases.
- `PeriodicStructuredMesh1D` in `pyapprox.pde.galerkin.mesh`: 1D Galerkin
  mesh with endpoint-identified (periodic) topology; requires the `fem`
  extra.
- Operator-inference PDE problems in `pyapprox_benchmarks.problems.pde`:
  periodic viscous Burgers and homogeneous Chafee–Infante.
- `GalerkinModel` accepts `method="implicit_midpoint"` for transient
  solves.
- `pyapprox-benchmarks` now ships a `py.typed` marker so downstream mypy
  checks its types instead of resolving imports as `Any`.
- Parameterized PDE forward models with adjoint gradients and
  Hessian-vector products across both the Galerkin and collocation
  solvers: physics expose typed coefficient-sensitivity assemblies,
  parameterizations share one chain-rule engine (composites reject
  overlapping coefficients), and steady/transient models select their
  derivative capability at construction. Third-party solver contract in
  `docs/conventions/pde_solver_extension.md`.
- 3D collocation linear elasticity at full parity with 2D, including
  Lamé-field sensitivities, boundary tractions, and domain-decomposition
  interface fluxes.

### Changed

- Derivative access goes through a `Derivatives` bundle: functions
  declare jacobian/hessian/hvp capabilities via `derivatives()` instead
  of duck-typed methods. The legacy capability-tier protocols, the
  parameterized adapter package, and the `as_derivatives` shim were
  removed; `DerivativeChecker` requires the bundle.
- Randomized SVD classes renamed for accuracy (no aliases kept):
  `SinglePassRandomizedSVD` → `TwoPassRandomizedSVD` and
  `DoublePassRandomizedSVD` → `SymmetricRandomizedSVD`.
- Adaptive sparse grid `result()` now defaults to
  `include_candidates=True`, so all evaluated candidate subspaces are
  included in the surrogate; it raises `RuntimeError` if a candidate
  lacks values. Sparse grid basis factories are now picklable (usable
  with joblib).
- `adjust_sign_svd` now requires the `bkd` argument (keyword-only); the
  implicit NumPy fallback was removed.
- `Backend.any_bool` / `Backend.all_bool` no longer accept a `keepdims`
  argument (it could not affect the scalar result and returned the wrong
  type on the NumPy backend when set).

### Fixed

- Surrogates and OED likelihood objects (Lagrange/tensor-product/sparse-grid
  interpolants, `MultiIndexBasis`, Gaussian OED likelihoods) could not be
  pickled, blocking multiprocess use
  ([#31](https://github.com/sandialabs/pyapprox/issues/31)).
- Nonlinear reaction Jacobian sign in the Galerkin
  advection–diffusion–reaction physics: `spatial_jacobian` was
  inconsistent with finite differences of `spatial_residual`, so Newton
  converged only linearly for nonlinear reactions.
- `ConstantSparseMassMatrix.as_matrix()` densified the sparse mass
  matrix, making Newton solves dense (~250× slower at 16k DOFs); it now
  stays sparse.
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
