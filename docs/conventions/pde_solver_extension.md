# Extending PyApprox with a new PDE solver module

Third-party (or in-repo) PDE discretizations plug into the shared
parameterization, time-integration, and forward-model machinery
through four seams. A module that honors all four gets adjoint
gradients, tangent-linear sensitivities, and second-order-adjoint
Hessian-vector products without writing any derivative propagation
code. The two in-repo solver families (`pde/galerkin`,
`pde/collocation`) are built exclusively on these seams and serve as
reference implementations.

## Seam 1 — physics assemblies and BC classification

The physics owns the residual and its typed derivative assemblies:

- `residual(state, time)` / `jacobian(state, time)`.
- Per parameterizable coefficient field `g`: a full-matrix assembly
  `residual_<coef>_jacobian(state)` returning
  `S(u) = ∂R/∂g` (shape `(nstates, nfield)`), and — when the term is
  linear in the state — the mixed assembly
  `residual_<coef>_state_jacobian(delta, state)` returning
  `A(δ) = ∂/∂u [S(u) δ]`. State-nonlinear terms supply genuine mixed
  contraction callables instead (see the quasilinear galerkin physics)
  or stage at first order (see the hyperelastic facade).
- If the coefficient enters boundary-condition rows (flux/traction
  BCs), a boundary assembly with the consumer-pinned signature
  `(state, time, bc_indices, normals) -> (n_bc, nfield)` where
  `bc_indices` are the BC's replaced state rows.
- `boundary_conditions()`, `apply_boundary_conditions(residual,
  jacobian, state, time)` (the public 4-arg surface; `time` is
  required — a defaulted t=0 silently corrupts transient BC values),
  `apply_bc_to_mass(mass)`, and `bc_dof_classification()` returning
  essential and row-replaced DOF lists with the invariant
  `essential ⊆ row_replaced`. Sparse and dense assemblies are both
  supported: the engine keeps dense backend arrays in backend space
  and crosses to numpy only for scipy-sparse operands.

## Seam 2 — facade-over-engine parameterization

Parameterizations are construction wiring only: a typed facade maps
domain kwargs (`diffusion_map=...`) onto
`_FieldParameterizationTerm` slots, wrapping bound physics methods in
the picklable signature adapters (`StateJacobianAdapter`,
`FieldStateJacobianAdapter`, `ConstantJacobianAdapter`, setter
adapters). The engine owns every chain rule and HVP; the capability
tier (first vs second order) is decided once at construction from the
field map's usable HVP — never by `hasattr`. Facades declare
`owned_coefficients()` so `CompositeParameterization` can reject
overlapping parts; parameter-coupled coefficients (e.g. Lame fields
tied through one Young's modulus) belong inside ONE term's field map,
not in a composite. Exemplars:
`CollocationAdvectionDiffusionParameterization` (multi-coefficient,
bc-flux slot), `CollocationElasticityParameterization` (stacked
field), `CollocationHyperelasticityParameterization` (staged first
order via `_WithoutHVP` and raising slots).

## Seam 3 — the time-integration residual surface

Implement (or wrap a stepper into)
`AdjointEnabledTimeSteppingResidualProtocol`
(`pyapprox.ode.protocols.time_stepping`), including
`bc_dof_classification`-derived row/column corrections if the solver
replaces residual rows (the collocation `BCEnforcing*` wrappers are
the reference). Everything above the protocol is shared:
`TimeIntegrator`, the tangent-linear sweep
(`solve_final_forward_sensitivity`), and the second-order adjoint
(`TimeAdjointOperatorWithHVP`) come for free.

## Seam 4 — FunctionProtocol forward models

Forward models satisfy `FunctionProtocol`: `__call__` maps parameter
samples to QoIs; `derivatives()` returns a `Derivatives` bundle whose
tier is fixed at construction from the parameterization bundle, the
physics capability, the BC rows, and the functional's protocol tier
(downgrades from missing physics capability warn; functional-driven
downgrades are silent). Build the pipeline once and rebind parameters
per sample. Reference: `pde/models/collocation/steady.py` and
`transient.py`, `pde/models/galerkin/transient.py`.

## Testing contract

- Validate every jacobian/HVP with `DerivativeChecker` (the
  FD-eps sweep). A marginal ratio with a V-shaped error curve is
  noise — assert the measured V-bottom plus a calibrated ratio with a
  comment; a plateau across eps decades is a real bug.
- Add the exact (FD-noise-immune) identities where they exist:
  bilinearity `A(δ) w == S(w) δ`, HVP symmetry
  `<H v, u> == <H u, v>` at 1e-12, and the mixed-tensor cross
  identity.
- Steady problems: the 14-check `ImplicitFunctionDerivativeChecker`
  suite through the adjoint operators at the widest available tier.
- Transformed domains: at least one FD run of the parameter
  derivatives on a non-identity mesh (curved-boundary normals and
  metric factors do not inherit correctness from identity-mesh
  tests).
- Pickle round-trip of every facade (multiprocessing ensembles ship
  forward models to workers): no lambdas or closures in stored
  callables — module-level adapter classes and bound methods only.
- No `hasattr` capability dispatch anywhere: capability is protocol
  isinstance checks and `None`-checked bundle fields, decided at
  construction.
