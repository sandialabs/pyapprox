# Optional Capabilities: Derivative Bundles

**Absence of a capability is `None`, never a missing attribute.** A class
that may or may not provide a derivative declares what it provides in a
frozen dataclass of Optional callables, returned by an accessor. These are
banned: runtime method injection (`self.jacobian = self._jacobian`),
`hasattr`/`getattr` capability sniffing, external monkey-patching
(`loss.jacobian = fn`), and capability-tier protocol taxonomies
(`FunctionWithJacobianProtocol`, `...WithHVPProtocol`, ...), where which
protocol a class satisfies says what it can do.

This document holds the rules shared by every bundle. Each bundle's own
module docstring is the reference for its fields, signatures and
invariants.

## Bundle families

| Bundle | Module | Accessor | Owner |
|---|---|---|---|
| `Derivatives` | `interface.functions.derivatives` | `derivatives()` | functions, objectives, constraints, surrogates; kernels via `param_derivatives()` / `input_derivatives(X2)` |
| `ParamDerivatives` | `pde.parameterizations.derivatives` | `param_derivatives()` | PDE parameterizations |
| `StateDerivatives` | `ode.state_derivatives` | `state_derivatives()` | spatial operators, boundary terms |
| `SteadyConstrainedDerivatives` | `pde.models.galerkin.constrained_derivatives` | built by `steady_constrained_derivatives` | Galerkin steady adapter |
| `SteadyFunctionalDerivatives` | `optimization.implicitfunction.functionals.derivatives` | `derivatives()` | steady functionals |
| `TransientFunctionalDerivatives` | `ode.functionals.protocols` | `derivatives()` | transient functionals |
| `SteadyStateEquationDerivatives` | `optimization.implicitfunction.state_equations.derivatives` | `derivatives()` | steady state equations |
| `ImplicitOperatorDerivatives` | `optimization.implicitfunction.operator.derivatives` | `derivatives()` | `AdjointOperator`, `ForwardSensitivityOperator` |

Reuse `Derivatives` when its field arities fit, binding any fixed context
in the accessor or the producer. A family whose derivatives take
arguments that vary per call (a state, a time, an adjoint vector) cannot
be curried into those arities and gets its own frozen bundle, as every
other row above does.

Required members stay methods on the protocol; only optional ones go in
the bundle. A state equation's Jacobians are required (Newton needs them)
and its second derivatives optional; a functional's Jacobians and second
derivatives are both optional, since a functional may be only evaluated.

Most families share two invariants, enforced in `__post_init__`: fields
that only make sense together come together (both Jacobians; all four
second-order contractions, because the second-order adjoint needs every
one), and fields built on first derivatives require them.

## Construction

Build bundles with the named constructors (`none()`, `first_order(...)`,
`second_order(...)`, and family-specific ones such as
`Derivatives.second_order_weighted`). The raw constructor is for unusual
combinations a named constructor does not cover (for `Derivatives`: hvp
and whvp together, a materialized single-sample `hessian`, batch-only
fields, an `InexactSuite` alongside other fields).

## Producers

- **Two-case naming rule.** Unconditional capability keeps a public method
  (`def jacobian(...)`, genuine class API) that the bundle references.
  Conditional capability uses a private method (`_jacobian_analytical`)
  opening with a guard such as `raise RuntimeError("jacobian is
  unavailable; check derivatives() before calling")`; the bundle is then
  the only public surface, so nobody calls a method that only sometimes
  works.
- **When to build the bundle.** Build and store `self._derivs` in
  `__init__` when the capability decision and the context it closes over
  exist at construction. Bound methods late-bind to instance state, so
  data arriving later (set by `fit()`) is fine; only invoking before it
  raises. Build in `derivatives()`, or rebuild at the decision point
  (e.g. `set_estimator()` with an overridable `_build_derivatives()`),
  when the decision needs post-construction state. Bundles of bound
  methods stored in `__init__` deepcopy and pickle correctly.
- **No fallback.** Absent capability stays `None`. Consumers choose their
  own fallback: scipy finite-differences when `jac=None`, ROL uses its
  secant. No producer substitutes finite differences.
- **Autograd is composed, never a fallback.**
  `WithAutogradJacobian(inner, bkd)` /
  `WithAutogradJacobianConstraint(inner, bkd)` at the orchestrator, or
  `autograd_derivatives(fun, bkd)` in `__init__` for classes that are
  inherently autograd-differentiable (e.g. ELBO objectives), and only
  when `isinstance(bkd, AutodiffBackend)`. `AutodiffBackend` is a
  runtime-checkable presence protocol: a backend method named `jacobian`
  opts the backend in (NumpyBkd does not, TorchBkd does).
- **Context is bound at construction, never passed per call.** A bundle's
  field arities hold for every producer of that family; extra context
  goes in the accessor (`input_derivatives(X2)`) or the producer's
  constructor. The "wrt what" lives in the accessor name: in any
  `Derivatives` bundle `jacobian` means d/d(this function's input), so a
  GP loss differentiates w.r.t. hyperparameters and a surrogate w.r.t. x,
  and a distribution's d(logpdf)/dx comes from `logpdf_derivatives()`.
  Parameter-family methods (`jacobian_wrt_params`) remain public named
  methods.
- **Capability may depend on injected configuration.** A transient
  functional that integrates over time has per-step HVPs only under a
  quadrature whose samples are the stored columns, so its bundle depends
  on the quadrature injected by `set_time_quadrature`, and its
  `derivatives()` raises before one is set. The component owning the
  scheme injects the quadrature first, then reads the bundle once.

## Consumers

- **Read once, then capture.** Read the bundle once at `bind()` or
  construction, narrow each field to a local, and store it in an
  always-present private Optional attribute
  (`self._jac: Optional[JacobianFn[Array]] = d.jacobian`): the value
  varies, the attribute's existence never does. This is the sanctioned
  capture idiom, distinct from banned public-method injection.
- **Derive capability from bundles, not types.** A consumer's own
  capability is the combination of its inputs' bundles: `AdjointOperator`
  offers an HVP only when both the state equation's and the functional's
  bundles carry all four second derivatives, and a forward model offers
  `jacobian` only when the parameterization and the functional both
  provide theirs. Steady and transient objects satisfy overlapping
  protocols, so consumers that need one kind check `time_dependence()` at
  runtime as well as the protocol.
- **Use the resolvers.** Never read `Derivatives`' raw hvp/whvp pair; call
  `d.resolved_hvp(obj.nqoi(), obj.bkd())` / `d.resolved_whvp(obj.nqoi())`
  with the owning object's nqoi (a constraint's own, never the
  objective's), so a vector constraint is never scalar-lifted by
  accident. See the `Derivatives` docstring for what the resolvers
  synthesize and why they never slice a materialized Hessian.
- Never `hasattr`/`getattr` for capability, and never
  `except NotImplementedError` around the `Function` sugar ABC; decide
  from the bundle.

## Wrappers

A wrapper mirrors its inner object's bundle: each populated field is
re-exposed, adjusted or wrapped, and absent fields stay absent. Wrapped
callables are module-level frozen-dataclass callables or bound methods,
never closures, so the wrapper pickles for multiprocessing.

When to read the inner bundle follows from when the inner capability is
fixed. If it is fixed at construction, capture it in `__init__`
(`TikhonovAugmentedSteadyFunctional`). If it can change afterwards, as a
transient functional's can when its quadrature is injected, read the
inner bundle when called and forward the injection
(`TikhonovAugmentedFunctional`).

## Protocols for optimizer-facing objects

Canonical locations (no re-exports):

- `interface.functions.protocols.function.FunctionProtocol`: value-only
  base shape (`bkd`/`nvars`/`nqoi`/`__call__(samples)`).
- `interface.functions.protocols.objective.ObjectiveProtocol`:
  FunctionProtocol plus `derivatives()`; nqoi == 1, enforced by
  validation rather than the type.
- `interface.functions.protocols.constraint.NonlinearConstraintProtocol`:
  the ObjectiveProtocol shape plus `lb()`/`ub()`, legitimately
  vector-valued. It lives in the interface layer because interface-level
  wrappers (`WithAutogradJacobianConstraint`) consume it.

`__call__`'s parameter is named `samples` everywhere (a single sample is a
batch of one); mypy protocol conformance checks positional parameter
names.

## Lifecycle

Bundles are frozen: capability is fixed per `bind()`/`minimize()` run.
When active hyperparameters change or a producer is refit, rebuild the
producer's bundle and rebind; never mutate one. To reconfigure capability
for one run, use `OverrideDerivatives(inner, bundle)` or
`bundle.with_(...)`.

## Typing

Zero `cast` / `type: ignore` in bundle-related code. The two constructs
that most often go wrong:

- mutating a bundle field (`d.jacobian = fn`): frozen, mypy `[misc]`;
- calling an Optional field without narrowing (`d.jacobian(x)`): narrow
  to a local first.

At genuinely untyped third-party boundaries (e.g. torch's
`Module.__call__` returning `Any`), narrow with a runtime `isinstance`
check that raises, never an unchecked annotation.

## Validation and checking

- `validate_objective` requires `ObjectiveProtocol` conformance, a
  well-formed bundle from `derivatives()`, and nqoi == 1.
- `with_shape_validation(d, nvars, nqoi)` wraps populated fields with
  boundary shape checks (opt-in, picklable wrappers).
- `DerivativeChecker` finite-difference-validates every populated field
  of a `Derivatives` bundle. It requires `derivatives()`
  (`interface.functions.derivative_checks.resolve`): objects without the
  accessor are rejected with a `TypeError`, and public derivative methods
  are never harvested.
