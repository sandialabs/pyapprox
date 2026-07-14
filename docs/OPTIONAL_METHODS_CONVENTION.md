# Optional Derivative Capabilities: the Derivatives Bundle

This document is the reference for how optional derivative capability is
expressed in PyApprox. **Absence of a capability is `None`, never a
missing attribute.** Runtime method injection
(`self.jacobian = self._jacobian`), `hasattr`/`getattr` capability
sniffing, external monkey-patching (`loss.jacobian = fn`), and
capability-tier protocol taxonomies (`FunctionWithJacobianProtocol`, …)
are banned and enforced by repository grep gates.

## The bundle

`pyapprox.interface.functions.derivatives.Derivatives` is a frozen
dataclass with Optional callable fields:

| Field            | Signature (single-sample optimizer forms)              |
|------------------|---------------------------------------------------------|
| `jacobian`       | `(nvars, 1) -> (nqoi, nvars)`                            |
| `jacobian_batch` | `(nvars, n) -> (n, nqoi, nvars)`                         |
| `jvp`            | `(sample, vec) -> (nqoi, 1)`                             |
| `hvp`            | `(sample, vec) -> (nvars, 1)` — nqoi == 1 Hessian        |
| `whvp`           | `(sample, vec, weights (nqoi, 1)) -> (nvars, 1)`         |
| `hessian`        | `(nvars, 1) -> (nvars, nvars)` — nqoi == 1 only          |
| `hessian_batch`  | `(nvars, n) -> (n, nvars, nvars)` — nqoi == 1 only       |
| `hvp_batch`      | `(samples, vecs) -> (n, nvars)` — **SCALAR-IMPLICIT**: unlike `jacobian_batch` there is no nqoi axis in the output (a recurring reshaping trap) |
| `whvp_batch`     | `(samples, vecs, weights (nqoi, 1)) -> (n, nvars)`       |
| `inexact`        | `InexactSuite(value, jacobian=None)`, both `(sample, tol)` — ROL tolerance-aware evaluation |

In any bundle, `jacobian` means d/d(THIS function's input), unqualified:
a GP loss's bundle differentiates w.r.t. hyperparameters because
hyperparameters ARE that function's input; a surrogate's bundle
differentiates w.r.t. x.

## Named constructors

Prefer named constructors; the raw constructor is sanctioned only for
unusual combinations (hvp AND whvp together; a materialized single-sample
`hessian`; batch-only fields; a bundle carrying an `InexactSuite`
alongside other fields):

- `Derivatives.none()`
- `Derivatives.first_order(jacobian, *, jvp=None, jacobian_batch=None)`
- `Derivatives.second_order(jacobian, hvp, *, jvp=None,
  jacobian_batch=None, hvp_batch=None, hessian_batch=None)` — nqoi == 1
- `Derivatives.second_order_weighted(jacobian, whvp, *, jvp=None,
  jacobian_batch=None, whvp_batch=None)` — vector-valued adjoint Hessian

## Protocols

Canonical locations (single import location; no re-exports):

- `pyapprox.interface.functions.protocols.function.FunctionProtocol` —
  value-only base shape (`bkd`/`nvars`/`nqoi`/`__call__(samples)`).
- `pyapprox.interface.functions.protocols.objective.ObjectiveProtocol` —
  FunctionProtocol plus `derivatives()`. Objectives must have
  `nqoi() == 1` (enforced by validation, not the type).
- `pyapprox.interface.functions.protocols.constraint.NonlinearConstraintProtocol`
  — ObjectiveProtocol shape plus `lb()`/`ub()`; legitimately
  vector-valued. Lives in the interface layer because interface-level
  wrappers (e.g. `WithAutogradJacobianConstraint`) consume it; the
  optimization layer imports downward, never the reverse.

`__call__`'s parameter is named `samples` everywhere (a single sample is
a batch with nsamples == 1); mypy protocol conformance checks positional
parameter names.

## Producers

- **Two-case naming rule**: unconditional capability keeps a public
  `def jacobian(...)` (genuine class API; the bundle references it).
  Conditional capability uses a private implementation method
  (`_jacobian_analytical`) opening with a
  `raise RuntimeError("jacobian is unavailable; check derivatives() "
  "before calling")` guard — the bundle is the only public surface, so
  humans cannot call a method that only sometimes works.
- **Bundle creation time (two-case rule)**: build and store
  `self._derivs` in `__init__` when the capability decision and any
  closed-over context exist at construction. Bound methods late-bind to
  instance state, so data arriving later (set by `fit()`) is fine — only
  pre-fit *invocation* raises. Build inside `derivatives()` (or rebuild
  at the decision point, e.g. `set_estimator()` with an overridable
  `_build_derivatives()`) when the capability decision itself needs
  post-construction state. Storing bound-method bundles in `__init__` is
  deepcopy/pickle-safe (the cycle re-points to the clone; verified by the
  design spike).
- **No fallback in producers**: absent capability stays `None` to the
  consumer. scipy applies its own finite differences when `jac=None`;
  ROL falls back to its internal secant. No finite-difference fallback
  exists anywhere in the bundle or in producers.
- **Autograd is a composition source, never a producer fallback**:
  `WithAutogradJacobian(inner, bkd)` /
  `WithAutogradJacobianConstraint(inner, bkd)` at the orchestrator, or
  `autograd_derivatives(fun, bkd)` explicitly in `__init__` for classes
  that are inherently autograd-differentiable (e.g. ELBO objectives) —
  always gated on `isinstance(bkd, AutodiffBackend)`. `AutodiffBackend`
  is a runtime-checkable presence protocol: a backend method named
  `jacobian` opts the backend into autograd dispatch (NumpyBkd is False,
  TorchBkd is True — both directions are locked by tests).

## Consumers

- Capture narrowed fields once at `bind()` into always-present private
  Optional attributes (`self._jac: Optional[JacobianFn[Array]] =
  d.jacobian`) — value varies, attribute shape never does. This is the
  SANCTIONED capture idiom, not to be confused with banned public-method
  injection.
- Never read the raw hvp/whvp pair — call
  `d.resolved_hvp(obj.nqoi(), obj.bkd())` / `d.resolved_whvp(obj.nqoi())`
  with the OWNING object's nqoi (a constraint's own nqoi, never the
  objective's), so a vector constraint is never accidentally
  scalar-lifted. The resolvers lift/synthesize exactly (w = [1]) when
  nqoi == 1 and deliberately do NOT synthesize an hvp from a
  materialized `hessian` (matrix-free must not silently become
  O(nvars^2) memory).
- Never `hasattr`/`getattr` for capability; never
  `except NotImplementedError` around the `Function` sugar ABC
  (consumers decide off the bundle; both are grep-gated).

## Wrappers

Generic wrappers (timing, work tracking, parallelization, restriction,
reparameterization) mirror the inner bundle: each populated field is
re-exposed wrapped in a module-level frozen-dataclass callable (picklable
for multiprocessing — never a closure), absent fields stay absent.

## Accessor currying rule

Non-optimizer derivative families carry the "wrt what / of what" in a
differently-named accessor that closes over extra context, so the bundle
field arities above always hold: kernels expose `param_derivatives()`
and `input_derivatives(X2)`; a distribution exposing d(logpdf)/dx uses a
`logpdf_derivatives()` accessor. Parameter-family methods
(`jacobian_wrt_params`, shape `(nqoi, nactive_params)`) remain public
named methods.

## Lifecycle

Bundles are frozen; capability is fixed per `bind()`/`minimize()` run.
When active hyperparameters change via
`hyp_list().set_active_values(...)` or a producer is refit, rebuild the
producer's bundle and rebind — never mutate a bundle. Reconfiguring
capability for one run uses `OverrideDerivatives(inner, bundle)` or
`bundle.with_(...)` (explicit construction, fully type-checked).

## The six forbidden constructs (mypy --strict error codes)

1. Mutating a bundle field (`d.jacobian = fn`) — frozen: `[misc]`
2. Calling an Optional field without narrowing (`d.jacobian(x)`) —
   `[misc]` (union-attr style); narrow to a local first.
3. A field callable with the wrong signature — `[arg-type]`
4. Passing a non-conforming object where `ObjectiveProtocol` is
   expected — `[arg-type]` (watch the `samples` parameter name).
5. `Derivatives.second_order(jac, None)` — `[arg-type]`; use
   `first_order` when hvp is unavailable.
6. `Derivatives.second_order_weighted(jac)` missing whvp — `[call-arg]`

Zero `cast` / `type: ignore` in bundle-related code. At genuinely
untyped third-party boundaries (e.g. torch's `Module.__call__` returning
`Any`), narrow with a runtime `isinstance` check that raises — never an
unchecked annotation.

## Validation and checking

- `validate_objective` requires `ObjectiveProtocol` conformance, a
  well-formed bundle from `derivatives()`, and nqoi == 1.
- `with_shape_validation(d, nvars, nqoi)` wraps populated fields with
  boundary shape checks (opt-in, picklable wrappers).
- `DerivativeChecker` finite-difference-validates every populated bundle
  field. (Migration note: until the last legacy modules are retired,
  the checker's `derivative_checks/_legacy_harvest.py` fallback may
  harvest public methods of not-yet-migrated objects, warning loudly;
  that file is a temporary, grep-gate-exempted exception and will be
  deleted.)
