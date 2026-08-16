# `@property`: when it is allowed, and the existing debt

## The convention

Encapsulated instance state is private (`_name`), read with `def name()` and
written with `def set_name()`. `@property` is not used for it.

### Why, given this is not the usual Python advice

Worth stating plainly, because the rule runs against the common guidance —
prefer a plain attribute, promote to `@property` only when validation or
computation is needed. `obj.nu` is idiomatic; `obj.nu()` is not. Two reasons
outweigh that here, and neither is general:

**Uniformity across a large protocol-driven surface.** `bkd()`, `nvars()`,
`nqoi()`, `nparams()`, `nstates()`, `nterms()` are methods on hundreds of
classes and on most protocols. Once that is the pattern, a single class
spelling one of them as an attribute is worse than the whole convention
being slightly unusual, because a caller then has to remember which spelling
each class uses. The `.bkd` cases listed under the debt below are exactly
that failure, and they are why the rule stays **blanket** rather than
narrowing to "interface-like accessors only": a narrower rule needs a
judgement call at every site, and "is this computed?" changes as code
evolves, so the uniformity it protects erodes anyway.

**Properties hide cost.** Where an attribute access can assemble a dense
matrix or trigger a solve, `obj.operator` reading as free when it is not is
a real hazard. `obj.operator()` at least signals that something happens.

Two things are outside that rule, because they are immutable data rather
than encapsulated state:

- **Frozen dataclass value objects** expose their fields directly
  (`d.jacobian`, `helper.X2`).
- **`Protocol` classes declaring structural read-only attributes** use
  `@property` — `ArrayProtocol.shape`/`ndim`/`dtype`,
  `TaskProtocol.indices`.

The second is a type-system constraint rather than a preference, and the
evidence is recorded here so it is not "fixed" by someone reading the
headline rule alone. All three candidate forms were checked under
`mypy --strict`:

| Form in the protocol | Verdict |
|---|---|
| bare annotation `indices: Sequence[int]` | **error**: "Protocol member expected settable variable, got read-only attribute". A frozen dataclass cannot satisfy it |
| `@property def indices(self)` | passes; matches `ArrayProtocol` |
| accessor `def indices(self)` | passes, but forces every implementation into a private field plus a method instead of a plain attribute, and diverges from the protocol idiom used elsewhere |

So a protocol that wants to accept frozen dataclasses has one option.

## Inheriting `ABC` is not an exception

The carve-out is for a `Protocol` *declaring* a member. It does not extend to
any abstract base class.

An `@abstractmethod` `@property` with no body is a declaration and is fine.
A `@property` on an `ABC` whose body returns `self._x` is ordinary
encapsulated state that happens to live on an abstract class, and the
convention applies to it. All seven `ABC` properties found in the audit
below were the second kind — none was abstract.

## The existing debt

An AST audit of `packages/pyapprox/src/pyapprox/` (August 2026) classified
every `@property` by its enclosing class:

| Kind | Count | Status |
|---|---|---|
| `Protocol` | 12 | Allowed |
| Frozen dataclass | 1 | Allowed |
| Abstract (`@abstractmethod`) | 1 | Allowed |
| **Plain class or concrete `ABC`** | **54** | **Predates the convention; convert on touch** |

By directory:

```
10  surrogates/affine/univariate/globalpoly/
 9  pde/collocation/mesh/
 6  probability/univariate/
 5  surrogates/affine/indices/
 5  surrogates/kernels/
 4  inverse/variational/
 4  ode/mixins/
 3  pde/collocation/time_integration/
 2  pde/collocation/operators/
 2  surrogates/affine/univariate/
 1  expdesign/evidence/
 1  ode/operator/
 1  optimization/linear/
 1  pde/galerkin/time_integration/
```

53 of the 54 are read-only; only `MaxLevelCriteria.max_level`
(`surrogates/affine/indices/admissibility.py`) has a setter, so it is the
one case of genuinely writable state exposed as an attribute.

### Convert these first when the file is touched

Some are merely stylistic. These are actively confusing, because the same
name is a *method* elsewhere, so a caller cannot tell which calling
convention they are getting:

- `CostFunction.bkd`, `RefinementCriteria.bkd`
  (`surrogates/affine/indices/refinement.py`)
- `BSpline1D.bkd`, `HierarchicalBSpline1D.bkd`
  (`surrogates/affine/univariate/bspline.py`)
- `ScipyContinuousMarginal.name`, `ScipyDiscreteMarginal.name`
  (`probability/univariate/`)

`bkd()` in particular is a method on `Backend`, `MarshallerProtocol`,
`EvaluatorProtocol` and most model classes.

### One deliberate case worth keeping

`SparseJacobian.shape` (`pde/collocation/operators/jacobian_types.py`)
mirrors `ArrayProtocol.shape`, so a `SparseJacobian` reads like an array at
call sites that also handle dense arrays. That is a reason rather than an
oversight — but it is a choice, and if it is kept it should say so in the
class docstring.

## How the debt gets paid

Convert these when you touch the file, alongside the usual type and lint
cleanup. They are not worth a dedicated sweep: each conversion changes a
public spelling (`obj.nu` becomes `obj.nu()`), so it touches callers and
tests, and doing 54 at once produces a large diff with no behavioral
content.

If a conversion turns out to be disproportionate to the change that brought
you there, raise it rather than skipping it silently.
