# Package layering: where code goes, and how it moves

## The rule

Every top-level package under `packages/pyapprox/src/pyapprox/` is a
sibling; levels come from what may import what, not from nesting. A package
imports only packages at its own level or below.

| Level | Packages |
|---|---|
| L0 | `util` (backends, hyperparameters, base protocols, expressions) |
| L1 | `linalg`, `multiindex`, `interface` |
| L2 | `polynomials`, `quadrature`, `kernels`, `implicitfunction`, `optimization`, `risk`, `probability` |
| L3 | `ode`, `statest` |
| L4 | `pde`, `randomfields`, `surrogates`, `inverse`, `generative` |
| L5 | `expdesign`, `sensitivity` |

Some of these packages do not exist yet; their code still lives where the
target map below says it comes from. Two independence rules sit on top of
the levels:

- `pde` never imports `randomfields`, `surrogates` or `probability`, and
  none of them imports `pde`. A discretization needs polynomial, kernel and
  random-field building blocks, which live below it; a statistics object
  reaches a PDE model only by satisfying a protocol the model owns.
- `polynomials` never imports `probability`. A marginal supplies its
  recurrence coefficients through a protocol `polynomials` owns; numerical
  construction (Lanczos or Stieltjes on a discretization of the measure) is
  the fallback for a marginal that does not.

`.importlinter` enforces what is checkable today, and `lint-imports` runs
in CI. A contract's `ignore_imports` lists edges that exist and are wrong:
the list only shrinks. An ignore that matches no import is an error, so a
change that removes an edge also deletes its ignore line.

## Where code goes: two deletion tests

**Delete the consumer.** For code X used by package P, imagine deleting P.
If X becomes dead, X belongs inside P. If other packages would still need
it, it belongs in a lower package they can all import. This decides where
shared code and protocols live, and it is why most protocols live with
their consumer rather than in a shared module.

**Delete everything above.** A package is at the right level if deleting
every package above it leaves it importable, with its own tests passing.
An import that fails this test reaches upward: move the imported code down,
or invert the dependency with a protocol the lower package owns.

## Consumer-owned protocols

Each consumer declares the minimal `@runtime_checkable` protocol it needs,
in its own module. Implementations satisfy it structurally and import
nothing from the consumer. Two consumers needing the same shape each keep
their own protocol; a shared implementation satisfies both.

Import-linter cannot enforce ownership; it only forbids the listed imports.
Drift between two consumers' protocols, or between a protocol and a
provider, is caught by a mypy-checked assignment in a test module,

```python
_: ConsumerProtocol[NumpyBkd] = Provider(bkd)
```

not by `isinstance`, which checks method names only.

## No registries

Callers construct objects and inject them; nothing in the library maps a
name to a class. A string arriving from outside Python (a CLI argument, a
config file) maps to a class only in a dict literal in the one module that
parses it. CLAUDE.md's Extensibility Requirements carry the full rule.

## Moving a directory

A move is a pure rename: one commit holding `git mv` of the whole
directory (and its test directories, so `ck tests` keeps mapping) plus one
entry in `MOVED` in `pyapprox/_moved.py`. No file's imports change, except
the moved package's imports of itself.

**When.** Nothing moves up front, and a directory may stay where it is
indefinitely. A directory in the target map moves when work is about to:

- add code to it: move first, then write the code in its final place;
- need an import a contract forbids that the move would make legal (for
  example `pde` needing `surrogates/affine/univariate`, which is legal once
  it is `polynomials`);
- remove an ignore entry that the move clears.

An area whose owner has content changes in flight is not moved until that
work lands.

**Recipe**, all in the one commit:

1. `git mv` the directory and its tests.
2. Add `"old.prefix": "new.prefix"` to `MOVED`, and rewrite the moved
   source's absolute imports of itself to the new prefix. Left on the old
   path, such an import loads the old parent package first; if that
   parent imports the moved code back through the alias, the new path
   hits a circular import when imported first. Import-linter also does
   not see old-path imports, so the new package's contracts would miss
   its own edges.
3. `python -m pyapprox._moved_stubs` to write the type stubs at the old
   path.
4. Add a `moved:<old prefix>=<count>` line to `ratchet_baselines.txt`, with
   the count `python -m pyapprox._moved_ratchet` reports.
5. Give a new top-level package a light `__init__` (see CLAUDE.md, Imports).
6. Delete numba caches (`*.nbi`, `*.nbc`) for the moved modules' `cache=True`
   functions, since stale ones fail with "underlying object has vanished",
   and any tracked autosaves in the directory.
7. Add the new package's contracts, with its remaining upward edges as
   ignores, and delete ignores that named the old paths: import-linter
   drops an import of an old path, so those ignores no longer match.
   Check with `lint-imports --no-cache`; its cache keeps the old edges
   for files that did not change.

Then run the full suite: old imports elsewhere now go through the alias.
Every worktree rebases onto the move on a scratch branch and runs its
affected tests before it lands. Moves that come due together share one
rebase window.

**Old paths after a move.** `import old.mod` returns the module object at
`new.mod` itself, so class identity, `isinstance`, `mock.patch` on old
target strings and objects pickled under the old path all keep working. It
warns once per prefix. The generated stubs make an old path type-check as
the new one. `ck consumers` reports old-path users as
`imports (via old path …)`.

**Migrating imports.** New code uses new paths: a pre-commit check refuses
an old path on an added line, and CI fails if a prefix's count rises above
its baseline. Files nobody is touching are not migrated. When a commit
lowers a count, it lowers the baseline too. When a count reaches zero and
the downstream repos have migrated, the prefix's `MOVED` entry and stubs
are deleted. A line that must name an old path (release notes, the
old-to-new table) carries `moved: allow`.

## Target map

**Directory moves**, whole directories with contents unchanged:

| New location | From |
|---|---|
| `kernels/` | `surrogates/kernels/` |
| `quadrature/` | `surrogates/quadrature/`; `util/sampling/` becomes `quadrature/sampling/` |
| `polynomials/` | `surrogates/affine/univariate/` becomes `polynomials/univariate/`; `affine/leja/` becomes `polynomials/leja/`; `affine/induced.py` becomes `polynomials/induced.py` |
| `multiindex/` | `surrogates/affine/indices/` |
| `linalg/` | `util/linalg/`; `util/rootfinding/` becomes `linalg/rootfinding/` |
| `randomfields/` | `surrogates/kle/` |

A moved directory whose contents are still wrong for its target (for
example `polynomials` importing `surrogates.affine.protocols` and
`probability`) keeps those upward edges as ignores until its content moves.

**Content moves**, made later inside the new packages:

| Destination | From |
|---|---|
| `polynomials` | basis parts of `surrogates/affine/{basis, protocols}`, `surrogates/sparsegrids/basis`, Gauss rules out of `quadrature` |
| `quadrature` | `affine/basis/quadrature_rules` (merged); quadrature and sampler protocols from `util/protocols`; generic samplers from `expdesign/quadrature`; Smolyak combination from `sparsegrids` |
| `multiindex` | admissibility, margins, Smolyak coefficients and the generic refinement loop from `sparsegrids/{refine, refinement, priority, subspace}` |
| `kernels` | `util/linalg/kernel_operators.py` |
| `linalg` | greedy point selection from `polynomials/leja`; `optimization/linear_solvers`, merged into the Krylov solvers; `pde/sparse_utils.apply_dirichlet_rows`; the row-replacement parts of `implicitfunction/pinned_rows`; the snapshot and streaming-SVD parts of `randomfields` |
| `optimization` | subset search from `expdesign/solver` |
| `probability` | `util/transforms/affine.py` |
| `surrogates/reduction` | encoder implementations, with one encoder protocol per consumer |

A symbol that moves between modules keeps its old module reachable for one
release through a module `__getattr__` that returns it from the new module
with a `DeprecationWarning`.
