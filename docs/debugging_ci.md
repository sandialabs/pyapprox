# Debugging a failure that only happens on CI

Some tests pass in every environment you can build and fail on CI anyway.
The usual causes are the runner's BLAS — which kernel it selects depends on
the instruction sets the CPU offers, and that changes how a reduction
reassociates — and the library versions a fresh resolve happens to pick.
Either can move a finite-difference check sitting near its round-off floor,
or change the path an iterative solver takes.

This document is for the point where you have stopped being able to
reproduce the failure and need CI itself to answer a question. The audience
is anyone working on this repository.

## Run one test instead of the suite

The Tests workflow takes a `tests` input, passed through to pytest, so a
dispatch can name exactly what to run:

```bash
gh workflow run Tests --repo sandialabs/pyapprox --ref <branch> \
  -f tier=fast \
  -f tests="packages/pyapprox/tests/path/to/test_x.py::TestY::test_z"
```

It accepts anything pytest does — a path, a node id, or a `-k` expression:

```bash
  -f tests="-k test_allocation_succeeds_without_analytical_derivatives"
```

**Set `tier=fast`.** The tier default is `slow`, which skips the fast jobs
entirely, so a targeted test in them would not run at all.

Leaving `tests` empty runs everything, and empty is what every push uses,
so the normal run is unaffected by this input existing.

A targeted run finishes in about a minute against roughly ten for the
suite. That difference decides whether you check a hypothesis three times
or guess once.

## Put the diagnostic in the exception

A failed CI job shows you a traceback and little else. Printing to stdout
from inside a passing-then-failing path is unreliable: output is
interleaved across `-n 4` workers, and a test that fails on an assertion
may never reach the print.

So carry what you need in the error itself. The difference this makes is
concrete: an optimizer whose failure says only `Optimization failed` cannot
tell you whether it took one step or a thousand, while

```
Optimization failed: Positive directional derivative for linesearch
(status 8, nit 1228). The best feasible allocation found costs 165;
the feasible reference costs 400; the optimizer stopped at an
allocation costing 161.701.
```

identifies the failure mode, proves the solver ran rather than breaking
down immediately, and shows the result was nearly good enough — all from
one run, with no second experiment needed. Report the quantity that
distinguishes your competing explanations, and prefer absolute numbers to
adjectives.

## Carrying instrumentation to CI, on a `debug/*` branch

A targeted run only helps if the code it runs contains the diagnostic, so
the instrumentation has to reach the remote. Put it on a branch named
`debug/<question>`.

**The name matters.** No workflow lists `debug/**`, so pushing such a
branch triggers nothing, while `workflow_dispatch` still works on any ref.
A `dev/**` name would instead fire tests, lint, architecture and security
on the first push — a full suite nobody asked for, in order to debug one
test. The tradeoff is that nothing runs on a `debug/*` branch unless you
dispatch it, so it is never evidence that anything else still passes;
that is the right bargain for a branch you are about to delete.

Delete it once the question is answered. These branches are not meant to
survive a rebase, and they fork from an integration branch that keeps
moving.

## Concurrency, and what cancels what

The Tests workflow groups concurrency on the ref and the event name, with
`cancel-in-progress`:

| Second run | Same ref as | Outcome |
|---|---|---|
| push | a running push | supersedes it |
| dispatch | a running push | independent |
| dispatch | a running dispatch | supersedes it |

Push superseding push is deliberate: results for a commit that has already
been overtaken are not worth the runners. The event name is in the group
because without it a dispatch joined the push's group and cancelled it,
which is how debugging one test discarded a full suite ten minutes in.

Dispatch still supersedes dispatch on the same ref. If you want two
experiments in flight at once, give each its own `debug/*` branch.
