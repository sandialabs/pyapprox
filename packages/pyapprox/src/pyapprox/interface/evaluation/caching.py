"""Serving an objective's quantities from a store where it can.

Distinct from resuming. A resumed run knows sample 7 is the same sample
because it regenerated the same array from the same seed, so positional
keys suffice. Caching answers a harder question -- "have I seen *this
point* before?" -- which needs identity derived from the values.

That is a modelling decision, so it is injected rather than fixed:
:class:`~pyapprox.interface.evaluation.protocols.SampleLookup` states
what a policy must do, and :class:`RoundedHashLookup` is one. A policy
that accepts any sample within a tolerance, searching what it has seen,
satisfies the same protocol.

**Composition, not reimplementation.** :class:`CachedObjective` wraps an
objective -- usually ``blocking(evaluator)`` -- rather than wrapping an
evaluator itself. ``BlockingModel`` already turns an evaluator into an
``ObjectiveProtocol`` and mirrors every capability it advertises;
rebuilding that here would mean rebuilding the bundle, and a wrapper
mirroring only the quantities its author remembered would silently drop
the rest, since an absent bundle field is indistinguishable from a model
that never had the capability::

    model = blocking(evaluator)
    cached = CachedObjective(model, store, RoundedHashLookup(bkd))

**A hit is free; a miss still waits.** Because this sits above the
blocking adapter, a cache miss blocks like any other call. An
evaluator-level cache -- one satisfying ``EvaluatorProtocol`` and
wrapping another evaluator -- would serve hits while misses were still
in flight, and would cover ``Ensemble`` too; its hard part is ``submit``
returning a ``Batch`` that reports cached samples as already succeeded.
Not built.

**Failures are not cached.** A blocking call raises rather than
returning a short array, so nothing reaches the store for a batch
containing a failure -- including the samples in it that succeeded.
Those are recomputed next time, which costs but cannot mislead: a stored
placeholder would make a failed sample read as known and never be
retried. Work cached by earlier successful calls is untouched.
"""

import hashlib
from typing import Any, Callable, Dict, Generic, Optional

import numpy as np

from pyapprox.interface.evaluation.protocols import (
    ResultStore,
    SampleLookup,
)
from pyapprox.interface.evaluation.records import (
    ComputeProvenance,
    Cost,
    Decoded,
)
from pyapprox.interface.functions.derivatives import Derivatives
from pyapprox.interface.functions.protocols.objective import (
    ObjectiveProtocol,
)
from pyapprox.util.backends.protocols import Array, Backend


class _Quantity:
    """How one cacheable quantity is carried and cut up.

    Named in data so one implementation serves every quantity: adding
    one is a row here rather than a copy of the caching body. ``axis``
    is the sample axis of the assembled array and is deliberately not
    uniform -- values are sample-last, jacobians and hessians
    sample-first -- which is exactly the detail worth stating once.
    """

    def __init__(self, field: str, axis: int) -> None:
        self.field = field
        self.axis = axis


#: Every bundle field this module can cache, and how its answer is
#: carried. Two groups, distinguished by what the answer depends on:
#:
#: - **Sample-keyed** (values, jacobians, hessians). The answer is a
#:   function of the sample alone, so a sample key suffices.
#: - **Direction-keyed** (jvp, hvp, whvp, and their batch forms). The
#:   answer depends on the direction as well, so the key covers the pair
#:   -- see :class:`_CachedDirectional`.
#:
#: ``axis`` is the sample axis of the assembled array, and it is
#: deliberately not uniform: values and jvps are sample-last while
#: jacobians, hessians and hvps are sample-first. That asymmetry is a
#: recurring source of transposed results, which is why it is stated
#: here once rather than re-derived per quantity.
QUANTITIES: Dict[str, _Quantity] = {
    "values": _Quantity("values", 1),
    "jacobians": _Quantity("jacobians", 0),
    "hessians": _Quantity("hessians", 0),
    "jvps": _Quantity("jvps", 1),
    "hvps": _Quantity("hvps", 0),
}


class RoundedHashLookup(Generic[Array]):
    """Identity by rounding, then hashing.

    Parameters
    ----------
    bkd : Backend[Array]
        Converts a column for hashing.
    decimals : int, optional
        The tolerance. Samples agreeing to this many decimal places are
        the same sample. Default 12.

    Notes
    -----
    Rounding before hashing is what makes this usable rather than
    correct-but-useless: hashing raw bytes would treat ``0.1 + 0.2`` and
    ``0.3`` as different points, and the miss would be invisible -- the
    cache simply never hits and the model runs again.

    Canonicalising to float64 matters as much: the same value in float32
    hashes differently otherwise, so a model run at reduced precision
    would never hit its own cache.

    Choosing ``decimals`` is the modelling decision this class cannot
    make. Too coarse and distinct samples collide, returning one
    sample's value for another -- silently. Too fine and nothing ever
    hits. The default is conservative; a caller who knows its model's
    scale should say so.

    Converting to numpy here is serialization rather than computation,
    the same boundary the stores use.
    """

    def __init__(self, bkd: Backend[Array], decimals: int = 12) -> None:
        self._bkd = bkd
        self._decimals = decimals

    def decimals(self) -> int:
        """The rounding tolerance in decimal places."""
        return self._decimals

    def key(self, *columns: Array) -> str:
        """The key these columns hash to under this policy.

        The columns are hashed in order with their shapes, so a
        ``(sample, vec)`` pair cannot collide with a single array
        holding the same numbers end to end.
        """
        if not columns:
            raise ValueError("at least one column is required")
        digest = hashlib.sha256()
        for column in columns:
            if column.ndim != 2 or column.shape[1] != 1:
                raise ValueError(
                    "each column must have shape (n, 1), got "
                    f"{tuple(column.shape)}"
                )
            canonical = np.round(
                self._bkd.to_numpy(column).astype(np.float64),
                self._decimals,
            )
            # +0.0 and -0.0 are equal but have different bytes.
            canonical = canonical + 0.0
            digest.update(str(canonical.shape).encode())
            digest.update(canonical.tobytes())
        return digest.hexdigest()[:32]

    def find(self, *columns: Array) -> Optional[str]:
        """The key for these columns.

        A hashing policy needs no memory: the key is a function of the
        arrays, so every combination is findable and the store decides
        whether anything is held under it. A searching policy would
        return None for a combination it has not seen.
        """
        return self.key(*columns)

    def remember(self, *columns: Array) -> str:
        """Record these columns as seen. Idempotent, and stateless here."""
        return self.key(*columns)


class CachedObjective(Generic[Array]):
    """An objective that computes only what its store does not hold.

    Parameters
    ----------
    model : ObjectiveProtocol[Array]
        The objective to serve from, usually ``blocking(evaluator)``.
    store : ResultStore[Array]
        Where hits come from and misses are recorded. Several objectives
        may share one, provided their lookups agree on identity.
    lookup : SampleLookup[Array]
        What counts as the same sample. Required rather than defaulted:
        a tolerance chosen by the library would be a modelling decision
        made on the caller's behalf, and a wrong one is silent.

    Notes
    -----
    Keys are qualified by quantity, so a stored value never answers a
    request for a jacobian at the same point. Without that, a
    sample-only key would report a hit and hand back a record whose
    jacobians field is None -- in whichever order the caller happened to
    ask, since an optimizer may want a gradient at a point before
    anything wants its value.
    """

    def __init__(
        self,
        model: ObjectiveProtocol[Array],
        store: ResultStore[Array],
        lookup: SampleLookup[Array],
    ) -> None:
        if not isinstance(model, ObjectiveProtocol):
            raise TypeError(
                "model must satisfy ObjectiveProtocol (evaluation plus "
                f"derivatives()), got {type(model).__name__}"
            )
        if not isinstance(store, ResultStore):
            raise TypeError(
                f"store must satisfy ResultStore, got {type(store).__name__}"
            )
        if not isinstance(lookup, SampleLookup):
            raise TypeError(
                "lookup must satisfy SampleLookup (find and remember), "
                f"got {type(lookup).__name__}"
            )
        self._model = model
        self._store = store
        self._lookup = lookup
        self._bkd = model.bkd()
        source = model.derivatives()
        # Construction-time branching over whatever the wrapped model
        # has: each populated batch field becomes a cached one, and an
        # absent capability stays absent.
        #
        # The single-sample forms are forwarded unchanged. They are
        # served by the wrapped model from the same solve as their batch
        # form, so caching them here would key the same answer twice
        # under two spellings.
        self._derivs: Derivatives[Array] = Derivatives(
            jacobian=source.jacobian,
            jacobian_batch=self._wrap(source.jacobian_batch, "jacobians"),
            hessian=source.hessian,
            hessian_batch=self._wrap(source.hessian_batch, "hessians"),
            jvp=source.jvp,
            hvp=source.hvp,
            hvp_batch=self._wrap_directional(
                source.hvp_batch, "hvps", weighted=False
            ),
            whvp=source.whvp,
            whvp_batch=self._wrap_directional(
                source.whvp_batch, "whvps", weighted=True
            ),
            inexact=source.inexact,
        )

    def _wrap(
        self,
        field: Optional[Callable[[Array], Array]],
        quantity: str,
    ) -> Optional[Callable[[Array], Array]]:
        """Cache a sample-keyed ``field`` if the model provides it."""
        if field is None:
            return None
        return _CachedQuantity(self, field, quantity)

    def _wrap_directional(
        self,
        field: Optional[Callable[..., Array]],
        quantity: str,
        weighted: bool,
    ) -> Optional[Any]:
        """Cache a direction-keyed ``field`` if the model provides it."""
        if field is None:
            return None
        return _CachedDirectional(self, field, quantity, weighted)

    def bkd(self) -> Backend[Array]:
        """Return the array backend."""
        return self._bkd

    def nvars(self) -> int:
        """Number of input variables."""
        return self._model.nvars()

    def nqoi(self) -> int:
        """Number of quantities of interest."""
        return self._model.nqoi()

    def model(self) -> ObjectiveProtocol[Array]:
        """The wrapped objective, for a caller that wants to bypass the cache."""
        return self._model

    def store(self) -> ResultStore[Array]:
        """The store being consulted."""
        return self._store

    def lookup(self) -> SampleLookup[Array]:
        """The identity policy in use."""
        return self._lookup

    def derivatives(self) -> Derivatives[Array]:
        """The wrapped model's bundle, with each batch field cached."""
        return self._derivs

    def key_for(self, column: Array, quantity: str) -> Optional[str]:
        """The key one sample's ``quantity`` is stored under, if known."""
        found = self._lookup.find(column)
        return None if found is None else f"{quantity}:{found}"

    def __call__(self, samples: Array) -> Array:
        """Evaluate ``samples``, computing only what is not held."""
        return self.cached(samples, self._model, "values")

    def cached(
        self,
        samples: Array,
        compute: Callable[[Array], Array],
        quantity: str,
    ) -> Array:
        """Serve ``quantity``, calling ``compute`` only for the misses.

        One body for every quantity: what differs -- which field carries
        the answer, and which axis indexes the sample -- is looked up in
        :data:`QUANTITIES` rather than written out again for each.
        """
        if samples.ndim != 2:
            raise ValueError(
                "samples must be 2D (nvars, nsamples), got shape "
                f"{tuple(samples.shape)}"
            )
        spec = QUANTITIES[quantity]
        nsamples = int(samples.shape[1])
        columns = [samples[:, i : i + 1] for i in range(nsamples)]

        # Resolve each column to the key it is held under, if any.
        keys: list[Optional[str]] = [
            self.key_for(column, quantity) for column in columns
        ]

        # A sample repeated within one batch is computed once, so the
        # first column claiming a key owns it.
        claimed: set[str] = set()
        todo: list[int] = []
        for i, key in enumerate(keys):
            if key is not None and self._store.load(key) is not None:
                continue
            if key is not None and key in claimed:
                continue
            if key is not None:
                claimed.add(key)
            todo.append(i)

        if todo:
            produced = compute(samples[:, todo])
            for j, i in enumerate(todo):
                key = f"{quantity}:{self._lookup.remember(columns[i])}"
                keys[i] = key
                self._store.save(
                    key, _one_sample(produced, j, spec, self._bkd), _no_cost()
                )

        pieces = []
        for i, key in enumerate(keys):
            if key is None:
                raise RuntimeError(
                    f"sample {i} has no key after computation; the lookup "
                    "returned None from remember()"
                )
            record = self._store.load(key)
            if record is None:
                raise RuntimeError(
                    f"sample {i} was computed but is not in the store "
                    f"under {key!r}"
                )
            pieces.append(getattr(record[0], spec.field))
        return (
            self._bkd.hstack(pieces)
            if spec.axis == 1
            else self._bkd.vstack(pieces)
        )

    def cached_directional(
        self,
        samples: Array,
        vecs: Array,
        weights: Optional[Array],
        compute: Callable[..., Array],
        quantity: str,
    ) -> Array:
        """Serve a directional quantity, keyed on the direction too.

        Column ``j`` of ``vecs`` is the direction for column ``j`` of
        ``samples``, so each pair is looked up together. Weights, when
        present, apply to the whole call rather than per sample, and are
        folded into every key: the same sample and direction under
        different weights are different answers.
        """
        if samples.shape != vecs.shape:
            raise ValueError(
                f"samples {tuple(samples.shape)} and vecs "
                f"{tuple(vecs.shape)} must have the same shape; column j "
                "of vecs is the direction for column j of samples"
            )
        spec = QUANTITIES[
            "hvps" if quantity in ("hvps", "whvps") else quantity
        ]
        nsamples = int(samples.shape[1])
        extra = () if weights is None else (weights,)

        keys: list[Optional[str]] = []
        for i in range(nsamples):
            found = self._lookup.find(
                samples[:, i : i + 1], vecs[:, i : i + 1], *extra
            )
            keys.append(None if found is None else f"{quantity}:{found}")

        claimed: set[str] = set()
        todo: list[int] = []
        for i, key in enumerate(keys):
            if key is not None and self._store.load(key) is not None:
                continue
            if key is not None and key in claimed:
                continue
            if key is not None:
                claimed.add(key)
            todo.append(i)

        if todo:
            args = (samples[:, todo], vecs[:, todo])
            produced = (
                compute(*args)
                if weights is None
                else compute(*args, weights)
            )
            for j, i in enumerate(todo):
                key = "{}:{}".format(
                    quantity,
                    self._lookup.remember(
                        samples[:, i : i + 1], vecs[:, i : i + 1], *extra
                    ),
                )
                keys[i] = key
                self._store.save(
                    key, _one_sample(produced, j, spec, self._bkd), _no_cost()
                )

        pieces = []
        for i, key in enumerate(keys):
            if key is None:
                raise RuntimeError(
                    f"sample {i} has no key after computation; the lookup "
                    "returned None from remember()"
                )
            record = self._store.load(key)
            if record is None:
                raise RuntimeError(
                    f"sample {i} was computed but is not in the store "
                    f"under {key!r}"
                )
            pieces.append(getattr(record[0], spec.field))
        return (
            self._bkd.hstack(pieces)
            if spec.axis == 1
            else self._bkd.vstack(pieces)
        )


class _CachedQuantity(Generic[Array]):
    """One sample-keyed bundle field, served through the shared cache.

    A class rather than a closure so the bundle stays picklable, which
    matters as soon as a cached model is handed to a process pool.
    """

    def __init__(
        self,
        cache: "CachedObjective[Array]",
        field: Callable[[Array], Array],
        quantity: str,
    ) -> None:
        self._cache = cache
        self._field = field
        self._quantity = quantity

    def __call__(self, samples: Array) -> Array:
        return self._cache.cached(samples, self._field, self._quantity)


class _CachedDirectional(Generic[Array]):
    """A directional field, keyed on the direction as well as the sample.

    ``hvp(x, v)`` is one answer for that *pair*: keying on ``x`` alone
    would return the product for whatever direction happened to be asked
    for first, which is wrong in a way no shape check catches. So the
    key covers the sample and the vector together, and for the weighted
    form the weights as well.

    Stacking the arrays and handing the result to the same lookup keeps
    one identity policy in play: whatever tolerance a caller chose for
    samples applies to directions too, which is usually what they meant.
    """

    def __init__(
        self,
        cache: "CachedObjective[Array]",
        field: Callable[..., Array],
        quantity: str,
        weighted: bool,
    ) -> None:
        self._cache = cache
        self._field = field
        self._quantity = quantity
        self._weighted = weighted

    def __call__(
        self, samples: Array, vecs: Array, weights: Optional[Array] = None
    ) -> Array:
        return self._cache.cached_directional(
            samples, vecs, weights, self._field, self._quantity
        )


def _one_sample(
    produced: Array, column: int, spec: _Quantity, bkd: Backend[Array]
) -> Decoded[Array]:
    """Cut one sample's piece out of a batch, as a storable record."""
    if spec.axis == 1:
        return Decoded(values=produced[:, column : column + 1], indices=[0])
    piece = produced[column : column + 1]
    # A derivative-only record carries no values, which Decoded spells
    # as an empty (nqoi, 0): the sample axis of values is the second, so
    # a task that decoded no values has nothing along it.
    empty = bkd.reshape(bkd.array([]), (int(piece.shape[1]), 0))
    # Named explicitly rather than expanded from spec.field: a **kwargs
    # dict erases which field is being set, so mypy cannot check the
    # record it builds -- and this record's field-to-shape agreement is
    # exactly what needs checking.
    if spec.field == "jacobians":
        return Decoded(values=empty, indices=[0], jacobians=piece)
    if spec.field == "hessians":
        return Decoded(values=empty, indices=[0], hessians=piece)
    if spec.field == "hvps":
        return Decoded(values=empty, indices=[0], hvps=piece)
    if spec.field == "jvps":
        return Decoded(values=empty, indices=[0], jvps=piece)
    raise ValueError(f"unknown cacheable quantity {spec.field!r}")


def _no_cost() -> Cost:
    """A cache hit is not charged again for what it did not recompute."""
    return Cost(
        wall_clock=0.0,
        compute=0.0,
        provenance=ComputeProvenance.NOT_APPLICABLE,
    )
