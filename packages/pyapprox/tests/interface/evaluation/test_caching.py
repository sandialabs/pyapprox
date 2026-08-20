"""Reusing a stored value when the same sample comes round again.

Distinct from resuming. A resumed run knows sample 7 is the same sample
because it regenerated the same array from the same seed, so positional
keys suffice. Caching answers a harder question -- "have I seen *this
point* before?" -- which needs identity derived from the values, and
that is a modelling decision rather than a framework one:

- Bit-exact comparison is not equality. ``0.1 + 0.2`` and
  ``0.30000000000000004`` are the same point and different bytes, and
  the same values in float32 and float64 differ again.
- How close is close enough is problem-specific. Two samples a
  nanometre apart are the same evaluation for a coarse mesh and
  different ones for a fine one.

The library supplies the machinery -- ``CachedObjective`` and the
``SampleLookup`` seam -- and the caller supplies the policy saying what
"same" means. These tests exercise both, including a lookup written here
rather than in the library, which is the evidence that the seam is
usable by someone outside it.
"""

import pytest
from pyapprox.interface.evaluation.adapters import (
    EvaluationFailure,
    blocking,
)
from pyapprox.interface.evaluation.caching import (
    CachedObjective,
    RoundedHashLookup,
)
from pyapprox.interface.evaluation.callable_marshaller import (
    CallableMarshaller,
)
from pyapprox.interface.evaluation.evaluator import Evaluator
from pyapprox.interface.evaluation.inline_dispatcher import InlineDispatcher
from pyapprox.interface.evaluation.protocols import SampleLookup
from pyapprox.interface.evaluation.records import Request
from pyapprox.interface.evaluation.stores import (
    InMemoryResultStore,
    NpzResultStore,
    StoreWriter,
)
from pyapprox.interface.functions.derivatives import Derivatives


class _CountingModel:
    """f(x) = sum(x^2), counting the samples it actually evaluates."""

    def __init__(self, bkd):
        self._bkd = bkd
        self.nevaluated = 0

    def __call__(self, samples):
        self.nevaluated += int(samples.shape[1])
        return self._bkd.sum(samples * samples, axis=0)[None, :]



def _caching_model(bkd, store=None, lookup=None, derivatives=None):
    """A cached objective over a counting model, for the tests below."""
    model = _CountingModel(bkd)
    marshaller = CallableMarshaller(
        model,
        bkd,
        nvars=2,
        nqoi=1,
        samples_per_task=1,
        derivatives=derivatives,
    )
    evaluator = Evaluator(marshaller, InlineDispatcher(marshaller.run))
    cached = CachedObjective(
        blocking(evaluator),
        store or InMemoryResultStore(),
        lookup or RoundedHashLookup(bkd),
    )
    return cached, model


class TestRoundedHashLookup:
    """The shipped identity policy: round, then hash."""

    def test_satisfies_the_protocol(self, bkd):
        assert isinstance(RoundedHashLookup(bkd), SampleLookup)

    def test_identical_samples_agree(self, bkd):
        lookup = RoundedHashLookup(bkd)
        a = bkd.array([[1.0], [2.0]])
        b = bkd.array([[1.0], [2.0]])
        assert lookup.key(a) == lookup.key(b)

    def test_different_samples_differ(self, bkd):
        lookup = RoundedHashLookup(bkd)
        a = bkd.array([[1.0], [2.0]])
        b = bkd.array([[1.0], [2.5]])
        assert lookup.key(a) != lookup.key(b)

    def test_representation_noise_is_absorbed(self, bkd):
        """0.1 + 0.2 != 0.3 in bytes, but it is the same sample.

        Hashing raw bytes would miss this, and the miss is invisible --
        the cache simply never hits and the model runs again.
        """
        lookup = RoundedHashLookup(bkd)
        noisy = bkd.array([[0.1 + 0.2], [1.0]])
        exact = bkd.array([[0.3], [1.0]])
        assert float(noisy[0, 0]) != float(exact[0, 0])
        assert lookup.key(noisy) == lookup.key(exact)

    def test_signed_zero_agrees(self, bkd):
        lookup = RoundedHashLookup(bkd)
        assert lookup.key(bkd.array([[0.0], [1.0]])) == lookup.key(
            bkd.array([[-0.0], [1.0]])
        )

    def test_tolerance_is_the_callers_choice(self, bkd):
        """Two points are the same or not depending on decimals."""
        a = bkd.array([[1.0], [1.0]])
        b = bkd.array([[1.0 + 1e-9], [1.0]])
        assert RoundedHashLookup(bkd, decimals=6).key(a) == (
            RoundedHashLookup(bkd, decimals=6).key(b)
        )
        assert RoundedHashLookup(bkd, decimals=12).key(a) != (
            RoundedHashLookup(bkd, decimals=12).key(b)
        )

    def test_find_and_remember_agree(self, bkd):
        """A hashing policy needs no memory, so both give one key."""
        lookup = RoundedHashLookup(bkd)
        sample = bkd.array([[1.0], [2.0]])
        assert lookup.find(sample) == lookup.remember(sample)

    def test_remember_is_idempotent(self, bkd):
        """A repeat must not mint a second key for the same sample."""
        lookup = RoundedHashLookup(bkd)
        sample = bkd.array([[1.0], [2.0]])
        assert lookup.remember(sample) == lookup.remember(sample)

    def test_several_arrays_key_jointly(self, bkd):
        """A directional answer depends on the sample and the direction."""
        lookup = RoundedHashLookup(bkd)
        x = bkd.array([[1.0], [2.0]])
        u = bkd.array([[1.0], [0.0]])
        v = bkd.array([[0.0], [1.0]])
        assert lookup.key(x, u) != lookup.key(x, v)
        assert lookup.key(x, u) == lookup.key(x, u)

    def test_a_pair_does_not_collide_with_one_long_array(self, bkd):
        """Shapes are hashed too, so concatenation is not equality.

        Without that, ``(x, v)`` and a single column holding the same
        numbers end to end would key alike, and a jvp could be served
        from a value cache.
        """
        lookup = RoundedHashLookup(bkd)
        pair = lookup.key(
            bkd.array([[1.0], [2.0]]), bkd.array([[3.0], [4.0]])
        )
        stacked = lookup.key(
            bkd.array([[1.0], [2.0], [3.0], [4.0]])
        )
        assert pair != stacked

    def test_rejects_a_batch(self, bkd):
        with pytest.raises(ValueError, match=r"shape \(n, 1\)"):
            RoundedHashLookup(bkd).key(
                bkd.array([[1.0, 2.0], [3.0, 4.0]])
            )

    def test_rejects_no_columns(self, bkd):
        with pytest.raises(ValueError, match="at least one column"):
            RoundedHashLookup(bkd).key()


class TestCaching:
    """A repeated sample costs nothing the second time."""

    def test_repeat_call_does_not_re_evaluate(self, bkd):
        cached, model = _caching_model(bkd)
        samples = bkd.array([[1.0, 2.0], [3.0, 4.0]])

        first = cached(samples)
        assert model.nevaluated == 2

        second = cached(samples)
        assert model.nevaluated == 2, "re-evaluated a cached sample"
        bkd.assert_allclose(second, first)

    def test_values_are_correct(self, bkd):
        cached, _ = _caching_model(bkd)
        samples = bkd.array([[1.0, 2.0], [3.0, 4.0]])
        # 1+9=10, 4+16=20
        bkd.assert_allclose(cached(samples), bkd.array([[10.0, 20.0]]))

    def test_partial_overlap_evaluates_only_the_new(self, bkd):
        cached, model = _caching_model(bkd)
        cached(bkd.array([[1.0, 2.0], [3.0, 4.0]]))
        assert model.nevaluated == 2

        # One sample seen, one new.
        result = cached(bkd.array([[2.0, 5.0], [4.0, 6.0]]))
        assert model.nevaluated == 3
        # 4+16=20, 25+36=61
        bkd.assert_allclose(result, bkd.array([[20.0, 61.0]]))

    def test_column_order_is_preserved(self, bkd):
        """Cached and fresh columns must land where they were asked for.

        The reordering trap again: mixing hits and misses means the
        submitted batch is a subset in a different order, so writing
        results back positionally would transpose values between
        samples.
        """
        cached, _ = _caching_model(bkd)
        cached(bkd.array([[5.0], [6.0]]))  # seed one sample: 25+36=61

        samples = bkd.array([[1.0, 5.0, 2.0], [3.0, 6.0, 4.0]])
        result = cached(samples)
        bkd.assert_allclose(result, bkd.array([[10.0, 61.0, 20.0]]))

    def test_duplicate_within_one_batch_runs_once(self, bkd):
        """A sample repeated inside a single call is evaluated once."""
        cached, model = _caching_model(bkd)
        samples = bkd.array([[1.0, 1.0, 2.0], [3.0, 3.0, 4.0]])
        result = cached(samples)
        assert model.nevaluated == 2
        bkd.assert_allclose(result, bkd.array([[10.0, 10.0, 20.0]]))

    def test_noisy_repeat_still_hits(self, bkd):
        """The rounding tolerance earns its keep end to end."""
        cached, model = _caching_model(bkd)
        cached(bkd.array([[0.3], [1.0]]))
        assert model.nevaluated == 1
        cached(bkd.array([[0.1 + 0.2], [1.0]]))
        assert model.nevaluated == 1, "representation noise caused a miss"

    def test_cache_survives_the_process(self, bkd, tmp_path):
        """A file-backed store makes the cache outlive the run."""
        directory = str(tmp_path / "cache")
        samples = bkd.array([[1.0, 2.0], [3.0, 4.0]])

        first, first_model = _caching_model(
            bkd, NpzResultStore(directory, bkd)
        )
        first(samples)
        assert first_model.nevaluated == 2

        # A new process: new evaluator, new model, same directory.
        second, second_model = _caching_model(
            bkd, NpzResultStore(directory, bkd)
        )
        result = second(samples)
        assert second_model.nevaluated == 0, "recomputed a cached sample"
        bkd.assert_allclose(result, bkd.array([[10.0, 20.0]]))


class _CountingJacobian:
    """d/dx sum(x^2) = 2x, counting the samples it differentiates.

    Separate from the forward counter because a jacobian-only request
    never reaches the forward callable: the two quantities come from
    different entry points, which is the whole reason their caches must
    be keyed apart.
    """

    def __init__(self, bkd):
        self._bkd = bkd
        self.nevaluated = 0

    def __call__(self, samples):
        n = int(samples.shape[1])
        self.nevaluated += n
        return self._bkd.reshape(2.0 * samples.T, (n, 1, 2))


def _jacobian_batch(bkd):
    """d/dx sum(x^2) = 2x, shaped (n, nqoi, nvars)."""
    return _CountingJacobian(bkd)


def _caching_jacobian_model(bkd, store=None):
    """A cached objective serving jacobians as well as values.

    Returns the cache and both counters, since the two quantities come
    from different callables and each has its own hit rate.
    """
    jac = _CountingJacobian(bkd)
    cached, model = _caching_model(
        bkd, store, derivatives=Derivatives(jacobian_batch=jac)
    )
    return cached, model, jac


def _jacobians_of(cached, samples):
    """Ask for jacobians the way a consumer does: through the bundle."""
    field = cached.derivatives().jacobian_batch
    assert field is not None
    return field(samples)


class TestCachingDerivatives:
    """The same pattern, for a quantity that is not values."""

    def test_repeat_jacobian_is_not_recomputed(self, bkd):
        cached, _, jac = _caching_jacobian_model(bkd)
        samples = bkd.array([[1.0, 2.0], [3.0, 4.0]])

        first = _jacobians_of(cached, samples)
        assert first.shape == (2, 1, 2)
        # d/dx sum(x^2) = 2x
        bkd.assert_allclose(
            first, bkd.array([[[2.0, 6.0]], [[4.0, 8.0]]])
        )

        before = jac.nevaluated
        second = _jacobians_of(cached, samples)
        bkd.assert_allclose(second, first)
        assert jac.nevaluated == before

    def test_jacobian_order_is_preserved(self, bkd):
        """Sample-first jacobians must be reassembled on the right axis.

        A jacobian batch is (n, nqoi, nvars) while values are
        (nqoi, n), so joining these two quantities uses different
        axes -- mixing them up yields a correctly sized array of
        misplaced derivatives.
        """
        cached, _, _ = _caching_jacobian_model(bkd)
        _jacobians_of(cached, bkd.array([[2.0], [4.0]]))  # seed the middle

        samples = bkd.array([[1.0, 2.0, 5.0], [3.0, 4.0, 6.0]])
        result = _jacobians_of(cached, samples)
        assert result.shape == (3, 1, 2)
        bkd.assert_allclose(
            result,
            bkd.array([[[2.0, 6.0]], [[4.0, 8.0]], [[10.0, 12.0]]]),
        )

    def test_a_value_cache_does_not_answer_a_jacobian_request(self, bkd):
        """The trap: keying on the sample alone conflates quantities.

        Both caches hold the same point. Qualifying the key with the
        quantity is what keeps the jacobian cache from reporting a hit
        because a *value* was stored, which would hand back a record
        whose jacobians field is None.
        """
        store = InMemoryResultStore()
        cached, _, _ = _caching_jacobian_model(bkd, store)

        sample = bkd.array([[1.0], [3.0]])
        cached(sample)
        value_keys = set(store.keys())

        _jacobians_of(cached, sample)
        # The jacobian request stored its own record rather than
        # reusing the value one.
        assert len(set(store.keys())) == len(value_keys) + 1
        record, _ = store.load(cached.key_for(sample, "jacobians"))
        assert record.jacobians is not None

    def test_jacobian_first_then_values_still_runs(self, bkd):
        """The other order: a cached jacobian must not answer for values.

        Both directions matter because a caller's order is not something
        the cache can rely on -- an optimizer may ask for a gradient at
        a point before anything asks for its value.
        """
        cached, val_counter, jac_counter = _caching_jacobian_model(bkd)

        sample = bkd.array([[1.0], [3.0]])
        _jacobians_of(cached, sample)
        assert jac_counter.nevaluated == 1
        assert val_counter.nevaluated == 0

        # The value is not in the cache under its own key, so this runs.
        result = cached(sample)
        assert val_counter.nevaluated == 1
        bkd.assert_allclose(result, bkd.array([[10.0]]))

    def test_each_quantity_is_cached_independently(self, bkd):
        """Asking for one quantity leaves the others still to compute.

        Sweeping both orders on one store: whichever is asked for first
        misses and runs, the second also misses and runs, and repeating
        either afterwards hits. So a cached value never stands in for a
        derivative, or the reverse, whatever the call order.
        """
        cached, val_counter, jac_counter = _caching_jacobian_model(bkd)
        sample = bkd.array([[2.0], [5.0]])

        cached(sample)
        _jacobians_of(cached, sample)
        assert val_counter.nevaluated == 1
        assert jac_counter.nevaluated == 1

        # Both are now cached, so neither repeat costs anything.
        cached(sample)
        _jacobians_of(cached, sample)
        assert val_counter.nevaluated == 1
        assert jac_counter.nevaluated == 1

        # And both still give the right answers. 4 + 25 = 29; 2x.
        bkd.assert_allclose(cached(sample), bkd.array([[29.0]]))
        bkd.assert_allclose(
            _jacobians_of(cached, sample), bkd.array([[[4.0, 10.0]]])
        )

    def test_values_and_jacobians_can_share_one_solve(self, bkd):
        """A fused request stores both, so either can hit afterwards.

        This is why Request names quantities rather than invocations: a
        solver forming both in one pass is asked once, and the cache
        holds what it produced.
        """
        model = _CountingModel(bkd)
        marshaller = CallableMarshaller(
            model,
            bkd,
            nvars=2,
            nqoi=1,
            samples_per_task=1,
            derivatives=Derivatives(jacobian_batch=_jacobian_batch(bkd)),
        )
        evaluator = Evaluator(marshaller, InlineDispatcher(marshaller.run))
        result = evaluator.submit(
            bkd.array([[1.0], [3.0]]),
            Request(values=True, jacobians=True),
        ).collect()

        assert result.values is not None
        assert result.jacobians is not None
        bkd.assert_allclose(result.values, bkd.array([[10.0]]))
        bkd.assert_allclose(result.jacobians, bkd.array([[[2.0, 6.0]]]))
        # One solve produced both.
        assert model.nevaluated == 1


class TestCachingFailures:
    """A sample that did not produce a value must not become a hit."""

    def _flaky(self, bkd, store, fails):
        """A model that raises for the samples in ``fails``, once."""

        def model(samples):
            if any(float(v) in fails for v in samples[0]):
                raise RuntimeError("solver diverged")
            return bkd.sum(samples * samples, axis=0)[None, :]

        marshaller = CallableMarshaller(
            model, bkd, nvars=2, nqoi=1, samples_per_task=1
        )
        evaluator = Evaluator(marshaller, InlineDispatcher(marshaller.run))
        return CachedObjective(
            blocking(evaluator), store, RoundedHashLookup(bkd)
        )

    def test_a_failure_stores_nothing(self, bkd):
        """Nothing reaches the store when the computation raises.

        The blocking model raises rather than returning a short array,
        so the caching wrapper never sees a partial result to record.
        A stored placeholder would be the dangerous outcome: the failed
        sample would read as known and never be retried.
        """
        store = InMemoryResultStore()
        cached = self._flaky(bkd, store, fails={2.0})
        with pytest.raises(EvaluationFailure):
            cached(bkd.array([[1.0, 2.0], [3.0, 4.0]]))
        assert list(store.keys()) == []

    def test_a_failed_sample_is_retried(self, bkd):
        """Once the cause is fixed, the sample is computed rather than skipped."""
        store = InMemoryResultStore()
        failing = self._flaky(bkd, store, fails={2.0})
        with pytest.raises(EvaluationFailure):
            failing(bkd.array([[2.0], [4.0]]))

        # A later run, with whatever was wrong now repaired.
        working = self._flaky(bkd, store, fails=set())
        result = working(bkd.array([[2.0], [4.0]]))
        # 4 + 16 = 20
        bkd.assert_allclose(result, bkd.array([[20.0]]))

    def test_a_failure_discards_its_own_batch_only(self, bkd):
        """The loss is per call, not per store.

        Samples that succeeded *in the same call* as a failure are lost,
        because a blocking call raises rather than returning a short
        array, so the wrapper never sees them. Recovering those needs
        the evaluator directly, where failure is a return value.

        Work cached by *earlier* successful calls is untouched -- a
        later failure does not invalidate a store -- which is what makes
        retrying after a fix cheap rather than a fresh start.
        """
        store = InMemoryResultStore()

        # An earlier call that succeeded entirely.
        good = self._flaky(bkd, store, fails=set())
        good(bkd.array([[5.0], [1.0]]))
        assert len(list(store.keys())) == 1

        # A later call containing a failure loses its own successes...
        cached = self._flaky(bkd, store, fails={3.0})
        with pytest.raises(EvaluationFailure):
            cached(bkd.array([[1.0, 3.0], [1.0, 1.0]]))
        # ...but the earlier result is still there, and only it.
        assert len(list(store.keys())) == 1

        # And the surviving record still answers: a model that would
        # raise for that sample returns the cached value instead of
        # computing it. 25 + 1 = 26.
        would_fail = self._flaky(bkd, store, fails={5.0})
        bkd.assert_allclose(
            would_fail(bkd.array([[5.0], [1.0]])), bkd.array([[26.0]])
        )


class _NearestLookup:
    """A caller-written policy: accept any seen sample within a radius.

    Exists in the tests rather than the library to prove the seam is
    usable from outside it. Where ``RoundedHashLookup`` derives a key
    from the arrays and needs no memory, this one searches what it has
    seen -- which is why the protocol has ``find`` and ``remember``
    rather than a single key function.

    Linear search, because clarity matters more here than speed; a real
    one would index.
    """

    def __init__(self, bkd, radius=1e-6):
        self._bkd = bkd
        self._radius = radius
        self._seen = []  # (stacked column, key)
        self._next = 0

    def _stack(self, columns):
        return self._bkd.vstack(list(columns))

    def find(self, *columns):
        probe = self._stack(columns)
        for seen, key in self._seen:
            if seen.shape != probe.shape:
                continue
            distance = self._bkd.max(self._bkd.abs(seen - probe))
            if float(self._bkd.to_numpy(distance)) <= self._radius:
                return key
        return None

    def remember(self, *columns):
        found = self.find(*columns)
        if found is not None:
            return found
        key = f"near{self._next}"
        self._next += 1
        self._seen.append((self._stack(columns), key))
        return key


class TestCustomLookup:
    """A policy the library does not ship must work unchanged."""

    def test_satisfies_the_protocol(self, bkd):
        assert isinstance(_NearestLookup(bkd), SampleLookup)

    def test_nearby_samples_hit(self, bkd):
        """Within the radius is a hit, which hashing could not express.

        A hashing policy puts two samples either side of a rounding
        boundary in different buckets however close they are; a
        searching one does not, and both satisfy the same protocol.
        """
        cached, model = _caching_model(
            bkd, lookup=_NearestLookup(bkd, radius=1e-3)
        )
        cached(bkd.array([[1.0], [2.0]]))
        assert model.nevaluated == 1

        cached(bkd.array([[1.0 + 1e-5], [2.0]]))
        assert model.nevaluated == 1, "a sample within the radius missed"

    def test_distant_samples_miss(self, bkd):
        cached, model = _caching_model(
            bkd, lookup=_NearestLookup(bkd, radius=1e-6)
        )
        cached(bkd.array([[1.0], [2.0]]))
        cached(bkd.array([[1.5], [2.0]]))
        assert model.nevaluated == 2

    def test_values_are_still_correct(self, bkd):
        cached, _ = _caching_model(bkd, lookup=_NearestLookup(bkd))
        samples = bkd.array([[1.0, 2.0], [3.0, 4.0]])
        bkd.assert_allclose(cached(samples), bkd.array([[10.0, 20.0]]))


class TestDirectionalCaching:
    """Directional quantities are keyed on the direction as well."""

    def _hvp_model(self, bkd, store=None):
        """A model whose Hessian is 2I, so H v = 2 v."""
        counter = {"n": 0}

        def hvp_batch(samples, vecs):
            counter["n"] += int(samples.shape[1])
            return 2.0 * vecs.T

        cached, _ = _caching_model(
            bkd,
            store,
            derivatives=Derivatives(hvp_batch=hvp_batch),
        )
        return cached, counter

    def test_repeat_pair_is_not_recomputed(self, bkd):
        cached, counter = self._hvp_model(bkd)
        field = cached.derivatives().hvp_batch
        assert field is not None

        X = bkd.array([[1.0], [3.0]])
        V = bkd.array([[1.0], [0.0]])
        first = field(X, V)
        assert counter["n"] == 1
        bkd.assert_allclose(first, bkd.array([[2.0, 0.0]]))

        field(X, V)
        assert counter["n"] == 1, "recomputed a cached (sample, vec) pair"

    def test_a_new_direction_is_computed(self, bkd):
        """The trap: keying on the sample alone would return H u for v.

        Same sample, different direction, different answer. A cache that
        ignored the direction would hand back the first product, which
        is wrong in a way no shape check catches.
        """
        cached, counter = self._hvp_model(bkd)
        field = cached.derivatives().hvp_batch
        assert field is not None

        X = bkd.array([[1.0], [3.0]])
        u = bkd.array([[1.0], [0.0]])
        v = bkd.array([[0.0], [1.0]])

        bkd.assert_allclose(field(X, u), bkd.array([[2.0, 0.0]]))
        assert counter["n"] == 1

        bkd.assert_allclose(field(X, v), bkd.array([[0.0, 2.0]]))
        assert counter["n"] == 2, "a new direction was served from cache"

    def test_mixed_hits_and_misses_keep_their_order(self, bkd):
        cached, counter = self._hvp_model(bkd)
        field = cached.derivatives().hvp_batch
        assert field is not None

        # Seed the middle pair.
        field(bkd.array([[2.0], [2.0]]), bkd.array([[0.0], [1.0]]))
        assert counter["n"] == 1

        X = bkd.array([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])
        V = bkd.array([[1.0, 0.0, 1.0], [0.0, 1.0, 1.0]])
        result = field(X, V)
        assert counter["n"] == 3
        bkd.assert_allclose(
            result, bkd.array([[2.0, 0.0], [0.0, 2.0], [2.0, 2.0]])
        )

    def test_mismatched_shapes_are_rejected(self, bkd):
        cached, _ = self._hvp_model(bkd)
        field = cached.derivatives().hvp_batch
        assert field is not None
        with pytest.raises(ValueError, match="same shape"):
            field(
                bkd.array([[1.0, 2.0], [3.0, 4.0]]),
                bkd.array([[1.0], [0.0]]),
            )

    def test_absent_directional_stays_absent(self, bkd):
        """A model without hvp_batch does not gain one by being cached."""
        cached, _ = _caching_model(bkd)
        assert cached.derivatives().hvp_batch is None


class TestCachingIsNotAutomatic:
    """What the library still refuses to do."""

    def test_the_evaluator_itself_does_not_cache(self, bkd):
        """Only the wrapper skips; the evaluator computes what it is given.

        Worth pinning: if this ever starts passing with a lower count,
        an evaluator has begun deciding what a key means, which is the
        invariant the store protocol exists to protect.
        """
        model = _CountingModel(bkd)
        marshaller = CallableMarshaller(
            model, bkd, nvars=2, nqoi=1, samples_per_task=1
        )
        evaluator = Evaluator(
            marshaller,
            InlineDispatcher(marshaller.run),
            on_complete=StoreWriter(InMemoryResultStore(), "sweep"),
        )
        samples = bkd.array([[1.0], [3.0]])
        evaluator.submit(samples).collect()
        evaluator.submit(samples).collect()
        assert model.nevaluated == 2

    def test_a_coarse_key_collides(self, bkd):
        """A tolerance too coarse returns one sample's value for another.

        This is the failure mode that keeps the key function in the
        caller's hands: it is silent, and only the caller knows which
        tolerance is defensible for its model.
        """

        cached, model = _caching_model(
            bkd, lookup=RoundedHashLookup(bkd, decimals=0)
        )
        cached(bkd.array([[1.0], [3.0]]))  # 1+9 = 10
        result = cached(bkd.array([[1.4], [3.0]]))  # truly 1.96+9 = 10.96
        assert model.nevaluated == 1
        # The cached value comes back, not the right one.
        bkd.assert_allclose(result, bkd.array([[10.0]]))
        assert float(result[0, 0]) != pytest.approx(10.96)
