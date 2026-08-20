"""A marshalled model, called like any other, and checked like any other.

Two things are proven here that the rest of the suite cannot.

**Correctness, not just plumbing.** Every other derivative test asserts
that what a stub returned came back in the right shape and order --
which passes even when the stub's derivative is wrong, because the same
author wrote both sides. Running ``DerivativeChecker`` against the
adapter compares the marshalled jacobian to finite differences of the
marshalled values, so a wrong derivative, a wrong axis or a mispaired
column shows up as a divergence rather than needing a hand-written
expectation.

**Composability.** The checker takes an ordinary function shape and
knows nothing about dispatchers. That it works at all is the evidence
that a marshalled model reaches the blocking half of the library.
"""

import pytest
from pyapprox.interface.evaluation.adapters import (
    BlockingModel,
    EvaluationFailure,
    blocking,
)
from pyapprox.interface.evaluation.callable_marshaller import (
    CallableMarshaller,
)
from pyapprox.interface.evaluation.evaluator import Evaluator
from pyapprox.interface.evaluation.inline_dispatcher import InlineDispatcher
from pyapprox.interface.functions.derivative_checks.derivative_checker import (
    DerivativeChecker,
)
from pyapprox.interface.functions.derivatives import Derivatives
from pyapprox.interface.functions.protocols.function import FunctionProtocol
from pyapprox.interface.functions.protocols.objective import (
    ObjectiveProtocol,
)


def _quadratic(bkd):
    """f(x) = sum(x^2), whose jacobian is 2x -- easy to check by hand."""

    def model(samples):
        return bkd.sum(samples * samples, axis=0)[None, :]

    return model


def _jacobian_batch(bkd):
    def jac(samples):
        n = samples.shape[1]
        return bkd.reshape(2.0 * samples.T, (n, 1, 2))

    return jac


def _model(bkd, fn=None, derivatives=None, **kwargs):
    marshaller = CallableMarshaller(
        _quadratic(bkd) if fn is None else fn,
        bkd,
        nvars=2,
        nqoi=1,
        derivatives=derivatives,
        **kwargs,
    )
    return BlockingModel(
        Evaluator(marshaller, InlineDispatcher(marshaller.run))
    )


class TestConformance:
    def test_satisfies_function_protocol(self, bkd):
        assert isinstance(_model(bkd), FunctionProtocol)

    def test_satisfies_objective_protocol(self, bkd):
        """Required: the wrappers it composes with check for this."""
        assert isinstance(_model(bkd), ObjectiveProtocol)

    def test_dimensions_pass_through(self, bkd):
        model = _model(bkd)
        assert model.nvars() == 2
        assert model.nqoi() == 1

    def test_rejects_a_non_evaluator(self):
        with pytest.raises(TypeError, match="EvaluatorProtocol"):
            BlockingModel("not an evaluator")

    def test_factory_builds_the_same_thing(self, bkd):
        marshaller = CallableMarshaller(
            _quadratic(bkd), bkd, nvars=2, nqoi=1
        )
        evaluator = Evaluator(marshaller, InlineDispatcher(marshaller.run))
        assert isinstance(blocking(evaluator), BlockingModel)


class TestValues:
    def test_call_returns_values(self, bkd):
        X = bkd.array([[1.0, 2.0], [3.0, 4.0]])
        bkd.assert_allclose(_model(bkd)(X), bkd.array([[10.0, 20.0]]))

    def test_agrees_with_the_bare_function(self, bkd):
        """Wrapping must not change the answer."""
        fn = _quadratic(bkd)
        X = bkd.array([[0.5, -1.5, 2.0], [1.0, 0.0, -3.0]])
        bkd.assert_allclose(_model(bkd)(X), fn(X))

    def test_empty_batch(self, bkd):
        assert _model(bkd)(bkd.zeros((2, 0))).shape == (1, 0)


class TestFailureRaises:
    """A blocking return has nowhere to report a partial result."""

    def test_any_failure_raises(self, bkd):
        def explodes(samples):
            raise RuntimeError("diverged")

        with pytest.raises(EvaluationFailure, match="did not return"):
            _model(bkd, fn=explodes)(bkd.ones((2, 3)))

    def test_error_names_the_failed_columns(self, bkd):
        """"Some samples failed" is not actionable; naming them is."""

        def picky(samples):
            if bkd.to_float(bkd.max(samples)) > 1.5:
                raise RuntimeError("diverged")
            return bkd.sum(samples * samples, axis=0)[None, :]

        X = bkd.array([[0.0, 1.0, 2.0], [0.0, 1.0, 2.0]])
        with pytest.raises(EvaluationFailure, match="2: failed"):
            _model(bkd, fn=picky, samples_per_task=1)(X)

    def test_partial_success_still_raises(self, bkd):
        """Silently returning a narrower array would be worse."""

        def picky(samples):
            if bkd.to_float(bkd.max(samples)) > 1.5:
                raise RuntimeError("diverged")
            return bkd.sum(samples * samples, axis=0)[None, :]

        X = bkd.array([[0.0, 1.0, 2.0], [0.0, 1.0, 2.0]])
        with pytest.raises(EvaluationFailure, match="1 of 3"):
            _model(bkd, fn=picky, samples_per_task=1)(X)


class TestDerivativeBundle:
    def test_absent_capability_stays_absent(self, bkd):
        """No probing: the bundle mirrors what the evaluator declares."""
        assert _model(bkd).derivatives().jacobian_batch is None

    def test_declared_capability_is_mirrored(self, bkd):
        model = _model(
            bkd, derivatives=Derivatives(jacobian_batch=_jacobian_batch(bkd))
        )
        assert model.derivatives().jacobian_batch is not None

    def test_jacobian_is_callable_and_correct(self, bkd):
        model = _model(
            bkd, derivatives=Derivatives(jacobian_batch=_jacobian_batch(bkd))
        )
        X = bkd.array([[1.0, 2.0], [3.0, 4.0]])
        jac = model.derivatives().jacobian_batch(X)
        bkd.assert_allclose(jac, bkd.reshape(2.0 * X.T, (2, 1, 2)))

    def test_bundle_holds_no_closures(self, bkd):
        """A process pool has to send this model to a worker.

        Pickling the bundle end to end also drags in the wrapped model,
        which in these tests is a locally-defined function and so is
        unpicklable for reasons that have nothing to do with the
        adapter. What matters here is that the adapter contributes
        classes rather than closures, since a closure would make the
        bundle unpicklable no matter what the model was.
        """
        model = _model(
            bkd, derivatives=Derivatives(jacobian_batch=_jacobian_batch(bkd))
        )
        jacobian = model.derivatives().jacobian
        assert jacobian is not None
        assert jacobian.__class__.__qualname__ == "_BlockingJacobian"


class TestDerivativesAreCorrect:
    """Finite differences, not a hand-written expectation.

    The check the rest of the suite cannot make: it compares the
    marshalled jacobian against differences of the marshalled *values*,
    so both sides come from the framework and a disagreement means the
    framework is wrong somewhere.
    """

    def test_jacobian_agrees_with_finite_differences(self, bkd):
        model = _model(
            bkd, derivatives=Derivatives(jacobian_batch=_jacobian_batch(bkd))
        )
        checker = DerivativeChecker(model)
        errors = checker.check_derivatives(
            bkd.array([[0.5], [-1.5]]), relative=True
        )
        assert bkd.to_float(checker.error_ratio(errors[0])) <= 1e-6

    def test_a_wrong_jacobian_is_caught(self, bkd):
        """The test that proves the check has teeth.

        A jacobian off by a factor cannot be distinguished from a
        correct one by any shape or ordering assertion, and must be
        caught here or nowhere.
        """

        def wrong(samples):
            n = samples.shape[1]
            return bkd.reshape(3.0 * samples.T, (n, 1, 2))

        model = _model(bkd, derivatives=Derivatives(jacobian_batch=wrong))
        checker = DerivativeChecker(model)
        errors = checker.check_derivatives(
            bkd.array([[0.5], [-1.5]]), relative=True
        )
        assert bkd.to_float(checker.error_ratio(errors[0])) > 1e-6

    def test_split_marshaller_gives_correct_derivatives(self, bkd):
        """Grouping must not disturb correctness.

        With one sample per task the jacobian is reassembled from
        several decoded pieces, which is where a wrong axis or a
        mispaired column would appear.
        """
        model = _model(
            bkd,
            derivatives=Derivatives(jacobian_batch=_jacobian_batch(bkd)),
            samples_per_task=1,
        )
        checker = DerivativeChecker(model)
        errors = checker.check_derivatives(
            bkd.array([[1.5], [0.25]]), relative=True
        )
        assert bkd.to_float(checker.error_ratio(errors[0])) <= 1e-6


def _hvp_batch(bkd):
    """H v for f(x) = sum(x^2): H is 2I, so H v = 2 v, shape (n, nvars)."""

    def hvp(samples, vecs):
        return 2.0 * vecs.T

    return hvp


def _whvp_batch(bkd):
    """Weighted H v: weights are (nqoi, 1), and nqoi is 1 here."""

    def whvp(samples, vecs, weights):
        return 2.0 * vecs.T * weights[0, 0]

    return whvp


def _jvp(bkd):
    """J v for f(x) = sum(x^2): J is 2x^T, so J v = 2 x.v, shape (nqoi, 1)."""

    def jvp(sample, vec):
        return bkd.reshape(bkd.sum(2.0 * sample * vec), (1, 1))

    return jvp


class TestBundleMirrorsEveryCapability:
    """A capability the evaluator has must survive the adapter.

    The gap this guards is silent: a bundle that omits a field reports
    ``None``, which is indistinguishable from a model that never had the
    capability. Nothing raises and no shape is wrong -- the capability
    just disappears, and a caller falls back to finite differences
    without ever learning why.
    """

    def test_directional_fields_are_mirrored(self, bkd):
        model = _model(
            bkd,
            derivatives=Derivatives(
                jvp=_jvp(bkd),
                hvp_batch=_hvp_batch(bkd),
                whvp_batch=_whvp_batch(bkd),
            ),
        )
        d = model.derivatives()
        assert d.jvp is not None
        assert d.hvp is not None
        assert d.hvp_batch is not None
        assert d.whvp is not None
        assert d.whvp_batch is not None

    def test_absent_capabilities_stay_absent(self, bkd):
        """Mirroring must not invent what the evaluator cannot serve."""
        d = _model(bkd, derivatives=Derivatives()).derivatives()
        assert d.jvp is None
        assert d.hvp is None
        assert d.hvp_batch is None
        assert d.whvp is None
        assert d.whvp_batch is None

    def test_hvp_does_not_imply_whvp(self, bkd):
        """The plain and weighted forms are advertised independently."""
        d = _model(
            bkd, derivatives=Derivatives(hvp_batch=_hvp_batch(bkd))
        ).derivatives()
        assert d.hvp_batch is not None
        assert d.whvp_batch is None


class TestDirectionalValues:
    """Directional derivatives come back correct, in the right axes."""

    def test_hvp_batch_shape_and_value(self, bkd):
        model = _model(
            bkd, derivatives=Derivatives(hvp_batch=_hvp_batch(bkd))
        )
        hvp_batch = model.derivatives().hvp_batch
        assert hvp_batch is not None
        X = bkd.array([[1.0, 2.0], [3.0, 4.0]])
        V = bkd.array([[1.0, 0.0], [0.0, 1.0]])
        result = hvp_batch(X, V)
        # sample-first (n, nvars), unlike values and jvps
        assert result.shape == (2, 2)
        bkd.assert_allclose(result, bkd.array([[2.0, 0.0], [0.0, 2.0]]))

    def test_hvp_single_transposes_to_column(self, bkd):
        """The single form is (nvars, 1) where the batch is (n, nvars).

        The axis flip between the two is why the single form is not a
        pass-through, and a missing transpose would still produce a
        two-element array when nvars is 2.
        """
        model = _model(
            bkd, derivatives=Derivatives(hvp_batch=_hvp_batch(bkd))
        )
        hvp = model.derivatives().hvp
        assert hvp is not None
        result = hvp(bkd.array([[1.0], [3.0]]), bkd.array([[5.0], [7.0]]))
        assert result.shape == (2, 1)
        bkd.assert_allclose(result, bkd.array([[10.0], [14.0]]))

    def test_whvp_batch_applies_weights(self, bkd):
        model = _model(
            bkd, derivatives=Derivatives(whvp_batch=_whvp_batch(bkd))
        )
        whvp_batch = model.derivatives().whvp_batch
        assert whvp_batch is not None
        X = bkd.array([[1.0, 2.0], [3.0, 4.0]])
        V = bkd.array([[1.0, 0.0], [0.0, 1.0]])
        result = whvp_batch(X, V, bkd.array([[0.5]]))
        # 0.5 * 2 * v = v
        bkd.assert_allclose(result, bkd.array([[1.0, 0.0], [0.0, 1.0]]))

    def test_jvp_returns_qoi_space_column(self, bkd):
        """J v lands in QoI space, so it is (nqoi, 1), not (nvars, 1)."""
        model = _model(bkd, derivatives=Derivatives(jvp=_jvp(bkd)))
        jvp = model.derivatives().jvp
        assert jvp is not None
        result = jvp(bkd.array([[1.0], [3.0]]), bkd.array([[1.0], [0.0]]))
        assert result.shape == (1, 1)
        # J = 2x = [2, 6], so J . [1, 0] = 2
        bkd.assert_allclose(result, bkd.array([[2.0]]))

    def test_hvp_rejects_a_batch(self, bkd):
        model = _model(
            bkd, derivatives=Derivatives(hvp_batch=_hvp_batch(bkd))
        )
        hvp = model.derivatives().hvp
        assert hvp is not None
        with pytest.raises(ValueError, match=r"single sample"):
            hvp(
                bkd.array([[1.0, 2.0], [3.0, 4.0]]),
                bkd.array([[1.0, 0.0], [0.0, 1.0]]),
            )

    def test_weights_are_rejected_by_the_plain_form(self, bkd):
        """hvp_batch and whvp_batch are distinct capabilities.

        Accepting weights on the plain form would let a caller believe a
        weighting was applied when the request never carried one.
        """
        model = _model(
            bkd, derivatives=Derivatives(hvp_batch=_hvp_batch(bkd))
        )
        hvp_batch = model.derivatives().hvp_batch
        assert hvp_batch is not None
        with pytest.raises(ValueError, match="takes no weights"):
            hvp_batch(
                bkd.array([[1.0], [3.0]]),
                bkd.array([[1.0], [0.0]]),
                bkd.array([[0.5]]),
            )
