"""In-memory marshalling, and the grouping choice it exposes.

The marshaller is where the mathematics lives, so these run on both
backends via the ``bkd`` fixture — value shapes, empty arrays and
derivative shapes are exactly where NumPy and Torch diverge.

Two behaviors carry most of the weight. **Vectorized models must not be
regressed**: the default puts a whole batch in one task, so ``f`` is
called once rather than once per sample. And **non-vectorized models must
be splittable**, because for them one task per batch means no progress
and no parallelism.
"""

import pytest
from pyapprox.interface.evaluation.callable_marshaller import (
    UNBOUNDED_SAMPLES_PER_TASK,
    CallableMarshaller,
    CallablePayload,
)
from pyapprox.interface.evaluation.protocols import (
    MarshalError,
    MarshallerProtocol,
    TaskProtocol,
)
from pyapprox.interface.evaluation.records import (
    JobStatus,
    Outcome,
    Request,
)
from pyapprox.interface.functions.derivatives import Derivatives


def _sumsq(bkd):
    """A vectorized model: (nvars, n) -> (1, n)."""

    def model(samples):
        return bkd.sum(samples * samples, axis=0)[None, :]

    return model


def _marshaller(bkd, **kwargs):
    return CallableMarshaller(
        _sumsq(bkd), bkd, nvars=2, nqoi=1, **kwargs
    )


def _succeeded(task, payload):
    return Outcome(
        task=task,
        indices=task.indices,
        status=JobStatus.SUCCEEDED,
        payload=payload,
    )


class TestConformance:
    def test_satisfies_marshaller_protocol(self, bkd):
        assert isinstance(_marshaller(bkd), MarshallerProtocol)

    def test_task_satisfies_task_protocol(self, bkd):
        m = _marshaller(bkd)
        (task,) = m.tasks(bkd.zeros((2, 1)), [0], Request.values_only())
        assert isinstance(task, TaskProtocol)

    def test_dimensions_are_reported(self, bkd):
        m = _marshaller(bkd)
        assert m.nvars() == 2
        assert m.nqoi() == 1
        assert m.ncores() == 1

    def test_run_is_a_bound_method_not_a_closure(self, bkd):
        """Picklability: a process pool has to send this to a worker."""
        import pickle

        m = CallableMarshaller(abs, bkd, nvars=1, nqoi=1)
        assert pickle.loads(pickle.dumps(m.run)) is not None


class TestValidation:
    def test_rejects_non_callable(self, bkd):
        with pytest.raises(TypeError, match="callable"):
            CallableMarshaller("not callable", bkd, nvars=1, nqoi=1)

    @pytest.mark.parametrize("bad", [0, -1])
    def test_rejects_bad_nvars(self, bkd, bad):
        with pytest.raises(ValueError, match="nvars"):
            CallableMarshaller(abs, bkd, nvars=bad, nqoi=1)

    @pytest.mark.parametrize("bad", [0, -1])
    def test_rejects_bad_nqoi(self, bkd, bad):
        with pytest.raises(ValueError, match="nqoi"):
            CallableMarshaller(abs, bkd, nvars=1, nqoi=bad)

    def test_rejects_zero_samples_per_task(self, bkd):
        with pytest.raises(ValueError, match="samples_per_task"):
            CallableMarshaller(abs, bkd, nvars=1, nqoi=1, samples_per_task=0)

    def test_rejects_zero_ncores(self, bkd):
        with pytest.raises(ValueError, match="ncores"):
            CallableMarshaller(abs, bkd, nvars=1, nqoi=1, ncores=0)

    def test_tasks_rejects_index_count_mismatch(self, bkd):
        m = _marshaller(bkd)
        with pytest.raises(ValueError, match="indices"):
            m.tasks(bkd.zeros((2, 3)), [0, 1], Request.values_only())


class TestGrouping:
    def test_default_is_unbounded(self, bkd):
        """A vectorized model must not be fanned out per sample."""
        assert (
            _marshaller(bkd).max_samples_per_task()
            == UNBOUNDED_SAMPLES_PER_TASK
        )

    def test_whole_batch_is_one_task_by_default(self, bkd):
        m = _marshaller(bkd)
        tasks = m.tasks(bkd.zeros((2, 100)), list(range(100)),
                        Request.values_only())
        assert len(tasks) == 1

    def test_vectorized_model_is_called_once(self, bkd):
        """The regression this default exists to prevent."""
        calls = []

        def counting(samples):
            calls.append(samples.shape[1])
            return bkd.sum(samples * samples, axis=0)[None, :]

        m = CallableMarshaller(counting, bkd, nvars=2, nqoi=1)
        (task,) = m.tasks(bkd.ones((2, 50)), list(range(50)),
                          Request.values_only())
        m.run(task)
        assert calls == [50]

    def test_split_marshaller_declares_one_per_task(self, bkd):
        """For a model whose __call__ loops internally."""
        m = _marshaller(bkd, samples_per_task=1)
        assert m.max_samples_per_task() == 1


class TestValues:
    def test_round_trip_is_exact(self, bkd):
        m = _marshaller(bkd)
        X = bkd.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        (task,) = m.tasks(X, [0, 1, 2], Request.values_only())
        decoded = m.values(_succeeded(task, m.run(task)))
        bkd.assert_allclose(
            decoded.values, bkd.array([[17.0, 29.0, 45.0]])
        )

    def test_indices_are_carried_through(self, bkd):
        """Never inferred; the task said which columns it covered."""
        m = _marshaller(bkd)
        (task,) = m.tasks(bkd.zeros((2, 2)), [7, 9], Request.values_only())
        decoded = m.values(_succeeded(task, m.run(task)))
        assert list(decoded.indices) == [7, 9]

    def test_empty_batch(self, bkd):
        m = _marshaller(bkd)
        (task,) = m.tasks(bkd.zeros((2, 0)), [], Request.values_only())
        decoded = m.values(_succeeded(task, m.run(task)))
        assert decoded.values.shape == (1, 0)

    def test_no_payload_raises_marshal_error(self, bkd):
        m = _marshaller(bkd)
        (task,) = m.tasks(bkd.zeros((2, 1)), [0], Request.values_only())
        with pytest.raises(MarshalError, match="no payload"):
            m.values(
                Outcome(
                    task=task,
                    indices=task.indices,
                    status=JobStatus.FAILED,
                )
            )

    def test_wrong_shape_raises_marshal_error(self, bkd):
        """A silently wrong shape would corrupt the batch far downstream."""
        m = _marshaller(bkd)
        (task,) = m.tasks(bkd.zeros((2, 3)), [0, 1, 2],
                          Request.values_only())
        bad = CallablePayload(values=bkd.zeros((1, 2)))
        with pytest.raises(MarshalError, match="shape"):
            m.values(_succeeded(task, bad))

    def test_derivative_fields_absent_by_default(self, bkd):
        m = _marshaller(bkd)
        (task,) = m.tasks(bkd.zeros((2, 1)), [0], Request.values_only())
        decoded = m.values(_succeeded(task, m.run(task)))
        assert decoded.jacobians is None
        assert decoded.hessians is None
        assert decoded.jvps is None
        assert decoded.hvps is None


class TestDerivatives:
    def test_no_capability_by_default(self, bkd):
        """The common case for an external forward solver."""
        assert _marshaller(bkd).derivatives().jacobian_batch is None

    def test_jacobians_are_dispatched_and_decoded(self, bkd):
        def jac_batch(samples):
            # d(sum x^2)/dx = 2x, shape (n, nqoi, nvars)
            n = samples.shape[1]
            return bkd.reshape(2.0 * samples.T, (n, 1, 2))

        m = _marshaller(
            bkd,
            derivatives=Derivatives(jacobian_batch=jac_batch),
        )
        X = bkd.array([[1.0, 2.0], [3.0, 4.0]])
        (task,) = m.tasks(X, [0, 1], Request(jacobians=True))
        decoded = m.values(_succeeded(task, m.run(task)))
        assert decoded.jacobians is not None
        bkd.assert_allclose(
            decoded.jacobians,
            bkd.reshape(2.0 * X.T, (2, 1, 2)),
        )

    def test_requesting_absent_capability_raises(self, bkd):
        """Reaching this is a bug, so the message must name the field."""
        m = _marshaller(bkd)
        (task,) = m.tasks(bkd.zeros((2, 1)), [0], Request(jacobians=True))
        with pytest.raises(MarshalError, match="jacobian_batch"):
            m.run(task)

    def test_values_and_jacobians_come_from_one_call(self, bkd):
        """The fused case: one invocation answers both."""
        calls = []

        def counting(samples):
            calls.append("values")
            return bkd.sum(samples * samples, axis=0)[None, :]

        def jac_batch(samples):
            calls.append("jacobian")
            n = samples.shape[1]
            return bkd.reshape(2.0 * samples.T, (n, 1, 2))

        m = CallableMarshaller(
            counting,
            bkd,
            nvars=2,
            nqoi=1,
            derivatives=Derivatives(jacobian_batch=jac_batch),
        )
        tasks = m.tasks(
            bkd.ones((2, 3)), [0, 1, 2], Request(values=True, jacobians=True)
        )
        assert len(tasks) == 1
        m.run(tasks[0])
        assert calls == ["values", "jacobian"]


class TestIndexCorrespondence:
    """Entry k of every decoded array must describe ``indices[k]``.

    Required of every marshaller, because nothing downstream can check
    it: a permuted decode has the right shape and the right count, and
    is wrong. Shape assertions elsewhere in this module do not cover it.

    This marshaller satisfies the contract by construction -- it hands a
    whole slice to one call and receives one array back -- so these
    tests are a guard against that changing, and the template for
    marshallers that read files or run their own concurrency, where the
    contract is genuinely at risk.
    """

    def test_values_identify_their_own_column(self, bkd):
        """Column j encodes j, so a permutation is visible."""
        m = _marshaller(bkd)
        X = bkd.array([[float(j) for j in range(5)] for _ in range(2)])
        (task,) = m.tasks(X, [0, 1, 2, 3, 4], Request.values_only())
        decoded = m.values(_succeeded(task, m.run(task)))
        for position, index in enumerate(decoded.indices):
            bkd.assert_allclose(
                decoded.values[:, position],
                bkd.asarray([2.0 * float(index) ** 2]),
            )

    def test_correspondence_holds_for_a_multi_sample_task(self, bkd):
        """The only shape that can express the failure.

        A per-sample marshaller cannot permute anything, so the risk
        lives entirely in tasks carrying several samples.
        """
        m = _marshaller(bkd, samples_per_task=4)
        X = bkd.array([[float(j) for j in range(4)] for _ in range(2)])
        (task,) = m.tasks(X, [10, 11, 12, 13], Request.values_only())
        decoded = m.values(_succeeded(task, m.run(task)))
        assert list(decoded.indices) == [10, 11, 12, 13]
        bkd.assert_allclose(
            decoded.values, bkd.array([[0.0, 2.0, 8.0, 18.0]])
        )

    def test_jacobians_follow_the_same_correspondence(self, bkd):
        """Derivatives are indexed sample-first, and equally at risk."""

        def jac_batch(samples):
            n = samples.shape[1]
            return bkd.reshape(2.0 * samples.T, (n, 1, 2))

        m = _marshaller(
            bkd, derivatives=Derivatives(jacobian_batch=jac_batch)
        )
        X = bkd.array([[float(j) for j in range(3)] for _ in range(2)])
        (task,) = m.tasks(X, [0, 1, 2], Request(jacobians=True))
        decoded = m.values(_succeeded(task, m.run(task)))
        assert decoded.jacobians is not None
        for position, index in enumerate(decoded.indices):
            bkd.assert_allclose(
                decoded.jacobians[position],
                bkd.reshape(
                    bkd.asarray([2.0 * float(index), 2.0 * float(index)]),
                    (1, 2),
                ),
            )

    def test_count_mismatch_is_rejected(self, bkd):
        """Catches truncation. Does not catch permutation -- see above."""
        m = _marshaller(bkd)
        (task,) = m.tasks(bkd.zeros((2, 3)), [0, 1, 2],
                          Request.values_only())
        bad = CallablePayload(values=bkd.zeros((1, 2)))
        with pytest.raises(MarshalError, match="shape"):
            m.values(_succeeded(task, bad))


class TestRelease:
    def test_release_is_a_noop(self, bkd):
        """In-process work holds no external resource."""
        m = _marshaller(bkd)
        (task,) = m.tasks(bkd.zeros((2, 1)), [0], Request.values_only())
        assert m.release(_succeeded(task, m.run(task))) is None
