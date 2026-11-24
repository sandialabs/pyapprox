"""The observation of weighted sums of state entries.

On a state of five entries, two observations: 0.5 y_1 + 2 y_3 and
-y_4 + 3 y_4 + 0 y_0 (a repeated index sums its weights; a zero weight
pads). Forward against the sums written out, the transpose by
<H x, r> = <x, H^T r> on random data, both to 1e-14; and the refusals.
"""

import numpy as np
import pytest

from pyapprox.implicitfunction.observation_terms import (
    StateObservationProtocol,
    WeightedEntries,
)
from pyapprox.util.backends.numpy import NumpyBkd
from pyapprox.util.backends.protocols import Array, Backend


def _observation(bkd: Backend[Array]) -> WeightedEntries[Array]:
    indices = bkd.asarray(np.array([[1, 3, 3], [4, 4, 0]]), dtype=bkd.index_dtype())
    weights = bkd.asarray([[0.5, 1.0, 1.0], [-1.0, 3.0, 0.0]])
    return WeightedEntries(bkd, 5, indices, weights)


class TestWeightedEntries:
    def test_forward_is_the_weighted_sums(self, bkd: Backend[Array]) -> None:
        observation = _observation(bkd)
        assert isinstance(observation, StateObservationProtocol)
        assert observation.nobs() == 2
        assert observation.domain().size == 5
        state = bkd.asarray(
            [[1.0, -2.0], [3.0, 0.5], [7.0, 7.0], [4.0, 1.0], [5.0, 2.0]]
        )
        expected = bkd.asarray(
            [[0.5 * 3.0 + 2.0 * 4.0, 0.5 * 0.5 + 2.0 * 1.0], [2.0 * 5.0, 2.0 * 2.0]]
        )
        bkd.assert_allclose(observation.apply(state), expected, rtol=1e-14)

    def test_transpose_is_the_adjoint(self, bkd: Backend[Array]) -> None:
        observation = _observation(bkd)
        rng = np.random.default_rng(3)
        x = bkd.asarray(rng.normal(size=(5, 1)))
        r = bkd.asarray(rng.normal(size=(2, 1)))
        bkd.assert_allclose(
            bkd.dot(observation.apply(x).T, r),
            bkd.dot(x.T, observation.apply_transpose(r)),
            rtol=1e-14,
        )


class TestWeightedEntriesRefusals:
    def test_refuses_mismatched_shapes(self, numpy_bkd: NumpyBkd) -> None:
        bkd = numpy_bkd
        with pytest.raises(ValueError, match="one shape"):
            WeightedEntries(
                bkd,
                5,
                bkd.asarray(np.array([[1, 2]]), dtype=bkd.index_dtype()),
                bkd.ones((1, 3)),
            )

    def test_refuses_no_observation(self, numpy_bkd: NumpyBkd) -> None:
        bkd = numpy_bkd
        with pytest.raises(ValueError, match="nobs >= 1"):
            WeightedEntries(
                bkd,
                5,
                bkd.asarray(np.zeros((0, 2)), dtype=bkd.index_dtype()),
                bkd.ones((0, 2)),
            )

    def test_refuses_an_index_out_of_range(self, numpy_bkd: NumpyBkd) -> None:
        bkd = numpy_bkd
        with pytest.raises(ValueError, match=r"indices \[5\] are not in"):
            WeightedEntries(
                bkd,
                5,
                bkd.asarray(np.array([[1, 5]]), dtype=bkd.index_dtype()),
                bkd.ones((1, 2)),
            )
