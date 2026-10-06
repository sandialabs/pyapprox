"""TimeIndexedObservationFunctional against a dense observation matrix.

Every quantity is compared with the same operator written as a dense
``(nobs, nstates)`` matrix ``O``: values ``vec[O y(t_{n_j})]`` (sensor
fastest), the per-step state Jacobian action, and the scalar rows.
"""

import numpy as np
import pytest

from pyapprox.ode.functionals.protocols import (
    TransientFunctionalWithJacobianProtocol,
    TransientFunctionalWithRowsProtocol,
    TransientFunctionalWithStateJacobianActionProtocol,
)
from pyapprox.ode.functionals.time_indexed_observation import (
    TimeIndexedObservationFunctional,
)
from pyapprox.util.backends.numpy import NumpyBkd
from pyapprox.util.backends.protocols import Array, Backend

_NSTATES, _NTIMES, _NPARAMS = 5, 4, 3
# Row 1 reads state 3 twice: repeated indices must add.
_INDICES = np.array([[0, 2], [3, 3]])
_WEIGHTS = np.array([[0.5, -1.5], [0.25, 0.5]])
_TIMES = [1, 3]


def _functional(bkd: Backend[Array]) -> TimeIndexedObservationFunctional[Array]:
    return TimeIndexedObservationFunctional(
        bkd.asarray(_INDICES, dtype=bkd.int64_dtype()),
        bkd.asarray(_WEIGHTS),
        _TIMES,
        _NSTATES,
        _NPARAMS,
        bkd,
    )


def _dense_operator() -> np.ndarray:
    obs = np.zeros((_INDICES.shape[0], _NSTATES))
    for row in range(_INDICES.shape[0]):
        for col, weight in zip(_INDICES[row], _WEIGHTS[row]):
            obs[row, col] += weight
    return obs


def _trajectory(bkd: Backend[Array]) -> Array:
    return bkd.asarray(np.random.default_rng(1).normal(size=(_NSTATES, _NTIMES)))


class TestTimeIndexedObservationFunctional:
    def test_satisfies_protocols(self, numpy_bkd: NumpyBkd) -> None:
        func = _functional(numpy_bkd)
        assert isinstance(func, TransientFunctionalWithJacobianProtocol)
        assert isinstance(
            func, TransientFunctionalWithStateJacobianActionProtocol
        )
        assert isinstance(func, TransientFunctionalWithRowsProtocol)

    def test_values_sensor_fastest(self, bkd: Backend[Array]) -> None:
        func = _functional(bkd)
        sol = _trajectory(bkd)
        obs = _dense_operator()
        sol_np = bkd.to_numpy(sol)
        expected = np.concatenate([obs @ sol_np[:, n] for n in _TIMES])
        assert func.nqoi() == 4
        bkd.assert_allclose(
            func(sol, bkd.zeros((_NPARAMS, 1))),
            bkd.asarray(expected[:, None]),
            rtol=1e-14,
        )

    def test_apply_state_jacobian(self, bkd: Backend[Array]) -> None:
        func = _functional(bkd)
        sol = _trajectory(bkd)
        param = bkd.zeros((_NPARAMS, 1))
        wmat_np = np.random.default_rng(2).normal(size=(_NSTATES, 3))
        wmat = bkd.asarray(wmat_np)
        observed = _dense_operator() @ wmat_np
        zeros = np.zeros_like(observed)
        for time_idx in range(_NTIMES):
            expected = np.vstack(
                [observed if n == time_idx else zeros for n in _TIMES]
            )
            bkd.assert_allclose(
                func.apply_state_jacobian(sol, param, time_idx, wmat),
                bkd.asarray(expected),
                rtol=1e-14,
                atol=1e-15,
            )

    def test_rows(self, bkd: Backend[Array]) -> None:
        """Row q is sensor q % nobs at time q // nobs: its value is Q[q]
        and its state Jacobian is O's row placed at that time."""
        func = _functional(bkd)
        sol = _trajectory(bkd)
        param = bkd.zeros((_NPARAMS, 1))
        values = func(sol, param)
        obs = _dense_operator()
        for qoi_idx in range(func.nqoi()):
            row = func.row_functional(qoi_idx)
            assert row.nqoi() == 1
            bkd.assert_allclose(
                row(sol, param), values[qoi_idx : qoi_idx + 1], rtol=1e-14
            )
            time_pos, sensor = divmod(qoi_idx, obs.shape[0])
            expected = np.zeros((_NSTATES, _NTIMES))
            expected[:, _TIMES[time_pos]] = obs[sensor]
            bkd.assert_allclose(
                row.state_jacobian(sol, param), bkd.asarray(expected),
                rtol=1e-14,
            )
            bkd.assert_allclose(
                row.param_jacobian(sol, param), bkd.zeros((1, _NPARAMS))
            )

    def test_vector_state_jacobian_raises(self, numpy_bkd: NumpyBkd) -> None:
        func = _functional(numpy_bkd)
        with pytest.raises(ValueError, match="scalar QoI"):
            func.state_jacobian(
                _trajectory(numpy_bkd), numpy_bkd.zeros((_NPARAMS, 1))
            )

    def test_invalid_arguments_raise(self, numpy_bkd: NumpyBkd) -> None:
        bkd = numpy_bkd
        indices = bkd.asarray(_INDICES, dtype=bkd.int64_dtype())
        with pytest.raises(ValueError, match="same shape"):
            TimeIndexedObservationFunctional(
                indices, bkd.ones((2, 3)), _TIMES, _NSTATES, _NPARAMS, bkd
            )
        with pytest.raises(ValueError, match="must not be empty"):
            TimeIndexedObservationFunctional(
                indices, bkd.asarray(_WEIGHTS), [], _NSTATES, _NPARAMS, bkd
            )
        with pytest.raises(ValueError, match="out of range"):
            _functional(bkd).row_functional(4)
