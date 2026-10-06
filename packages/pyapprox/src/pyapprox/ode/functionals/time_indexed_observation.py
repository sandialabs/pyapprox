"""Linear observations of the state at selected time steps.

Computes :math:`Q = \\mathrm{vec}[O\\, y(t_{n_j})]_j`, sensor index
fastest, where each row of the observation operator :math:`O` reads a
few weighted states.
"""

from typing import Generic, Sequence

from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.backends.validation import validate_backend


class TimeIndexedObservationFunctional(Generic[Array]):
    """
    Observations :math:`Q = \\mathrm{vec}[O\\, y(t_{n_j})]_j` of the
    state at chosen time steps.

    Row :math:`s` of :math:`O` reads :math:`k` states,

    .. math::

        (O y)_s = \\sum_k w_{sk} \\, y_{i_{sk}},

    so a probe at a point (:math:`k` = the element's local DOFs) and a
    weighted DOF selection (:math:`k = 1`) are both cases of it. The
    operator is stored as indices and weights, never as an
    ``(nobs, nstates)`` matrix, and every product is a gather.

    QoI ``j * nobs + s`` is sensor ``s`` at time ``time_indices[j]``.

    The functional is linear in the state and does not depend on the
    parameters (its ``param_jacobian`` is zero). It supports the
    tangent-linear Jacobian through ``apply_state_jacobian`` and the
    adjoint Jacobian through ``row_functional``.

    Parameters
    ----------
    state_indices : Array
        State index of each term. Integer, shape: ``(nobs, k)``.
    weights : Array
        Weight of each term. Shape: ``(nobs, k)``.
    time_indices : Sequence[int]
        Trajectory columns observed, each in ``[0, ntimes)``.
    nstates : int
        Total number of state variables.
    nparams : int
        Total number of parameters.
    bkd : Backend
        Backend for array operations.
    """

    def __init__(
        self,
        state_indices: Array,
        weights: Array,
        time_indices: Sequence[int],
        nstates: int,
        nparams: int,
        bkd: Backend[Array],
    ) -> None:
        validate_backend(bkd)
        if state_indices.ndim != 2 or state_indices.shape != weights.shape:
            raise ValueError(
                "state_indices and weights must have the same shape "
                f"(nobs, k), got {state_indices.shape} and {weights.shape}"
            )
        if len(time_indices) == 0:
            raise ValueError("time_indices must not be empty")
        self._state_indices = state_indices
        self._weights = weights
        self._time_indices = [int(n) for n in time_indices]
        self._nobs = state_indices.shape[0]
        self._nstates = nstates
        self._nparams = nparams
        self._bkd = bkd

    def bkd(self) -> Backend[Array]:
        """Return the backend."""
        return self._bkd

    def nqoi(self) -> int:
        """Return the number of QoI outputs: ``nobs * len(time_indices)``."""
        return self._nobs * len(self._time_indices)

    def nstates(self) -> int:
        """Return the number of state variables."""
        return self._nstates

    def nparams(self) -> int:
        """Return the total number of parameters."""
        return self._nparams

    def nunique_params(self) -> int:
        """Return the number of parameters unique to the functional."""
        return 0

    def nobservations(self) -> int:
        """Return the number of rows of the observation operator."""
        return self._nobs

    def time_indices(self) -> list[int]:
        """Return the observed trajectory columns."""
        return list(self._time_indices)

    def _observe(self, states: Array) -> Array:
        """Apply the observation operator to the rows of ``states``.

        ``states`` has shape ``(nstates, ncols)``; the result has shape
        ``(nobs, ncols)``.
        """
        result = self._bkd.zeros((self._nobs, states.shape[1]))
        for kk in range(self._state_indices.shape[1]):
            result = result + (
                self._weights[:, kk : kk + 1]
                * states[self._state_indices[:, kk], :]
            )
        return result

    def __call__(self, sol: Array, param: Array) -> Array:
        """
        Evaluate the observations.

        Parameters
        ----------
        sol : Array
            Solution trajectory. Shape: (nstates, ntimes)
        param : Array
            Parameters. Shape: (nparams, 1)

        Returns
        -------
        Array
            Observations, sensor index fastest. Shape: (nqoi, 1)
        """
        blocks = [
            self._observe(sol[:, n_j : n_j + 1]) for n_j in self._time_indices
        ]
        return self._bkd.vstack(blocks)

    def state_jacobian(self, sol: Array, param: Array) -> Array:
        """
        Compute dQ/dy as the gradient of a scalar QoI.

        Defined only when ``nqoi() == 1``; a vector QoI's state Jacobian
        is applied through ``apply_state_jacobian`` or taken row by row
        through ``row_functional``.

        Parameters
        ----------
        sol : Array
            Solution trajectory. Shape: (nstates, ntimes)
        param : Array
            Parameters. Shape: (nparams, 1)

        Returns
        -------
        Array
            State Jacobian. Shape: (nstates, ntimes)
        """
        if self.nqoi() != 1:
            raise ValueError(
                f"state_jacobian is the gradient of a scalar QoI; this "
                f"functional has nqoi={self.nqoi()}. Use "
                "apply_state_jacobian or row_functional."
            )
        dqdu = self._bkd.copy(self._bkd.zeros(sol.shape))
        n_0 = self._time_indices[0]
        for kk in range(self._state_indices.shape[1]):
            idx = self._state_indices[0, kk]
            dqdu[idx, n_0] = dqdu[idx, n_0] + self._weights[0, kk]
        return dqdu

    def apply_state_jacobian(
        self, sol: Array, param: Array, time_idx: int, wmat: Array
    ) -> Array:
        """
        Apply dQ/dy at one time to a matrix.

        Parameters
        ----------
        sol : Array
            Solution trajectory. Shape: (nstates, ntimes)
        param : Array
            Parameters. Shape: (nparams, 1)
        time_idx : int
            Time index, in ``[0, ntimes)``.
        wmat : Array
            Matrix to apply to. Shape: (nstates, ncols)

        Returns
        -------
        Array
            The observation of ``wmat`` in the blocks observing
            ``time_idx``, zeros elsewhere. Shape: (nqoi, ncols)
        """
        zeros = self._bkd.zeros((self._nobs, wmat.shape[1]))
        observed = self._observe(wmat)
        blocks = [
            observed if n_j == time_idx else zeros
            for n_j in self._time_indices
        ]
        return self._bkd.vstack(blocks)

    def param_jacobian(self, sol: Array, param: Array) -> Array:
        """
        Compute dQ/dp: zero, the observations do not depend on the
        parameters directly.

        Parameters
        ----------
        sol : Array
            Solution trajectory. Shape: (nstates, ntimes)
        param : Array
            Parameters. Shape: (nparams, 1)

        Returns
        -------
        Array
            Parameter Jacobian. Shape: (nqoi, nparams)
        """
        return self._bkd.zeros((self.nqoi(), self._nparams))

    def row_functional(
        self, qoi_idx: int
    ) -> "TimeIndexedObservationFunctional[Array]":
        """
        Return QoI ``qoi_idx`` as a scalar functional: one sensor at one
        time.

        Parameters
        ----------
        qoi_idx : int
            QoI index, in ``[0, nqoi)``.

        Returns
        -------
        TimeIndexedObservationFunctional
            Scalar functional (nqoi = 1) with the same parameters.
        """
        if not 0 <= qoi_idx < self.nqoi():
            raise ValueError(
                f"qoi_idx {qoi_idx} out of range [0, {self.nqoi()})"
            )
        time_pos, sensor = divmod(qoi_idx, self._nobs)
        return TimeIndexedObservationFunctional(
            self._state_indices[sensor : sensor + 1, :],
            self._weights[sensor : sensor + 1, :],
            [self._time_indices[time_pos]],
            self._nstates,
            self._nparams,
            self._bkd,
        )
