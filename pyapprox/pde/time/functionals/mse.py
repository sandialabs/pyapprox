"""
Mean Squared Error functional for transient problems.

Computes Q = (1/2σ²) Σᵢ ||y(tᵢ) - obs(tᵢ)||² over observation times.
"""

from typing import Generic, List, Tuple

from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.backends.validation import validate_backend


class TransientMSEFunctional(Generic[Array]):
    """
    Mean Squared Error functional for time-dependent observations.

    Computes Q = (1/2σ²) Σᵢ ||y_k(tᵢ) - obs_k(tᵢ)||² where the sum is over
    observed state indices k and observed time indices tᵢ.

    Parameters
    ----------
    nstates : int
        Total number of state variables.
    nresidual_params : int
        Number of parameters in the ODE residual.
    obs_tuples : List[Tuple[int, Array]]
        List of (state_idx, time_indices) pairs specifying observations.
        Each tuple identifies which state and at which times it is observed.
    noise_std : float, optional
        Known noise standard deviation. If None, treated as a parameter.
    bkd : Backend
        Backend for array operations.
    """

    def __init__(
        self,
        nstates: int,
        nresidual_params: int,
        obs_tuples: List[Tuple[int, Array]],
        noise_std: float = None,
        bkd: Backend[Array] = None,
    ):
        validate_backend(bkd)
        self._nstates = nstates
        self._noise_std = noise_std
        self._obs_state_indices = bkd.asarray([tup[0] for tup in obs_tuples], dtype=int)
        self._obs_time_indices = [tup[1] for tup in obs_tuples]
        self._nobs = sum([indices.shape[0] for indices in self._obs_time_indices])
        # If noise_std is None, sigma is a parameter
        self._nparams = nresidual_params + self.nunique_params()
        self._bkd = bkd
        self._param = None
        self._obs = None

    def bkd(self) -> Backend[Array]:
        """Return the backend."""
        return self._bkd

    def nqoi(self) -> int:
        """Return the number of QoI outputs."""
        return 1

    def nstates(self) -> int:
        """Return the number of state variables."""
        return self._nstates

    def nparams(self) -> int:
        """Return the total number of parameters."""
        return self._nparams

    def nunique_params(self) -> int:
        """Return number of parameters unique to the functional."""
        if self._noise_std is None:
            return 1  # sigma is a parameter
        return 0

    def set_observations(self, observations: Array) -> None:
        """
        Set the target observations.

        Parameters
        ----------
        observations : Array
            Flattened observations. Shape: (nobs,) where nobs is total
            number of observation points across all states and times.
        """
        if observations.shape[0] != self._nobs:
            raise ValueError(
                f"observations has {observations.shape[0]} elements but "
                f"expected {self._nobs}"
            )
        self._obs = observations

    def set_param(self, param: Array) -> None:
        """
        Set the parameters.

        Parameters
        ----------
        param : Array
            Parameters. Shape: (nparams, 1)
        """
        self._param = param
        if self._noise_std is None:
            self._sigma = float(param[0, 0])
        else:
            self._sigma = self._noise_std

    def _observations_from_solution(self, sol: Array) -> Array:
        """
        Extract observations from solution trajectory.

        Parameters
        ----------
        sol : Array
            Solution trajectory. Shape: (nstates, ntimes)

        Returns
        -------
        Array
            Predicted observations. Shape: (nobs,)
        """
        obs = []
        for state_idx, time_idx in zip(self._obs_state_indices, self._obs_time_indices):
            obs.append(sol[int(state_idx)][time_idx])
        return self._bkd.hstack(obs)

    def __call__(self, sol: Array, param: Array) -> Array:
        """
        Evaluate the MSE functional.

        Parameters
        ----------
        sol : Array
            Solution trajectory. Shape: (nstates, ntimes)
        param : Array
            Parameters. Shape: (nparams, 1)

        Returns
        -------
        Array
            Q = (1/2σ²) ||pred - obs||². Shape: (1, 1)
        """
        self.set_param(param)
        pred_obs = self._observations_from_solution(sol)
        mse = self._bkd.sum((pred_obs - self._obs) ** 2) / (2.0 * self._sigma**2)
        return self._bkd.atleast_2d(mse)

    def state_jacobian(self, sol: Array, param: Array) -> Array:
        """
        Compute dQ/dy.

        Non-zero only at observed state/time pairs.

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
        self.set_param(param)
        dqdu = self._bkd.zeros(sol.shape)
        dqdu = self._bkd.copy(dqdu)
        idx = 0
        for state_idx, time_idx in zip(self._obs_state_indices, self._obs_time_indices):
            state_idx = int(state_idx)
            n_obs_at_state = time_idx.shape[0]
            dqdu[state_idx, time_idx] = (
                sol[state_idx, time_idx] - self._obs[idx : idx + n_obs_at_state]
            ) / self._sigma**2
            idx += n_obs_at_state
        return dqdu

    def param_jacobian(self, sol: Array, param: Array) -> Array:
        """
        Compute dQ/dp.

        Non-zero only if sigma is a parameter.

        Parameters
        ----------
        sol : Array
            Solution trajectory. Shape: (nstates, ntimes)
        param : Array
            Parameters. Shape: (nparams, 1)

        Returns
        -------
        Array
            Parameter Jacobian. Shape: (1, nparams)
        """
        self.set_param(param)
        jac = self._bkd.zeros((1, self._nparams))
        if self._noise_std is None:
            # dQ/dσ = -Q/σ (since Q ~ 1/σ²)
            qoi_val = float(self(sol, param)[0, 0])
            jac = self._bkd.copy(jac)
            jac[0, 0] = -2.0 * qoi_val / self._sigma
        return jac

    # =========================================================================
    # HVP Methods
    # =========================================================================

    def state_state_hvp(
        self, sol: Array, param: Array, time_idx: int, wvec: Array
    ) -> Array:
        """
        Compute (d^2Q/dy^2)·w at a specific time.

        For MSE, d²Q/dy² = (1/σ²)·I at observation times.

        Parameters
        ----------
        sol : Array
            Solution trajectory. Shape: (nstates, ntimes)
        param : Array
            Parameters. Shape: (nparams, 1)
        time_idx : int
            Time index.
        wvec : Array
            Direction vector. Shape: (nstates, 1)

        Returns
        -------
        Array
            HVP result. Shape: (nstates, 1)
        """
        self.set_param(param)
        hvp = self._bkd.zeros((self._nstates, 1))
        hvp = self._bkd.copy(hvp)

        # Check if this time is observed for any state
        for state_idx, obs_time_idx in zip(
            self._obs_state_indices, self._obs_time_indices
        ):
            state_idx = int(state_idx)
            if time_idx in obs_time_idx:
                hvp[state_idx, 0] = float(wvec[state_idx, 0]) / self._sigma**2

        return hvp

    def state_param_hvp(
        self, sol: Array, param: Array, time_idx: int, vvec: Array
    ) -> Array:
        """
        Compute (d^2Q/dy dp)·v at a specific time.

        Non-zero only if sigma is a parameter.
        """
        self.set_param(param)
        hvp = self._bkd.zeros((self._nstates, 1))

        if self._noise_std is None:
            # d²Q/(dy dσ) = -2(y - obs)/σ³
            hvp = self._bkd.copy(hvp)
            v_sigma = float(vvec[0, 0])
            for state_idx, obs_time_idx in zip(
                self._obs_state_indices, self._obs_time_indices
            ):
                state_idx = int(state_idx)
                # Find the position in obs array
                idx = 0
                for si, ti in zip(self._obs_state_indices, self._obs_time_indices):
                    if int(si) == state_idx:
                        break
                    idx += ti.shape[0]

                if time_idx in obs_time_idx:
                    local_idx = list(obs_time_idx).index(time_idx)
                    resid = float(sol[state_idx, time_idx]) - float(
                        self._obs[idx + local_idx]
                    )
                    hvp[state_idx, 0] = -2.0 * resid * v_sigma / self._sigma**3

        return hvp

    def param_state_hvp(
        self, sol: Array, param: Array, time_idx: int, wvec: Array
    ) -> Array:
        """
        Compute (d^2Q/dp dy)·w at a specific time.

        Non-zero only if sigma is a parameter.
        """
        self.set_param(param)
        hvp = self._bkd.zeros((self._nparams, 1))

        if self._noise_std is None:
            # d²Q/(dσ dy) = -2(y - obs)/σ³ (same as state_param by symmetry)
            hvp = self._bkd.copy(hvp)
            for state_idx, obs_time_idx in zip(
                self._obs_state_indices, self._obs_time_indices
            ):
                state_idx = int(state_idx)
                idx = 0
                for si, ti in zip(self._obs_state_indices, self._obs_time_indices):
                    if int(si) == state_idx:
                        break
                    idx += ti.shape[0]

                if time_idx in obs_time_idx:
                    local_idx = list(obs_time_idx).index(time_idx)
                    resid = float(sol[state_idx, time_idx]) - float(
                        self._obs[idx + local_idx]
                    )
                    w_state = float(wvec[state_idx, 0])
                    hvp[0, 0] += -2.0 * resid * w_state / self._sigma**3

        return hvp

    def param_param_hvp(self, sol: Array, param: Array, vvec: Array) -> Array:
        """
        Compute (d^2Q/dp^2)·v.

        Non-zero only if sigma is a parameter.
        """
        self.set_param(param)
        hvp = self._bkd.zeros((self._nparams, 1))

        if self._noise_std is None:
            # d²Q/dσ² = 6Q/σ² (since dQ/dσ = -2Q/σ)
            hvp = self._bkd.copy(hvp)
            qoi_val = float(self(sol, param)[0, 0])
            v_sigma = float(vvec[0, 0])
            hvp[0, 0] = 6.0 * qoi_val * v_sigma / self._sigma**2

        return hvp

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"nstates={self._nstates}, "
            f"nparams={self._nparams}, "
            f"nobs={self._nobs}, "
            f"noise_std={self._noise_std})"
        )
