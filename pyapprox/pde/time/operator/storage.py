"""
Storage for time trajectory data in adjoint and HVP computations.
"""

from typing import Generic, Tuple

from pyapprox.util.backends.protocols import Array, Backend


class TimeTrajectoryStorage(Generic[Array]):
    """
    Storage for time trajectory data in adjoint and HVP computations.

    This class manages the storage and retrieval of forward trajectories,
    adjoint trajectories, and sensitivity trajectories used in gradient
    and Hessian-vector product computations.

    Parameters
    ----------
    nstates : int
        Number of state variables.
    nparams : int
        Number of parameters.
    bkd : Backend
        Backend for array operations.
    """

    def __init__(self, nstates: int, nparams: int, bkd: Backend[Array]):
        self._bkd = bkd
        self._nstates = nstates
        self._nparams = nparams
        self._attribute_names = [
            "_forward_sols",
            "_adjoint_sols",
            "_times",
            "_sensitivity_sols",
            "_second_adjoint_sols",
            # Jacobian trajectories (cached to avoid recomputation)
            "_diag_jacobians",
            "_offdiag_jacobians",
            "_param_jacobians",
        ]

    def set_parameter(self, param: Array) -> None:
        """
        Set the parameters and clear cached data.

        Parameters
        ----------
        param : Array
            Parameters. Shape: (nparams, 1)
        """
        self._param = param
        self._clear()

    def _clear(self) -> None:
        """Clear all stored attributes."""
        for attr_name in self._attribute_names:
            if hasattr(self, attr_name):
                delattr(self, attr_name)

    def has_parameter(self, param: Array) -> bool:
        """
        Check if the given parameters match the stored parameters.

        Parameters
        ----------
        param : Array
            Parameters to check.

        Returns
        -------
        bool
            True if the parameters match, False otherwise.
        """
        if not hasattr(self, "_param"):
            return False
        return self._bkd.allclose(param, self._param, atol=3e-16, rtol=3e-16)

    # =========================================================================
    # Forward Trajectory
    # =========================================================================

    def set_forward_trajectory(
        self, sols: Array, times: Array
    ) -> None:
        """
        Set the forward solution trajectory.

        Parameters
        ----------
        sols : Array
            Forward solutions. Shape: (nstates, ntimes)
        times : Array
            Time points. Shape: (ntimes,)
        """
        self._forward_sols = sols
        self._times = times

    def has_forward_trajectory(self) -> bool:
        """Check if the forward trajectory is set."""
        return hasattr(self, "_forward_sols")

    def get_forward_trajectory(self) -> Tuple[Array, Array]:
        """
        Get the forward solution trajectory.

        Returns
        -------
        Tuple[Array, Array]
            Forward solutions (nstates, ntimes) and times (ntimes,)
        """
        if not self.has_forward_trajectory():
            raise AttributeError("must call set_forward_trajectory first")
        return self._forward_sols, self._times

    # =========================================================================
    # Adjoint Trajectory
    # =========================================================================

    def set_adjoint_trajectory(self, sols: Array) -> None:
        """
        Set the adjoint solution trajectory.

        Parameters
        ----------
        sols : Array
            Adjoint solutions. Shape: (nstates, ntimes)
        """
        self._adjoint_sols = sols

    def has_adjoint_trajectory(self) -> bool:
        """Check if the adjoint trajectory is set."""
        return hasattr(self, "_adjoint_sols")

    def get_adjoint_trajectory(self) -> Array:
        """
        Get the adjoint solution trajectory.

        Returns
        -------
        Array
            Adjoint solutions. Shape: (nstates, ntimes)
        """
        if not self.has_adjoint_trajectory():
            raise AttributeError("must call set_adjoint_trajectory first")
        return self._adjoint_sols

    # =========================================================================
    # Forward Sensitivity Trajectory
    # =========================================================================

    def set_sensitivity_trajectory(self, sols: Array) -> None:
        """
        Set the forward sensitivity trajectory w = (dy/dp)·v.

        Parameters
        ----------
        sols : Array
            Sensitivity solutions. Shape: (nstates, ntimes)
        """
        self._sensitivity_sols = sols

    def has_sensitivity_trajectory(self) -> bool:
        """Check if the sensitivity trajectory is set."""
        return hasattr(self, "_sensitivity_sols")

    def get_sensitivity_trajectory(self) -> Array:
        """
        Get the forward sensitivity trajectory.

        Returns
        -------
        Array
            Sensitivity solutions. Shape: (nstates, ntimes)
        """
        if not self.has_sensitivity_trajectory():
            raise AttributeError("must call set_sensitivity_trajectory first")
        return self._sensitivity_sols

    # =========================================================================
    # Second Adjoint Trajectory
    # =========================================================================

    def set_second_adjoint_trajectory(self, sols: Array) -> None:
        """
        Set the second adjoint trajectory for HVP computation.

        Parameters
        ----------
        sols : Array
            Second adjoint solutions. Shape: (nstates, ntimes)
        """
        self._second_adjoint_sols = sols

    def has_second_adjoint_trajectory(self) -> bool:
        """Check if the second adjoint trajectory is set."""
        return hasattr(self, "_second_adjoint_sols")

    def get_second_adjoint_trajectory(self) -> Array:
        """
        Get the second adjoint trajectory.

        Returns
        -------
        Array
            Second adjoint solutions. Shape: (nstates, ntimes)
        """
        if not self.has_second_adjoint_trajectory():
            raise AttributeError(
                "must call set_second_adjoint_trajectory first"
            )
        return self._second_adjoint_sols

    # =========================================================================
    # Jacobian Trajectories (for caching to avoid recomputation)
    # =========================================================================

    def set_jacobian_trajectories(
        self,
        diag: Array,
        offdiag: Array,
        param: Array,
    ) -> None:
        """
        Set all Jacobian trajectories at once.

        Parameters
        ----------
        diag : Array
            Diagonal Jacobians dR_n/dy_n. Shape: (nstates, nstates, ntimes)
        offdiag : Array
            Off-diagonal Jacobians dR_n/dy_{n-1}. Shape: (nstates, nstates, ntimes)
        param : Array
            Parameter Jacobians dR_n/dp. Shape: (nstates, nparams, ntimes)
        """
        self._diag_jacobians = diag
        self._offdiag_jacobians = offdiag
        self._param_jacobians = param

    def has_jacobian_trajectories(self) -> bool:
        """Check if all Jacobian trajectories are set."""
        return (
            hasattr(self, "_diag_jacobians")
            and hasattr(self, "_offdiag_jacobians")
            and hasattr(self, "_param_jacobians")
        )

    def get_jacobian_trajectories(self) -> Tuple[Array, Array, Array]:
        """
        Get all Jacobian trajectories.

        Returns
        -------
        Tuple[Array, Array, Array]
            (diag, offdiag, param) Jacobian arrays.
        """
        if not self.has_jacobian_trajectories():
            raise AttributeError("must call set_jacobian_trajectories first")
        return (
            self._diag_jacobians,
            self._offdiag_jacobians,
            self._param_jacobians,
        )

    def get_jacobians_at_step(self, n: int) -> Tuple[Array, Array, Array]:
        """
        Get Jacobians at a specific time step.

        Parameters
        ----------
        n : int
            Time step index.

        Returns
        -------
        Tuple[Array, Array, Array]
            (diag_n, offdiag_n, param_n) Jacobians at step n.
        """
        if not self.has_jacobian_trajectories():
            raise AttributeError("must call set_jacobian_trajectories first")
        return (
            self._diag_jacobians[:, :, n],
            self._offdiag_jacobians[:, :, n],
            self._param_jacobians[:, :, n],
        )

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"nstates={self._nstates}, "
            f"nparams={self._nparams})"
        )
