"""
Time-integrated weighted L2 functional for transient problems.

Computes :math:`Q = \\sum_j w_j\\, \\hat{y}_j^\\top W \\hat{y}_j`, the
scheme-consistent quadrature approximation of
:math:`\\int_0^T y(t)^\\top W y(t)\\, dt`.
"""

from typing import Generic, Optional

from pyapprox.ode.time_quadrature import TrajectoryQuadratureProtocol
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.backends.validation import validate_backend


class TimeIntegratedWeightedL2Functional(Generic[Array]):
    """
    Time-integrated weighted L2 functional
    :math:`Q \\approx \\int_0^T y(t)^\\top W y(t)\\, dt`.

    The time quadrature is NOT a constructor argument: it is a property
    of the time-integration scheme, injected via
    :meth:`set_time_quadrature` by the component that owns the scheme
    (``GalerkinTransientForwardModel`` injects after each forward
    solve; manual drivers pass ``stepper.trajectory_quadrature(times)``
    from their own solve). This makes mismatched scheme/quadrature
    orders unrepresentable. Unlike endpoint functionals, the state
    Jacobian is nonzero at every step, exercising the per-step adjoint
    right-hand side.

    Per-step HVP methods require a time-diagonal quadrature (all nodal
    rules); the midpoint rule couples adjacent steps and raises.

    Parameters
    ----------
    weight_matrix : Array
        Symmetric matrix W, e.g. a subdomain-weighted mass matrix.
        Shape: ``(nstates, nstates)``.
    nparams : int
        Total number of parameters.
    bkd : Backend
        Backend for array operations.
    """

    def __init__(
        self,
        weight_matrix: Array,
        nparams: int,
        bkd: Backend[Array],
    ):
        validate_backend(bkd)
        if (
            weight_matrix.ndim != 2
            or weight_matrix.shape[0] != weight_matrix.shape[1]
        ):
            raise ValueError(
                "weight_matrix must have shape (nstates, nstates), got "
                f"{weight_matrix.shape}"
            )
        asymmetry = bkd.max(bkd.abs(weight_matrix - weight_matrix.T))
        scale = bkd.max(bkd.abs(weight_matrix))
        if bkd.to_float(asymmetry) > 1e-12 * max(bkd.to_float(scale), 1.0):
            raise ValueError(
                "weight_matrix must be symmetric; the state Jacobian "
                "formula 2 w_j W y_j assumes W = W^T"
            )
        self._weight_matrix = weight_matrix
        self._nstates = weight_matrix.shape[0]
        self._nparams = nparams
        self._bkd = bkd
        self._quadrature: Optional[
            TrajectoryQuadratureProtocol[Array]
        ] = None

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
        """Return the number of parameters unique to the functional."""
        return 0

    def weight_matrix(self) -> Array:
        """Return W. Shape: ``(nstates, nstates)``."""
        return self._weight_matrix

    def set_time_quadrature(
        self, quadrature: TrajectoryQuadratureProtocol[Array]
    ) -> None:
        """
        Inject the scheme-implied trajectory quadrature.

        Parameters
        ----------
        quadrature : TrajectoryQuadratureProtocol
            The rule of the stepper that produced the trajectory,
            obtained from ``stepper.trajectory_quadrature(times)``.
        """
        if not isinstance(quadrature, TrajectoryQuadratureProtocol):
            raise TypeError(
                "quadrature must satisfy TrajectoryQuadratureProtocol, "
                f"got {type(quadrature).__name__}"
            )
        self._quadrature = quadrature

    def time_quadrature(self) -> TrajectoryQuadratureProtocol[Array]:
        """Return the injected quadrature, raising if not set."""
        if self._quadrature is None:
            raise RuntimeError(
                "time quadrature not set. It is injected by the model "
                "that owns the time-integration scheme; manual drivers "
                "must call set_time_quadrature("
                "stepper.trajectory_quadrature(times)) with the stepper "
                "and times of their solve."
            )
        return self._quadrature

    def _validate_sol(
        self, sol: Array, quadrature: TrajectoryQuadratureProtocol[Array]
    ) -> None:
        if sol.shape[0] != self._nstates:
            raise ValueError(
                f"sol has {sol.shape[0]} states but weight_matrix has "
                f"{self._nstates}"
            )
        if sol.shape[1] != quadrature.ntimes():
            raise ValueError(
                f"sol has {sol.shape[1]} times but the injected "
                f"quadrature covers {quadrature.ntimes()}"
            )

    def __call__(self, sol: Array, param: Array) -> Array:
        """
        Evaluate the functional.

        Parameters
        ----------
        sol : Array
            Solution trajectory. Shape: (nstates, ntimes)
        param : Array
            Parameters. Shape: (nparams, 1)

        Returns
        -------
        Array
            :math:`Q = \\sum_j w_j\\, \\hat{y}_j^\\top W \\hat{y}_j`.
            Shape: (1, 1)
        """
        quadrature = self.time_quadrature()
        self._validate_sol(sol, quadrature)
        sampled = quadrature.sample(sol)
        weighted = self._bkd.dot(self._weight_matrix, sampled)
        per_sample = self._bkd.sum(sampled * weighted, axis=0)
        return self._bkd.reshape(
            self._bkd.sum(quadrature.weights() * per_sample), (1, 1)
        )

    def state_jacobian(self, sol: Array, param: Array) -> Array:
        """
        Compute dQ/dy, nonzero at every time step.

        Chain rule through the sampling operator:
        :math:`dQ/dy = S^\\top (2 w \\odot W \\hat{y})`.

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
        quadrature = self.time_quadrature()
        self._validate_sol(sol, quadrature)
        sampled = quadrature.sample(sol)
        weighted = self._bkd.dot(self._weight_matrix, sampled)
        graded = 2.0 * weighted * self._bkd.reshape(
            quadrature.weights(), (1, quadrature.nsamples())
        )
        return quadrature.accumulate(graded)

    def param_jacobian(self, sol: Array, param: Array) -> Array:
        """
        Compute dQ/dp. Zero: Q does not depend on parameters directly.

        Returns
        -------
        Array
            Parameter Jacobian. Shape: (1, nparams)
        """
        return self._bkd.zeros((1, self._nparams))

    # =========================================================================
    # HVP Methods (time-diagonal quadrature only)
    # =========================================================================

    def state_state_hvp(
        self, sol: Array, param: Array, time_idx: int, wvec: Array
    ) -> Array:
        """
        Compute (d^2Q/dy^2)·w at one time: :math:`2 \\omega_n W w`
        with :math:`\\omega = S^\\top w` the per-column weights.

        Raises for time-coupled quadratures (midpoint rule), whose
        second derivative is block-tridiagonal in time — outside the
        per-step HVP interface.

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
        nodal_weights = self.time_quadrature().nodal_weights()
        return (
            2.0
            * nodal_weights[time_idx]
            * self._bkd.dot(self._weight_matrix, wvec)
        )

    def state_param_hvp(
        self, sol: Array, param: Array, time_idx: int, vvec: Array
    ) -> Array:
        """Compute (d^2Q/dy dp)·v. Zero: no state-parameter coupling."""
        return self._bkd.zeros((self._nstates, 1))

    def param_state_hvp(
        self, sol: Array, param: Array, time_idx: int, wvec: Array
    ) -> Array:
        """Compute (d^2Q/dp dy)·w. Zero: no state-parameter coupling."""
        return self._bkd.zeros((self._nparams, 1))

    def param_param_hvp(self, sol: Array, param: Array, vvec: Array) -> Array:
        """Compute (d^2Q/dp^2)·v. Zero: Q does not depend on parameters."""
        return self._bkd.zeros((self._nparams, 1))

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"nstates={self._nstates}, "
            f"nparams={self._nparams}, "
            f"quadrature={self._quadrature!r})"
        )
