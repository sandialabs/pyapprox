"""The flux through essential boundaries, from the raw residual.

At an essential (Dirichlet) DOF the equation ``F_d = 0`` is never solved:
the constraint replaces it by ``u_d = g``. Evaluating the raw residual
row there gives what the constraint had to supply. With ``F = b - K u``,
``-F_d = (K u - b)_d`` is the finite-element "reaction": the support
force of a clamped edge in elasticity, the heat flux through a
fixed-temperature wall, the force on a wall with prescribed velocity, or
the flux across a coupling interface. It is also the Lagrange multiplier
of the essential constraint, which is why the adjoint at essential rows
is called the reaction.

Computing the flux this way, rather than by differentiating the discrete
solution at the boundary, is variationally consistent: it converges at
the rate of the solution's energy error and satisfies global balance
exactly (summed over all essential rows with unit weights, it equals the
negative of the total applied load).
"""

from typing import Generic, Literal, Optional

from pyapprox.optimization.implicitfunction.functionals.derivatives import (
    SteadyFunctionalDerivatives,
)
from pyapprox.pde.models.galerkin.parameterized_residual import (
    GalerkinParameterizedResidual,
)
from pyapprox.pde.parameterizations.derivatives import (
    ParamHVPFn,
    ParamJacobianFn,
)
from pyapprox.pde.steady_view import SteadyViewProtocol
from pyapprox.util.backends.protocols import Array, Backend


class EssentialBoundaryFluxFunctional(Generic[Array]):
    """``Q = -W F(u, p, t)`` restricted to selected essential rows.

    A steady functional: the time is the view's, bound at construction.
    Row ``i`` of ``weights`` defines QoI ``i``; for example unit weights
    on the x-components of a clamped edge give the total x-reaction, and
    weights equal to the node coordinate give a moment.

    Its derivatives come from the raw residual's: ``dQ/du = -W dF/du``
    always, with ``dQ/dp = -W dF/dp`` when the residual provides it, so
    the bundle is first order exactly when the residual has a parameter
    Jacobian. For a scalar QoI the four HVPs are the residual's
    contractions with adjoint ``-w``, so the bundle is second order
    exactly when the residual's is.

    Parameters
    ----------
    residual : GalerkinParameterizedResidual
        The raw residual, built on ``view``'s spatial operator.
    view : SteadyViewProtocol
        Supplies the essential DOFs (its constraint set) and the time.
    dofs : Array
        Essential DOFs whose rows are read, unique. Shape: (ndofs,).
    weights : Array
        One row per QoI. Shape: (nqoi, ndofs).

    Raises
    ------
    TypeError
        If ``residual`` or ``view`` has the wrong type.
    ValueError
        If ``residual`` and ``view`` do not share a spatial operator, a
        DOF is repeated or not essential, or ``weights`` has the wrong
        shape.
    """

    def __init__(
        self,
        residual: GalerkinParameterizedResidual[Array],
        view: SteadyViewProtocol[Array],
        dofs: Array,
        weights: Array,
    ) -> None:
        if not isinstance(residual, GalerkinParameterizedResidual):
            raise TypeError(
                "residual must be a GalerkinParameterizedResidual, got "
                f"{type(residual).__name__}"
            )
        if not isinstance(view, SteadyViewProtocol):
            raise TypeError(
                "view must satisfy SteadyViewProtocol, got "
                f"{type(view).__name__}"
            )
        if residual.spatial_operator() is not view.spatial_operator():
            raise ValueError(
                "residual and view must share one spatial operator"
            )
        bkd = residual.bkd()
        dof_list = [bkd.to_int(dofs[ii]) for ii in range(dofs.shape[0])]
        if len(set(dof_list)) != len(dof_list):
            raise ValueError("dofs must be unique")
        essential = {
            bkd.to_int(d) for d in view.constraint_set().dofs()
        }
        missing = [d for d in dof_list if d not in essential]
        if missing:
            raise ValueError(
                f"dofs {missing} are not essential; the flux is read "
                "from rows the constraints replace"
            )
        if weights.ndim != 2 or weights.shape[1] != len(dof_list):
            raise ValueError(
                f"weights must have shape (nqoi, {len(dof_list)}), got "
                f"{weights.shape}"
            )
        self._residual = residual
        self._bkd = bkd
        self._time = view.time()
        self._nqoi = weights.shape[0]
        # The weights scattered onto all rows: zero off the selection.
        full = bkd.zeros((residual.nstates(), self._nqoi))
        full = bkd.copy(full)
        full[dofs, :] = weights.T
        self._full_weights = full

        derivs = residual.derivatives()
        self._param_jacobian_fn: Optional[ParamJacobianFn[Array]] = (
            derivs.param_jacobian
        )
        self._state_state_fn: Optional[ParamHVPFn[Array]] = None
        self._state_param_fn: Optional[ParamHVPFn[Array]] = None
        self._param_state_fn: Optional[ParamHVPFn[Array]] = None
        self._param_param_fn: Optional[ParamHVPFn[Array]] = None
        self._derivs: SteadyFunctionalDerivatives[Array]
        if derivs.param_jacobian is None:
            self._derivs = SteadyFunctionalDerivatives.none()
            return
        state_state = derivs.state_state_hvp
        state_param = derivs.state_param_hvp
        param_state = derivs.param_state_hvp
        param_param = derivs.param_param_hvp
        if (
            self._nqoi != 1
            or state_state is None
            or state_param is None
            or param_state is None
            or param_param is None
        ):
            self._derivs = SteadyFunctionalDerivatives.first_order(
                self._state_jacobian, self._param_jacobian
            )
            return
        self._state_state_fn = state_state
        self._state_param_fn = state_param
        self._param_state_fn = param_state
        self._param_param_fn = param_param
        self._derivs = SteadyFunctionalDerivatives.second_order(
            self._state_jacobian,
            self._param_jacobian,
            self._state_state_hvp,
            self._state_param_hvp,
            self._param_state_hvp,
            self._param_param_hvp,
        )

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        return self._bkd

    def nqoi(self) -> int:
        """Return the number of QoIs (rows of the weights)."""
        return self._nqoi

    def nstates(self) -> int:
        """Return the number of states."""
        return self._residual.nstates()

    def nparams(self) -> int:
        """Return the number of parameters."""
        return self._residual.nparams()

    def nunique_params(self) -> int:
        """Return the number of parameters unique to the functional."""
        return 0

    def time_dependence(self) -> Literal["steady"]:
        """Return the kind of state this functional reads.

        One steady state, shape ``(nstates, 1)``.
        """
        return "steady"

    def derivatives(self) -> SteadyFunctionalDerivatives[Array]:
        """Return the optional derivatives this functional provides."""
        return self._derivs

    def __call__(self, state: Array, param: Array) -> Array:
        """Evaluate ``Q = -W F``. Shape: (nqoi, 1)."""
        residual = self._residual.residual(state[:, 0], self._time, param[:, 0])
        return -(self._full_weights.T @ residual)[:, None]

    def _adjoint(self) -> Array:
        """``-w`` for the scalar QoI, the adjoint of the contractions."""
        return -self._full_weights[:, 0]

    def _state_jacobian(self, state: Array, param: Array) -> Array:
        """``-W dF/du``. Shape: (nqoi, nstates)."""
        jacobian = self._residual.state_jacobian(
            state[:, 0], self._time, param[:, 0]
        )
        return -(jacobian.T @ self._full_weights).T

    def _param_jacobian(self, state: Array, param: Array) -> Array:
        """``-W dF/dp``. Shape: (nqoi, nparams)."""
        fn = self._param_jacobian_fn
        if fn is None:
            raise RuntimeError(
                "param_jacobian is unavailable; check derivatives() before "
                "calling"
            )
        return -self._full_weights.T @ fn(state[:, 0], self._time, param[:, 0])

    def _contraction(
        self, fn: Optional[ParamHVPFn[Array]], state: Array, param: Array,
        vec: Array,
    ) -> Array:
        if fn is None:
            raise RuntimeError(
                "the HVPs are unavailable; check derivatives() before calling"
            )
        return fn(
            state[:, 0], self._time, param[:, 0], self._adjoint(), vec[:, 0]
        )[:, None]

    def _state_state_hvp(
        self, state: Array, param: Array, wvec: Array
    ) -> Array:
        """``(d^2Q/du^2) w``. Shape: (nstates, 1)."""
        return self._contraction(self._state_state_fn, state, param, wvec)

    def _state_param_hvp(
        self, state: Array, param: Array, vvec: Array
    ) -> Array:
        """``(d^2Q/du dp) v``. Shape: (nstates, 1)."""
        return self._contraction(self._state_param_fn, state, param, vvec)

    def _param_state_hvp(
        self, state: Array, param: Array, wvec: Array
    ) -> Array:
        """``(d^2Q/dp du) w``. Shape: (nparams, 1)."""
        return self._contraction(self._param_state_fn, state, param, wvec)

    def _param_param_hvp(
        self, state: Array, param: Array, vvec: Array
    ) -> Array:
        """``(d^2Q/dp^2) v``. Shape: (nparams, 1)."""
        return self._contraction(self._param_param_fn, state, param, vvec)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(nqoi={self._nqoi}, "
            f"nstates={self.nstates()}, time={self._time})"
        )
