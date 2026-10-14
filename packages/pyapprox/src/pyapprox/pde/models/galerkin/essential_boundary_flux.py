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

from typing import Generic, Literal, Optional, Tuple

from pyapprox.ode.functionals.protocols import TransientFunctionalDerivatives
from pyapprox.ode.mass_matrix import MassMatrixProtocol
from pyapprox.ode.time_quadrature import (
    StepResidualRuleProtocol,
    TrajectoryQuadratureProtocol,
)
from pyapprox.optimization.implicitfunction.functionals.derivatives import (
    SteadyFunctionalDerivatives,
)
from pyapprox.pde.galerkin.protocols.system import (
    GalerkinTransientSystemProtocol,
)
from pyapprox.pde.galerkin.time_integration.constrained_mass import (
    ConstrainedMassMatrixProtocol,
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


class _EssentialRowWeights(Generic[Array]):
    """Weights on selected essential rows, applied without forming the
    full ``(nstates, nqoi)`` weight matrix.

    QoI ``i`` is ``sum_k weights[i, k] R[dofs[k]]`` for a residual
    ``R``: :meth:`apply` reads only the selected rows, so its cost is
    independent of the number of states; :meth:`adjoint` forms the one
    ``(nstates,)`` vector a contraction needs.

    Raises
    ------
    ValueError
        If a DOF is repeated or not essential, or ``weights`` has the
        wrong shape.
    """

    def __init__(
        self,
        bkd: Backend[Array],
        nstates: int,
        essential_dofs: Array,
        dofs: Array,
        weights: Array,
    ) -> None:
        dof_list = [bkd.to_int(dofs[ii]) for ii in range(dofs.shape[0])]
        if len(set(dof_list)) != len(dof_list):
            raise ValueError("dofs must be unique")
        essential = {bkd.to_int(d) for d in essential_dofs}
        missing = [d for d in dof_list if d not in essential]
        if missing:
            raise ValueError(
                f"dofs {missing} are not essential; the flux is read "
                "from rows the constraints replace"
            )
        if weights.ndim != 2 or weights.shape[1] != len(dof_list):
            raise ValueError(
                f"weights must have shape (nqoi, {len(dof_list)}), got {weights.shape}"
            )
        self._bkd = bkd
        self._nstates = nstates
        self._dofs = dofs
        self._weights = weights

    def nqoi(self) -> int:
        return int(self._weights.shape[0])

    def apply(self, rows: Array) -> Array:
        """``weights @ rows[dofs]``: shape ``(nqoi,)`` from a vector
        ``(nstates,)``, ``(nqoi, ncols)`` from a matrix ``(nstates,
        ncols)``, dense or sparse."""
        return self._weights @ rows[self._dofs]

    def adjoint(self, qoi: int) -> Array:
        """QoI ``qoi``'s weights placed on all rows, zero off the
        selection. Shape: ``(nstates,)``."""
        out = self._bkd.copy(self._bkd.zeros((self._nstates,)))
        out[self._dofs] = self._weights[qoi]
        return out


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
                f"view must satisfy SteadyViewProtocol, got {type(view).__name__}"
            )
        if residual.spatial_operator() is not view.spatial_operator():
            raise ValueError("residual and view must share one spatial operator")
        bkd = residual.bkd()
        self._residual = residual
        self._bkd = bkd
        self._time = view.time()
        self._nqoi = weights.shape[0]
        self._row_weights = _EssentialRowWeights(
            bkd,
            residual.nstates(),
            view.constraint_set().dofs(),
            dofs,
            weights,
        )

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
        return -self._row_weights.apply(residual)[:, None]

    def _adjoint(self) -> Array:
        """``-w`` for the scalar QoI, the adjoint of the contractions."""
        return -self._row_weights.adjoint(0)

    def _state_jacobian(self, state: Array, param: Array) -> Array:
        """``-W dF/du``. Shape: (nqoi, nstates)."""
        jacobian = self._residual.state_jacobian(state[:, 0], self._time, param[:, 0])
        return -self._row_weights.apply(jacobian)

    def _param_jacobian(self, state: Array, param: Array) -> Array:
        """``-W dF/dp``. Shape: (nqoi, nparams)."""
        fn = self._param_jacobian_fn
        if fn is None:
            raise RuntimeError(
                "param_jacobian is unavailable; check derivatives() before calling"
            )
        return -self._row_weights.apply(fn(state[:, 0], self._time, param[:, 0]))

    def _contraction(
        self,
        fn: Optional[ParamHVPFn[Array]],
        state: Array,
        param: Array,
        vec: Array,
    ) -> Array:
        if fn is None:
            raise RuntimeError(
                "the HVPs are unavailable; check derivatives() before calling"
            )
        return fn(state[:, 0], self._time, param[:, 0], self._adjoint(), vec[:, 0])[
            :, None
        ]

    def _state_state_hvp(self, state: Array, param: Array, wvec: Array) -> Array:
        """``(d^2Q/du^2) w``. Shape: (nstates, 1)."""
        return self._contraction(self._state_state_fn, state, param, wvec)

    def _state_param_hvp(self, state: Array, param: Array, vvec: Array) -> Array:
        """``(d^2Q/du dp) v``. Shape: (nstates, 1)."""
        return self._contraction(self._state_param_fn, state, param, vvec)

    def _param_state_hvp(self, state: Array, param: Array, wvec: Array) -> Array:
        """``(d^2Q/dp du) w``. Shape: (nparams, 1)."""
        return self._contraction(self._param_state_fn, state, param, wvec)

    def _param_param_hvp(self, state: Array, param: Array, vvec: Array) -> Array:
        """``(d^2Q/dp^2) v``. Shape: (nparams, 1)."""
        return self._contraction(self._param_param_fn, state, param, vvec)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(nqoi={self._nqoi}, "
            f"nstates={self.nstates()}, time={self._time})"
        )


class TimeIntegratedEssentialBoundaryFluxFunctional(Generic[Array]):
    """The flux through essential boundaries integrated over the solve.

    The time-integrated analog of :class:`EssentialBoundaryFluxFunctional`:
    total heat lost through a fixed-temperature wall over ``[0, T]``, or
    the impulse a clamped support delivers. It is the raw step residual
    summed over all steps and read on the selected essential rows,

    .. math::

        Q = W^\\top \\Big( M Y c
            - \\sum_j w_j F(\\hat{y}_j, t_j, p) \\Big),

    with the scheme's step-residual rule supplying the stage states
    :math:`\\hat{y}_j`, times :math:`t_j`, weights :math:`w_j` and mass
    telescope :math:`c` (:math:`c = e_N - e_0` for one-step schemes). In
    the steady limit it is :math:`-W^\\top F` times the duration.

    Integrating the steady flux over time instead would drop
    :math:`M \\dot u` on the boundary rows, the content stored in the
    elements touching the boundary: that converges only at first order
    and breaks discrete conservation. This form is exact: summed over
    all essential rows with unit weights it equals the change of the
    total content minus the time-integrated sources and natural fluxes,
    as the scheme computes them.

    ``F`` is the raw residual, not the time-integration adapter's, which
    replaces the essential rows. ``M`` is taken from the injected rule:
    the mass the scheme stepped with, consistent or lumped, before its
    essential rows were replaced. So the functional cannot pair a scheme
    with a mass it did not use. The scheme's rule is injected through
    :meth:`set_time_quadrature`; a scheme without one (Heun, whose step
    evaluates ``F`` at an unstored predictor) is refused.

    Parameters
    ----------
    residual : GalerkinParameterizedResidual
        The raw residual, built on ``system``'s spatial operator.
    system : GalerkinTransientSystemProtocol
        Supplies the essential DOFs (its constraint set).
    dofs : Array
        Essential DOFs whose rows are read, unique. Shape: (ndofs,).
    weights : Array
        One row per QoI. Shape: (nqoi, ndofs).

    Raises
    ------
    TypeError
        If ``residual`` or ``system`` has the wrong type.
    ValueError
        If ``residual`` and ``system`` do not share a spatial operator, a
        DOF is repeated or not essential, or ``weights`` has the wrong
        shape.
    """

    def __init__(
        self,
        residual: GalerkinParameterizedResidual[Array],
        system: GalerkinTransientSystemProtocol[Array],
        dofs: Array,
        weights: Array,
    ) -> None:
        if not isinstance(residual, GalerkinParameterizedResidual):
            raise TypeError(
                "residual must be a GalerkinParameterizedResidual, got "
                f"{type(residual).__name__}"
            )
        if not isinstance(system, GalerkinTransientSystemProtocol):
            raise TypeError(
                "system must satisfy GalerkinTransientSystemProtocol, got "
                f"{type(system).__name__}"
            )
        if residual.spatial_operator() is not system.spatial_operator():
            raise ValueError("residual and system must share one spatial operator")
        self._residual = residual
        self._bkd = residual.bkd()
        self._nqoi = weights.shape[0]
        self._row_weights = _EssentialRowWeights(
            self._bkd,
            residual.nstates(),
            system.constraint_set().dofs(),
            dofs,
            weights,
        )
        self._rule: Optional[StepResidualRuleProtocol[Array]] = None
        self._mass: Optional[MassMatrixProtocol[Array]] = None

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

    def time_dependence(self) -> Literal["transient"]:
        """Return the kind of state this functional reads.

        A trajectory of states, shape ``(nstates, ntimes)``.
        """
        return "transient"

    def derivatives(self) -> TransientFunctionalDerivatives[Array]:
        """Return the optional derivatives this functional provides."""
        return TransientFunctionalDerivatives.none()

    def set_time_quadrature(
        self, quadrature: TrajectoryQuadratureProtocol[Array]
    ) -> None:
        """Take the scheme's step-residual rule from its quadrature.

        Raises
        ------
        TypeError
            If ``quadrature`` does not satisfy the protocol.
        ValueError
            If the scheme provides no step-residual rule, or one whose
            stage states are nonlinear in the trajectory (not yet
            supported).
        TypeError
            If the scheme's mass does not keep the mass before its
            essential rows were replaced, so the rows this functional
            reads are not available as the scheme assembled them.
        """
        if not isinstance(quadrature, TrajectoryQuadratureProtocol):
            raise TypeError(
                "quadrature must satisfy TrajectoryQuadratureProtocol, "
                f"got {type(quadrature).__name__}"
            )
        rule = quadrature.step_residual_rule()
        if rule is None:
            raise ValueError(
                f"{type(self).__name__} needs the scheme's step residuals, "
                "and this scheme provides no step-residual rule (Heun "
                "evaluates F at an unstored predictor state). Use backward "
                "or forward Euler, Crank-Nicolson or implicit midpoint."
            )
        if not rule.is_linear():
            raise ValueError(
                f"{type(self).__name__} supports step-residual rules whose "
                "stage states are linear in the trajectory; this one is "
                "not"
            )
        scheme_mass = rule.mass_matrix()
        if not isinstance(scheme_mass, ConstrainedMassMatrixProtocol):
            raise TypeError(
                f"{type(self).__name__} reads the essential rows of the "
                "scheme's mass as the scheme assembled them, but the "
                f"scheme's mass ({type(scheme_mass).__name__}) does not keep "
                "the mass before those rows were replaced. Build the scheme "
                "through the Galerkin time-integration adapter."
            )
        self._rule = rule
        self._mass = scheme_mass.unconstrained()

    def _require_scheme(
        self,
    ) -> Tuple[StepResidualRuleProtocol[Array], MassMatrixProtocol[Array]]:
        if self._rule is None or self._mass is None:
            raise RuntimeError(
                "time quadrature not set. It is injected by the model "
                "that owns the time-integration scheme; manual drivers "
                "must call set_time_quadrature("
                "stepper.trajectory_quadrature(times)) with the stepper "
                "and times of their solve."
            )
        return self._rule, self._mass

    def _summed_step_residual(self, sol: Array, param: Array) -> Array:
        """``M Y c - sum_j w_j F(y_j, t_j, p)``. Shape: (nstates,)."""
        rule, mass = self._require_scheme()
        if sol.ndim != 2 or sol.shape[0] != self.nstates():
            raise ValueError(
                f"sol must have shape ({self.nstates()}, ntimes), got {sol.shape}"
            )
        if sol.shape[1] != rule.ntimes():
            raise ValueError(
                f"sol has {sol.shape[1]} times but the injected rule "
                f"covers {rule.ntimes()}"
            )
        total = mass.apply(sol @ rule.mass_telescope_coefficients())
        stages = rule.sample(sol)
        stage_times = rule.sample_times()
        weights = rule.weights()
        for jj in range(rule.nsamples()):
            total = total - weights[jj] * self._residual.residual(
                stages[:, jj],
                self._bkd.to_float(stage_times[jj]),
                param[:, 0],
            )
        return total

    def __call__(self, sol: Array, param: Array) -> Array:
        """Evaluate ``Q``. Shape: (nqoi, 1).

        Parameters
        ----------
        sol : Array
            Trajectory. Shape: (nstates, ntimes).
        param : Array
            Parameters. Shape: (nparams, 1).
        """
        return self._row_weights.apply(self._summed_step_residual(sol, param))[:, None]

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(nqoi={self._nqoi}, nstates={self.nstates()})"
