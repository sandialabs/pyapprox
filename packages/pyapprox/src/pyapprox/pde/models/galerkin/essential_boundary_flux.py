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

from typing import Generic, List, Literal, Optional, Tuple, TypeVar

from pyapprox.implicitfunction.observation_terms import RowWeights
from pyapprox.implicitfunction.observed_residual import (
    ObservedResidual,
    pinned_rows_residual,
)
from pyapprox.implicitfunction.pinned_residual import PinnedResidual
from pyapprox.ode.functionals.protocols import TransientFunctionalDerivatives
from pyapprox.ode.mass_matrix import MassMatrixProtocol
from pyapprox.ode.setups import MassShiftStrategy
from pyapprox.ode.step_residual_terms import LastBlock, StepResidualTerm
from pyapprox.ode.stepping_methods import SteppingMethodProtocol
from pyapprox.ode.time_quadrature import (
    StepResidualRuleProtocol,
    TrajectoryQuadratureProtocol,
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
from pyapprox.pde.models.galerkin.transient_system import GalerkinTransientSystem
from pyapprox.pde.parameterizations.derivatives import (
    ParamHVPFn,
    ParamJacobianMatrixFn,
)
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.linalg.operators import FormatProtocol, RowSetProtocol

F = TypeVar("F", bound=FormatProtocol)


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
        return self._bkd.index_update(
            self._bkd.zeros((self._nstates,)), self._dofs, self._weights[qoi]
        )


def essential_boundary_flux(
    bkd: Backend[Array],
    residual: PinnedResidual[Array, F],
    dofs: Array,
    weights: Array,
) -> ObservedResidual[Array, Array]:
    """``Q = -W F(u)`` on selected essential rows, a state term of a steady
    functional.

    Row ``i`` of ``weights`` defines QoI ``i``: unit weights on the
    x-components of a clamped edge give the total x-reaction, weights
    equal to the node coordinate a moment.

    Built from the solved residual itself (the one
    :func:`~pyapprox.pde.models.galerkin.steady_residual.galerkin_steady_residual`
    returns), so ``F``, the parameters it reads,
    the time it is frozen at and the state space are the solve's by
    construction. Its derivatives are the residual's: first order when
    ``F`` has a parameter Jacobian, second order (for any number of QoIs)
    when ``F``'s bundle has its second derivatives; none without a
    parameter Jacobian.

    Parameters
    ----------
    bkd : Backend
        The backend.
    residual : PinnedResidual
        The solved steady residual, its pinned rows the essential ones.
    dofs : Array
        Essential DOFs whose rows are read, unique. Shape: (ndofs,).
    weights : Array
        One row per QoI. Shape: (nqoi, ndofs).

    Raises
    ------
    TypeError
        If ``residual`` is not a ``PinnedResidual``.
    ValueError
        If ``dofs`` is empty, a DOF is repeated, out of range or not
        essential (a natural boundary's DOF, whose row the solve does not
        replace and where ``F`` vanishes), or ``weights`` has the wrong
        shape.
    """
    if not isinstance(residual, PinnedResidual):
        raise TypeError(
            f"residual must be a PinnedResidual, got {type(residual).__name__}"
        )
    # W = -S w^T: the sign makes Q the reaction -F on the selected rows.
    return pinned_rows_residual(
        bkd, residual, -_essential_selection(bkd, residual.pin().rows(), dofs, weights)
    )


def time_integrated_essential_boundary_flux(
    system: GalerkinTransientSystem[Array, F],
    method: SteppingMethodProtocol[Array, Array],
    setups: MassShiftStrategy[Array, Array, F],
    dofs: Array,
    weights: Array,
) -> StepResidualTerm[Array, Array]:
    r"""The flux through essential boundaries over each step, a step term
    of a transient functional.

    The step residual of the unconstrained system, read on selected
    essential rows: on step :math:`n`, entry :math:`i` is
    :math:`\sum_k w_{ik} R_n[d_k]` for the last block :math:`R_n` of the
    unconstrained stepper's residual at the solved trajectory. Summed over
    the steps (unit weights over the steps, the functional's own sum), it
    is the time-integrated flux, as
    :class:`TimeIntegratedEssentialBoundaryFluxFunctional` computes it:
    with unit weights over every essential row, the change of the content
    minus the integrated sources and natural fluxes, exactly as the scheme
    computes them. The solved system's residual vanishes on those rows, so
    it would read no flux.

    Parameters
    ----------
    system : GalerkinTransientSystem
        The system the trajectory was solved on; its unconstrained system
        is stepped by ``method`` here.
    method : SteppingMethodProtocol
        The method the trajectory was solved with.
    setups : MassShiftStrategy
        Sets up the unconstrained stepper's linear solves, for the sweeps'
        second derivatives; its solver pairs with the system's.
    dofs : Array
        Essential DOFs whose rows are read, unique. Shape: (ndofs,).
    weights : Array
        One row per entry of the term. Shape: (nqoi, ndofs).

    Raises
    ------
    TypeError
        If ``method`` does not satisfy ``SteppingMethodProtocol``.
    ValueError
        As for :func:`essential_boundary_flux`.
    """
    if not isinstance(method, SteppingMethodProtocol):
        raise TypeError(
            f"method must satisfy SteppingMethodProtocol, got {type(method).__name__}"
        )
    bkd = system.constraints.bkd()
    rows = system.essential_rows.rows()
    observation = RowWeights(bkd, rows, _essential_selection(bkd, rows, dofs, weights))
    stepper = method.stepper(system.unconstrained, setups)
    return StepResidualTerm.every_step(
        stepper, LastBlock(observation, stepper.residual_space())
    )


def _essential_selection(
    bkd: Backend[Array], rows: RowSetProtocol[Array], dofs: Array, weights: Array
) -> Array:
    """``S w^T``: the weights of each QoI placed on its DOFs' rows, zero
    elsewhere. Shape: (nstates, nqoi).

    Raises
    ------
    ValueError
        If ``dofs`` is empty, a DOF is repeated, out of range or not in
        ``rows``, or ``weights`` is not of shape (nqoi, ndofs).
    """
    dof_list = [bkd.to_int(dofs[ii]) for ii in range(dofs.shape[0])]
    if not dof_list:
        raise ValueError("dofs must name at least one essential DOF")
    if len(set(dof_list)) != len(dof_list):
        raise ValueError("dofs must be unique")
    nstates = rows.descriptor().size
    outside = [dof for dof in dof_list if not 0 <= dof < nstates]
    if outside:
        raise ValueError(f"dofs {outside} are not in [0, {nstates})")
    if weights.ndim != 2 or weights.shape[1] != len(dof_list):
        raise ValueError(
            f"weights must have shape (nqoi, {len(dof_list)}), got "
            f"{tuple(weights.shape)}"
        )
    # The 0/1 selection S of the DOFs, from plain integers (constant data).
    selection = bkd.eye(nstates)[:, dof_list]
    # Zeroing the essential rows leaves a DOF's column nonzero exactly when
    # that DOF is not essential.
    unpinned = rows.zero_rows(selection)
    free = [
        dof
        for column, dof in enumerate(dof_list)
        if bkd.to_float(bkd.max(bkd.abs(unpinned[:, column]))) != 0.0
    ]
    if free:
        raise ValueError(
            f"dofs {free} are not essential; the flux is read from rows "
            "the constraints replace"
        )
    result: Array = bkd.dot(selection, weights.T)
    return result


class TimeIntegratedEssentialBoundaryFluxFunctional(Generic[Array]):
    """The flux through essential boundaries integrated over the solve.

    The time-integrated analog of :func:`essential_boundary_flux`:
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
        self._system = system
        self._dofs = dofs
        self._weights = weights
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
        derivs = residual.derivatives()
        self._param_jacobian_fn: Optional[ParamJacobianMatrixFn[Array]] = (
            derivs.param_jacobian
        )
        self._state_state_fn: Optional[ParamHVPFn[Array]] = derivs.state_state_hvp
        self._state_param_fn: Optional[ParamHVPFn[Array]] = derivs.state_param_hvp
        self._param_state_fn: Optional[ParamHVPFn[Array]] = derivs.param_state_hvp
        self._param_param_fn: Optional[ParamHVPFn[Array]] = derivs.param_param_hvp

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
        """Return the optional derivatives this functional provides.

        First order whenever the raw residual has a parameter Jacobian:
        the gradient of a scalar QoI, the per-column action used by the
        tangent-linear sweep, and per-QoI rows for the adjoint. Second
        order (per-step HVPs) only for a scalar QoI, a residual with all
        four contractions, and a time-diagonal rule: implicit midpoint's
        stage states couple adjacent columns, so its HVPs are absent.
        Which holds depends on the scheme, so the answer exists only once
        its rule is injected.

        Raises
        ------
        RuntimeError
            If no quadrature has been injected.
        """
        rule, _ = self._require_scheme()
        if self._param_jacobian_fn is None:
            return TransientFunctionalDerivatives.none()
        if (
            self._nqoi != 1
            or not rule.is_time_diagonal()
            or self._state_state_fn is None
            or self._state_param_fn is None
            or self._param_state_fn is None
            or self._param_param_fn is None
        ):
            return TransientFunctionalDerivatives.first_order(
                self._state_jacobian,
                self._param_jacobian,
                apply_state_jacobian=self._apply_state_jacobian,
                row_functional=self._row_functional,
            )
        return TransientFunctionalDerivatives.second_order(
            self._state_jacobian,
            self._param_jacobian,
            self._state_state_hvp,
            self._state_param_hvp,
            self._param_state_hvp,
            self._param_param_hvp,
            apply_state_jacobian=self._apply_state_jacobian,
            row_functional=self._row_functional,
        )

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

    def _checked_scheme(
        self, sol: Array
    ) -> Tuple[StepResidualRuleProtocol[Array], MassMatrixProtocol[Array]]:
        """The injected rule and mass, after checking ``sol`` against them."""
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
        return rule, mass

    def _stages(
        self, sol: Array
    ) -> Tuple[StepResidualRuleProtocol[Array], List[Tuple[Array, float, Array]]]:
        """The rule and each stage's (state, time, weight)."""
        rule, _ = self._checked_scheme(sol)
        stages = rule.sample(sol)
        stage_times = rule.sample_times()
        weights = rule.weights()
        return rule, [
            (stages[:, jj], self._bkd.to_float(stage_times[jj]), weights[jj])
            for jj in range(rule.nsamples())
        ]

    def _stage_map(self, rule: StepResidualRuleProtocol[Array], sol: Array) -> Array:
        """The linear stage map ``S``, shape (nsamples, ntimes):
        ``y_hat_j = sum_n S[j, n] y_n``. The adjoint action applied to
        the identity, valid because only linear rules are accepted."""
        return rule.accumulate(sol, self._bkd.eye(rule.nsamples()))

    def _summed_step_residual(self, sol: Array, param: Array) -> Array:
        """``M Y c - sum_j w_j F(y_j, t_j, p)``. Shape: (nstates,)."""
        rule, mass = self._checked_scheme(sol)
        total = mass.apply(sol @ rule.mass_telescope_coefficients())
        for state, time, weight in self._stages(sol)[1]:
            total = total - weight * self._residual.residual(state, time, param[:, 0])
        return total

    def _state_jacobian(self, sol: Array, param: Array) -> Array:
        """``dQ/dY`` for a scalar QoI. Shape: (nstates, ntimes).

        ``(M^T a) c^T`` from the mass term, minus the stage map's adjoint
        of ``w_j J(y_hat_j)^T a`` from the residual terms, with ``a`` the
        selected-row weights.
        """
        if self._nqoi != 1:
            raise ValueError(
                "the state Jacobian is the gradient of a scalar QoI; this "
                f"functional has nqoi={self._nqoi}. Use the per-column action "
                "or the per-QoI rows."
            )
        rule, mass = self._checked_scheme(sol)
        adjoint = self._row_weights.adjoint(0)
        mass_term = self._bkd.outer(
            mass.apply_transpose(adjoint), rule.mass_telescope_coefficients()
        )
        columns = [
            weight
            * self._row_weights.apply(
                self._residual.state_jacobian(state, time, param[:, 0])
            )[0]
            for state, time, weight in self._stages(sol)[1]
        ]
        return mass_term - rule.accumulate(sol, self._bkd.stack(columns, axis=1))

    def _param_jacobian(self, sol: Array, param: Array) -> Array:
        """``-sum_j w_j W dF/dp(y_hat_j)``. Shape: (nqoi, nparams).

        The scheme's mass does not depend on the parameters.
        """
        fn = self._param_jacobian_fn
        if fn is None:
            raise RuntimeError(
                "param_jacobian is unavailable; check derivatives() before calling"
            )
        total = self._bkd.zeros((self._nqoi, self.nparams()))
        for state, time, weight in self._stages(sol)[1]:
            total = total - weight * self._row_weights.apply(
                fn(state, time, param[:, 0])
            )
        return total

    def _apply_state_jacobian(
        self, sol: Array, param: Array, time_idx: int, wmat: Array
    ) -> Array:
        """``(dQ/dy_n) W`` for stored column ``n``. Shape: (nqoi, ncols).

        ``c_n W M V`` from the mass term, minus ``S[j, n] w_j W J_j V``
        over the stages that read column ``n``.
        """
        rule, mass = self._checked_scheme(sol)
        bkd = self._bkd
        coefficient = rule.mass_telescope_coefficients()[time_idx]
        mass_wmat = bkd.stack(
            [mass.apply(wmat[:, kk]) for kk in range(wmat.shape[1])], axis=1
        )
        total = coefficient * self._row_weights.apply(mass_wmat)
        stage_map = self._stage_map(rule, sol)
        for jj, (state, time, weight) in enumerate(self._stages(sol)[1]):
            share = bkd.to_float(stage_map[jj, time_idx])
            if share == 0.0:
                continue
            jacobian = self._residual.state_jacobian(state, time, param[:, 0])
            total = total - share * weight * self._row_weights.apply(jacobian @ wmat)
        return total

    def _row_functional(
        self, qoi_idx: int
    ) -> "TimeIntegratedEssentialBoundaryFluxFunctional[Array]":
        """QoI ``qoi_idx`` as a scalar functional on the same scheme."""
        if not 0 <= qoi_idx < self._nqoi:
            raise ValueError(f"qoi_idx {qoi_idx} out of range [0, {self._nqoi})")
        row = TimeIntegratedEssentialBoundaryFluxFunctional(
            self._residual,
            self._system,
            self._dofs,
            self._weights[qoi_idx : qoi_idx + 1],
        )
        row._rule, row._mass = self._require_scheme()
        return row

    def _step_contraction(
        self,
        fn: Optional[ParamHVPFn[Array]],
        sol: Array,
        param: Array,
        vec: Array,
        time_idx: Optional[int],
        power: int,
        size: int,
    ) -> Array:
        """``-sum_j S[j, n]^power w_j`` times the residual contraction
        with adjoint ``a`` at stage ``j``; over every stage when
        ``time_idx`` is None (the parameter-parameter block). Zero, of
        length ``size``, for a column no stage reads (backward Euler's
        first, forward Euler's last)."""
        if fn is None:
            raise RuntimeError(
                "the HVPs are unavailable; check derivatives() before calling"
            )
        rule, stages = self._stages(sol)
        stage_map = self._stage_map(rule, sol)
        adjoint = self._row_weights.adjoint(0)
        total: Optional[Array] = None
        for jj, (state, time, weight) in enumerate(stages):
            share = (
                1.0
                if time_idx is None
                else self._bkd.to_float(stage_map[jj, time_idx]) ** power
            )
            if share == 0.0:
                continue
            term = share * weight * fn(state, time, param[:, 0], adjoint, vec[:, 0])
            total = term if total is None else total + term
        if total is None:
            return self._bkd.zeros((size, 1))
        return -total[:, None]

    def _state_state_hvp(
        self, sol: Array, param: Array, time_idx: int, wvec: Array
    ) -> Array:
        """``(d^2Q/dy_n^2) w``. Shape: (nstates, 1)."""
        return self._step_contraction(
            self._state_state_fn, sol, param, wvec, time_idx, 2, self.nstates()
        )

    def _state_param_hvp(
        self, sol: Array, param: Array, time_idx: int, vvec: Array
    ) -> Array:
        """``(d^2Q/dy_n dp) v``. Shape: (nstates, 1)."""
        return self._step_contraction(
            self._state_param_fn, sol, param, vvec, time_idx, 1, self.nstates()
        )

    def _param_state_hvp(
        self, sol: Array, param: Array, time_idx: int, wvec: Array
    ) -> Array:
        """``(d^2Q/dp dy_n) w``. Shape: (nparams, 1)."""
        return self._step_contraction(
            self._param_state_fn, sol, param, wvec, time_idx, 1, self.nparams()
        )

    def _param_param_hvp(self, sol: Array, param: Array, vvec: Array) -> Array:
        """``(d^2Q/dp^2) v``. Shape: (nparams, 1)."""
        return self._step_contraction(
            self._param_param_fn, sol, param, vvec, None, 1, self.nparams()
        )

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
